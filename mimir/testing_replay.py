from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .audio.live import LiveAudioConfig, LiveAudioController
from .audio.recordings import CallRecordingStore, RecordedPcmSource, ReplayClock
from .live_trace import sanitize, trace_live_event
from .session_types import normalize_question_key


class ReplayEventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._closed = False
        self._final_turns: dict[str, set[str]] = {"remote": set(), "mic": set()}
        self._final_remote_offsets: list[int] = []
        self._questions: list[tuple[str, int]] = []
        self._duplicate_questions = 0
        self._questions_before_final = 0
        self._answers = 0
        self._unclear = 0
        self._errors: list[str] = []
        self._write_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._writer = threading.Thread(
            target=self._write_events,
            name=f"mimir-replay-events-{path.parent.name}",
            daemon=True,
        )
        self._writer.start()

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        offset_ms = int((time.monotonic() - self._started) * 1_000)
        item = {
            "ts": int(time.time() * 1_000),
            "offsetMs": offset_ms,
            "event": event,
            "payload": sanitize(payload),
        }
        raw = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._closed:
                return
            if event == "transcript" and payload.get("isFinal") is True:
                source = str(payload.get("source") or "")
                if source in self._final_turns:
                    turn_id = str(payload.get("turnId") or "").strip()
                    if not turn_id:
                        turn_id = f"{source}:{payload.get('timestampMs')}:{payload.get('text')}"
                    is_new = turn_id not in self._final_turns[source]
                    self._final_turns[source].add(turn_id)
                    if source == "remote" and is_new:
                        self._final_remote_offsets.append(offset_ms)
            elif event == "question":
                key = normalize_question_key(str(payload.get("question") or ""))
                if key:
                    if any(
                        previous_key == key and offset_ms - previous_offset <= 5_000
                        for previous_key, previous_offset in self._questions[-5:]
                    ):
                        self._duplicate_questions += 1
                    self._questions.append((key, offset_ms))
                    if not any(final_offset <= offset_ms for final_offset in self._final_remote_offsets):
                        self._questions_before_final += 1
            elif event == "answer_done":
                self._answers += 1
            elif event == "transcript_uncertain":
                self._unclear += 1
            elif event in {"audio_error", "stt_error", "answer_error"}:
                error = str(payload.get("error") or event)
                self._errors.append(error)
            self._write_queue.put(raw)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._write_queue.put(None)
        self._writer.join(timeout=5)
        if self._writer.is_alive():
            with self._lock:
                self._errors.append("Журнал повторной проверки не успел закрыться")

    def _write_events(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    raw = self._write_queue.get()
                    if raw is None:
                        handle.flush()
                        return
                    handle.write(raw)
                    handle.write("\n")
        except Exception as error:
            with self._lock:
                self._errors.append(str(error) or error.__class__.__name__)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "remoteTurns": len(self._final_turns["remote"]),
                "micTurns": len(self._final_turns["mic"]),
                "questions": len(self._questions),
                "answers": self._answers,
                "duplicates": self._duplicate_questions,
                "questionsBeforeFinal": self._questions_before_final,
                "unclear": self._unclear,
                "errors": len(self._errors),
                "errorMessages": list(self._errors),
            }


class TestingReplayController:
    def __init__(
        self,
        store: CallRecordingStore,
        session: Any,
        load_config: Callable[[], Any],
        read_secret: Callable[[str], str | None],
        speechkit_factory: Callable[[str], Any],
        local_factory: Callable[[str], Any],
    ) -> None:
        self.store = store
        self.session = session
        self.load_config = load_config
        self.read_secret = read_secret
        self.speechkit_factory = speechkit_factory
        self.local_factory = local_factory
        self._lock = threading.Lock()
        self._controller: LiveAudioController | None = None
        self._event_log: ReplayEventLog | None = None
        self._monitor: threading.Thread | None = None
        self._started = 0.0
        self._starting = False
        self._requested_stop = False
        self._state: dict[str, Any] = self._idle_state()

    def start(self, recording_id: str) -> dict[str, Any]:
        recording = self.store.get(recording_id)
        if recording is None:
            raise ValueError("Запись для проверки не найдена")
        if recording.get("status") == "recording":
            raise ValueError("Нельзя повторять запись, пока она создается")
        if recording.get("status") == "failed":
            raise ValueError("Поврежденную запись нельзя повторить")

        with self._lock:
            if self._starting or self._state["state"] == "running":
                raise ValueError("Повторная проверка уже запущена")
            if self._controller is not None and self._controller.has_live_threads():
                raise ValueError("Предыдущий звуковой поток еще завершается")
            self._controller = None
            self._starting = True

        event_log: ReplayEventLog | None = None
        sink_added = False
        try:
            integrity = self.store.verify(recording_id)
            if not integrity.get("ok"):
                detail = "; ".join(str(item) for item in integrity.get("errors") or [])
                raise ValueError(f"Запись повреждена: {detail or 'неизвестная ошибка'}")
            paths = {
                source: self.store.track_path(recording_id, source)
                for source in ("remote", "mic")
            }
            config = self.load_config()
            local_mode = str(getattr(config, "audio_mode", "speechkit")) == "local_vosk"
            mode = "local_vosk" if local_mode else "speechkit"
            recognizer_factory = self.local_factory if local_mode else self.speechkit_factory
            api_key = "" if local_mode else (
                self.read_secret("yandex_speechkit")
                or self.read_secret("yandex_ai_studio")
                or ""
            )
            if not local_mode and not api_key:
                raise ValueError("Не настроен ключ SpeechKit для повторной проверки")

            clock = ReplayClock(speed=1.0, participants=len(paths))

            def source_factory(source: str, _capture_config: Any) -> RecordedPcmSource:
                return RecordedPcmSource(
                    paths[source],
                    clock,
                    chunk_duration_ms=int(recording.get("chunkDurationMs") or 200),
                )

            run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
            run_directory = self.store.root / recording_id / "runs" / run_id
            event_log = ReplayEventLog(run_directory / "events.jsonl")
            controller = LiveAudioController(
                self.session,
                recognizer_factory,
                source_factory,
                mode=mode,
                requires_api_key=not local_mode,
            )

            self.session.stop()
            self.session.set_answer_provider_override("ollama" if local_mode else None)
            self.session.add_event_sink(event_log)
            sink_added = True
            started = time.monotonic()
            controller.start(
                LiveAudioConfig(
                    sources=("remote", "mic"),
                    language="ru-RU",
                    sample_rate_hertz=int(recording.get("sampleRateHertz") or 16_000),
                    chunk_duration_ms=int(recording.get("chunkDurationMs") or 200),
                    vad_enabled=True,
                    record_testing=False,
                ),
                api_key,
            )
        except Exception:
            if sink_added and event_log is not None:
                try:
                    self.session.remove_event_sink(event_log)
                except Exception:
                    pass
            if event_log is not None:
                event_log.close()
            try:
                self.session.set_answer_provider_override(None)
            except Exception:
                pass
            with self._lock:
                self._starting = False
            raise

        with self._lock:
            self._starting = False
            self._controller = controller
            self._event_log = event_log
            self._started = started
            self._requested_stop = False
            self._state = {
                "state": "running",
                "recordingId": recording_id,
                "runId": run_id,
                "elapsedMs": 0,
                "durationMs": int(recording.get("durationMs") or 0),
                "error": "",
            }
        trace_live_event("testing.replay_start", recordingId=recording_id, runId=run_id)
        monitor = threading.Thread(
            target=self._monitor_replay,
            args=(controller, event_log, recording, integrity, run_id, started),
            name=f"mimir-testing-replay-{run_id}",
            daemon=True,
        )
        with self._lock:
            self._monitor = monitor
        monitor.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            controller = self._controller
            running = self._state["state"] == "running"
            if running:
                self._requested_stop = True
        if controller is not None and (running or controller.has_live_threads()):
            controller.stop()
            self.session.stop()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
            if payload["state"] == "running":
                payload["elapsedMs"] = int((time.monotonic() - self._started) * 1_000)
            return payload

    def is_running(self) -> bool:
        with self._lock:
            return self._state["state"] == "running"

    def has_live_audio(self) -> bool:
        with self._lock:
            controller = self._controller
        return controller is not None and controller.has_live_threads()

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            monitor = self._monitor
        if monitor is not None:
            monitor.join(timeout=timeout)
        return self.snapshot()

    def _monitor_replay(
        self,
        controller: LiveAudioController,
        event_log: ReplayEventLog,
        recording: dict[str, Any],
        integrity: dict[str, Any],
        run_id: str,
        started: float,
    ) -> None:
        try:
            self._run_replay_monitor(
                controller,
                event_log,
                recording,
                integrity,
                run_id,
                started,
            )
        except BaseException as fatal_error:
            message = str(fatal_error) or fatal_error.__class__.__name__
            for action in (
                controller.stop,
                self.session.stop,
                lambda: self.session.remove_event_sink(event_log),
                event_log.close,
                lambda: self.session.set_answer_provider_override(None),
            ):
                try:
                    action()
                except BaseException:
                    pass
            elapsed_ms = int((time.monotonic() - started) * 1_000)
            report = {
                "remoteTurns": 0,
                "micTurns": 0,
                "questions": 0,
                "answers": 0,
                "duplicates": 0,
                "errors": 1,
            }
            with self._lock:
                self._state = {
                    "state": "failed",
                    "recordingId": str(recording.get("id") or ""),
                    "runId": run_id,
                    "elapsedMs": elapsed_ms,
                    "durationMs": int(recording.get("durationMs") or 0),
                    "error": message,
                    "report": report,
                }
                self._controller = controller if controller.has_live_threads() else None
                self._event_log = None
            trace_live_event(
                "testing.replay_done",
                recordingId=recording.get("id"),
                runId=run_id,
                status="failed",
                error=message,
            )

    def _run_replay_monitor(
        self,
        controller: LiveAudioController,
        event_log: ReplayEventLog,
        recording: dict[str, Any],
        integrity: dict[str, Any],
        run_id: str,
        started: float,
    ) -> None:
        operational_errors: list[str] = []

        def requested_stop() -> bool:
            with self._lock:
                return self._requested_stop

        def note_error(error: BaseException | str) -> None:
            message = str(error) or error.__class__.__name__
            if message not in operational_errors:
                operational_errors.append(message)

        try:
            duration_seconds = int(recording.get("durationMs") or 0) / 1_000
            audio_deadline = started + max(90.0, duration_seconds + 90.0)
            while controller.snapshot().get("running"):
                if requested_stop():
                    break
                if time.monotonic() >= audio_deadline:
                    note_error("Звуковой поток не завершился в отведенное время")
                    controller.stop()
                    break
                time.sleep(0.1)
            answer_deadline = time.monotonic() + 120
            while not requested_stop() and self.session.is_processing():
                if time.monotonic() >= answer_deadline:
                    note_error("Подготовка ответов не завершилась за две минуты")
                    break
                time.sleep(0.1)
        except Exception as replay_error:
            note_error(replay_error)

        for action in (
            controller.stop,
            self.session.stop,
            lambda: self.session.remove_event_sink(event_log),
            event_log.close,
            lambda: self.session.set_answer_provider_override(None),
        ):
            try:
                action()
            except Exception as cleanup_error:
                note_error(cleanup_error)
        lingering_audio = controller.has_live_threads()
        if lingering_audio:
            note_error("Старый звуковой поток не завершился; новый запуск временно заблокирован")

        try:
            session_snapshot = self.session.snapshot()
        except Exception as snapshot_error:
            note_error(snapshot_error)
            session_snapshot = {"memory": {"exchanges": []}}
        try:
            summary = event_log.summary()
        except Exception as summary_error:
            note_error(summary_error)
            summary = {
                "remoteTurns": 0,
                "micTurns": 0,
                "questions": 0,
                "answers": 0,
                "duplicates": 0,
                "questionsBeforeFinal": 0,
                "unclear": 0,
                "errors": 0,
                "errorMessages": [],
            }
        for message in operational_errors:
            if message not in summary["errorMessages"]:
                summary["errorMessages"].append(message)
                summary["errors"] = int(summary["errors"]) + 1
        broken_chains = sum(
            1
            for exchange in session_snapshot.get("memory", {}).get("exchanges", [])
            if exchange.get("question") and not exchange.get("hint")
        )
        track_frames = [
            int(track.get("frames") or 0)
            for track in dict(integrity.get("tracks") or {}).values()
            if isinstance(track, dict)
        ]
        aligned = len(track_frames) == 2 and len(set(track_frames)) == 1
        checks = [
            {"id": "audio_integrity", "passed": integrity.get("ok") is True},
            {"id": "source_alignment", "passed": aligned},
            {"id": "no_duplicate_questions", "passed": summary["duplicates"] == 0},
            {
                "id": "questions_after_final_transcript",
                "passed": summary["questionsBeforeFinal"] == 0,
            },
            {"id": "question_hint_answer_chain", "passed": broken_chains == 0},
            {"id": "no_runtime_errors", "passed": summary["errors"] == 0},
        ]
        was_stopped = requested_stop()
        passed = not was_stopped and all(check["passed"] for check in checks)
        report_status = (
            "passed"
            if passed
            else "stopped"
            if was_stopped and not lingering_audio
            else "failed"
        )
        report = {
            "schemaVersion": 1,
            "recordingId": str(recording.get("id") or ""),
            "runId": run_id,
            "status": report_status,
            "startedAtMs": int(time.time() * 1_000 - (time.monotonic() - started) * 1_000),
            "finishedAtMs": int(time.time() * 1_000),
            "elapsedMs": int((time.monotonic() - started) * 1_000),
            "metrics": summary,
            "checks": checks,
        }
        try:
            self.store.write_report(str(recording.get("id") or ""), report, run_id=run_id)
        except Exception as report_error:
            note_error(report_error)
            message = operational_errors[-1]
            if message not in summary["errorMessages"]:
                summary["errorMessages"].append(message)
                summary["errors"] = int(summary["errors"]) + 1
            passed = False
        state = (
            "completed"
            if passed
            else "stopped"
            if was_stopped and not lingering_audio
            else "failed"
        )
        with self._lock:
            self._state = {
                "state": state,
                "recordingId": str(recording.get("id") or ""),
                "runId": run_id,
                "elapsedMs": report["elapsedMs"],
                "durationMs": int(recording.get("durationMs") or 0),
                "error": "; ".join(summary["errorMessages"]) if summary["errors"] else "",
                "report": {
                    "remoteTurns": summary["remoteTurns"],
                    "micTurns": summary["micTurns"],
                    "questions": summary["questions"],
                    "answers": summary["answers"],
                    "duplicates": summary["duplicates"],
                    "errors": summary["errors"],
                },
            }
            self._controller = controller if lingering_audio else None
            self._event_log = None
        trace_live_event(
            "testing.replay_done",
            recordingId=recording.get("id"),
            runId=run_id,
            status=state,
            **summary,
        )

    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "state": "idle",
            "recordingId": None,
            "elapsedMs": 0,
            "durationMs": 0,
            "error": "",
        }
