from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..dialogue import MIC_SOURCE, REMOTE_SOURCE
from ..live_trace import trace_live_event, trace_path_payload
from ..providers.base import ProviderError
from ..stt import AudioStreamConfig, SpeechKitStreamRunner, StreamingRecognizer
from .capture import AudioCaptureConfig, AudioCaptureError, SoundcardPcmSource
from .applications import ProcessLoopbackPcmSource
from .vad import EnergyVadConfig, EnergyVadGate

if TYPE_CHECKING:
    from .recordings import CallRecordingStore


class SessionSink(Protocol):
    def start(self) -> dict[str, object]:
        ...

    def ingest_transcript(
        self,
        source: str,
        text: str,
        is_final: bool = True,
        detect_question: bool = True,
        is_refinement: bool = False,
    ) -> dict[str, object]:
        ...

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        ...


class PcmSource(Protocol):
    def chunks(self, stop_event: threading.Event) -> Iterator[bytes]:
        ...


RecognizerFactory = Callable[[str], StreamingRecognizer]
PcmSourceFactory = Callable[[str, AudioCaptureConfig], PcmSource]
LiveAudioFallbackStarter = Callable[["LiveAudioConfig", str], object]


@dataclass(frozen=True)
class LiveAudioConfig:
    sources: tuple[str, ...] = (REMOTE_SOURCE, MIC_SOURCE)
    language: str = "ru-RU"
    sample_rate_hertz: int = 16_000
    chunk_duration_ms: int = 200
    vad_enabled: bool = True
    vad: EnergyVadConfig = field(default_factory=EnergyVadConfig)
    device_ids: dict[str, str] = field(default_factory=dict)
    application_process_id: int = 0
    record_testing: bool = False


class LiveAudioController:
    def __init__(
        self,
        session: SessionSink,
        recognizer_factory: RecognizerFactory,
        source_factory: PcmSourceFactory | None = None,
        *,
        mode: str = "speechkit",
        requires_api_key: bool = True,
        fallback_starter: LiveAudioFallbackStarter | None = None,
        recording_store: CallRecordingStore | None = None,
    ) -> None:
        self.session = session
        self.recognizer_factory = recognizer_factory
        self.source_factory = source_factory or default_source_factory
        self.mode = mode
        self.requires_api_key = requires_api_key
        self.fallback_starter = fallback_starter
        self.recording_store = recording_store
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._recognizers: dict[str, StreamingRecognizer] = {}
        self._active_sources: set[str] = set()
        self._running = False
        self._config: LiveAudioConfig | None = None
        self._fallback_started = False
        self._recording_id = ""
        self._recording_sources: set[str] = set()
        self._recording_errors: list[str] = []

    def start(self, config: LiveAudioConfig, api_key: str) -> dict[str, object]:
        key = api_key.strip()
        if self.requires_api_key and not key:
            raise ProviderError("Yandex SpeechKit API key is not configured")
        sources = normalize_sources(config.sources)
        if not sources:
            raise ValueError("at least one audio source is required")

        with self._lock:
            if self._running:
                return self.snapshot_locked()
            self._stop_event = threading.Event()
            self._threads = {}
            self._recognizers = {}
            self._active_sources = set(sources)
            self._config = LiveAudioConfig(
                sources=sources,
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
                vad_enabled=config.vad_enabled,
                vad=config.vad,
                device_ids=dict(config.device_ids),
                application_process_id=config.application_process_id,
                record_testing=config.record_testing,
            )
            self._running = True
            self._fallback_started = False
            self._recording_id = ""
            self._recording_sources = set()
            self._recording_errors = []

        recording_id = self._start_recording()
        self.session.start()
        trace_live_event(
            "audio.start",
            mode=self.mode,
            sources=list(sources),
            language=self._config.language,
            sampleRateHertz=self._config.sample_rate_hertz,
            chunkDurationMs=self._config.chunk_duration_ms,
            vadEnabled=self._config.vad_enabled,
            deviceIds=self._config.device_ids,
            applicationProcessId=self._config.application_process_id,
            recordTesting=self._config.record_testing,
            recordingId=recording_id,
        )
        self.publish(
            "audio_status",
            {
                "status": "starting",
                "mode": self.mode,
                "running": True,
                "sources": list(sources),
            },
        )

        for source in sources:
            thread = threading.Thread(
                target=self._run_source,
                args=(source, key, recording_id),
                name=f"mimir-audio-{source}",
                daemon=True,
            )
            with self._lock:
                self._threads[source] = thread
            thread.start()

        return self.snapshot()

    def stop(self) -> dict[str, object]:
        with self._lock:
            stop_event = self._stop_event
            threads = list(self._threads.values())
            recognizers = list(self._recognizers.values())
            was_running = self._running
            self._running = False
            self._active_sources = set()
        if not was_running and not threads:
            return self.snapshot()

        if was_running:
            trace_live_event("audio.stop", mode=self.mode)
            self.publish("audio_status", {"status": "stopping", "mode": self.mode, "running": False})
        if stop_event is not None:
            stop_event.set()
        stopped = self.wait_for_threads(timeout=2.0)
        if not stopped:
            with self._lock:
                recognizers = list(self._recognizers.values())
            for recognizer in recognizers:
                cancel = getattr(recognizer, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        pass
            stopped = self.wait_for_threads(timeout=1.0)
        if was_running:
            self.publish(
                "audio_status",
                {
                    "status": "stopped" if stopped else "stopping",
                    "mode": self.mode,
                    "running": False,
                },
            )
        return self.snapshot()

    def wait_for_threads(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        with self._lock:
            self._threads = {
                source: thread for source, thread in self._threads.items() if thread.is_alive()
            }
            if not self._threads:
                self._stop_event = None
            return not self._threads

    def has_live_threads(self) -> bool:
        with self._lock:
            return any(thread.is_alive() for thread in self._threads.values())

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, object]:
        config = self._config
        return {
            "running": self._running,
            "mode": self.mode,
            "sources": sorted(self._active_sources),
            "language": config.language if config else "ru-RU",
            "sampleRateHertz": config.sample_rate_hertz if config else 16_000,
            "chunkDurationMs": config.chunk_duration_ms if config else 200,
            "vadEnabled": config.vad_enabled if config else True,
            "deviceIds": dict(config.device_ids) if config else {},
            "applicationProcessId": config.application_process_id if config else 0,
            "recordTesting": config.record_testing if config else False,
            "recordingId": self._recording_id,
            "tracePath": trace_path_payload(),
        }

    def _run_source(self, source: str, api_key: str, recording_id: str) -> None:
        config = self._config
        stop_event = self._stop_event
        if config is None or stop_event is None:
            return

        capture_config = AudioCaptureConfig(
            sample_rate_hertz=config.sample_rate_hertz,
            chunk_duration_ms=config.chunk_duration_ms,
            device_id=config.device_ids.get(source),
            process_id=config.application_process_id if source == REMOTE_SOURCE else None,
        )
        error_text = ""
        try:
            source_reader = self.source_factory(source, capture_config)
            recognizer = self.recognizer_factory(api_key)
            with self._lock:
                self._recognizers[source] = recognizer
            stream_config = AudioStreamConfig(
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
            )
            captured_chunks: Iterable[bytes] = source_reader.chunks(stop_event)
            if recording_id:
                captured_chunks = self._recording_chunks(source, captured_chunks, recording_id)
            chunks = self._stt_chunks(
                source,
                captured_chunks,
                config,
                stop_event,
            )
            self.publish(
                "audio_status",
                {"source": source, "mode": self.mode, "status": "streaming", "running": True},
            )
            runner = SpeechKitStreamRunner(recognizer, stream_config)
            for event in runner.run(source, chunks):
                self.record_stt_result(event.source, event.is_final)
                trace_live_event(
                    "stt.transcript",
                    source=event.source,
                    mode=self.mode,
                    text=event.text,
                    isFinal=event.is_final,
                    endOfUtterance=event.end_of_utterance,
                )
                self.session.ingest_transcript(
                    event.source,
                    event.text,
                    is_final=event.is_final,
                    is_refinement=event.is_refinement,
                )
            if not stop_event.is_set():
                self.publish("audio_status", {"source": source, "mode": self.mode, "status": "done"})
        except Exception as error:
            if not stop_event.is_set():
                error_text = str(error) or error.__class__.__name__
                self.publish(
                    "audio_error",
                    {
                        "source": source,
                        "mode": self.mode,
                        "error": str(error),
                        "running": self.snapshot()["running"],
                    },
                )
                if source == REMOTE_SOURCE:
                    if not self._start_fallback(str(error)):
                        self.mark_degraded("remote_producer", str(error))
        finally:
            with self._lock:
                self._recognizers.pop(source, None)
            self._finish_recording_source(recording_id, source, error_text)
            self._finish_source(source)

    def _start_recording(self) -> str:
        config = self._config
        store = self.recording_store
        if config is None or not config.record_testing or store is None:
            return ""
        try:
            descriptor = store.start(
                application={"processId": config.application_process_id},
                audio_mode=self.mode,
                started_monotonic_ns=time.monotonic_ns(),
            )
            recording_id = str(store.active_recording_id() or "")
            if not recording_id and isinstance(descriptor, dict):
                recording_id = str(
                    descriptor.get("recordingId")
                    or descriptor.get("recording_id")
                    or descriptor.get("id")
                    or ""
                )
            if not recording_id:
                raise RuntimeError("Хранилище не вернуло номер записи")
        except Exception as error:
            self._publish_recording_error(str(error) or error.__class__.__name__)
            return ""
        with self._lock:
            self._recording_id = recording_id
            self._recording_sources = set(config.sources)
            self._recording_errors = []
        self.publish(
            "testing_recording_status",
            {"status": "recording", "recordingId": recording_id},
        )
        return recording_id

    def _recording_chunks(
        self,
        source: str,
        chunks: Iterable[bytes],
        recording_id: str,
    ) -> Iterator[bytes]:
        store = self.recording_store
        recording_failed = store is None
        for chunk in chunks:
            if not recording_failed and store is not None:
                try:
                    written = store.write(
                        source,
                        chunk,
                        captured_at_ns=time.monotonic_ns(),
                    )
                    if not written:
                        raise RuntimeError("Хранилище отклонило звуковой фрагмент")
                except Exception as error:
                    recording_failed = True
                    message = str(error) or error.__class__.__name__
                    self._note_recording_error(recording_id, f"{source}: {message}")
                    self._publish_recording_error(message, recording_id=recording_id, source=source)
            yield chunk

    def _note_recording_error(self, recording_id: str, error: str) -> None:
        with self._lock:
            if recording_id == self._recording_id and error not in self._recording_errors:
                self._recording_errors.append(error)

    def _finish_recording_source(self, recording_id: str, source: str, error: str) -> None:
        if not recording_id:
            return
        if error:
            self._note_recording_error(recording_id, f"{source}: {error}")
        with self._lock:
            if recording_id != self._recording_id:
                return
            self._recording_sources.discard(source)
            if self._recording_sources:
                return
            errors = list(self._recording_errors)
            self._recording_id = ""
            self._recording_errors = []
        final_error = "; ".join(errors)
        store = self.recording_store
        if store is None:
            return
        try:
            store.finish(error=final_error)
        except Exception as finish_error:
            self._publish_recording_error(
                str(finish_error) or finish_error.__class__.__name__,
                recording_id=recording_id,
            )
            return
        self.publish(
            "testing_recording_status",
            {
                "status": "failed" if final_error else "completed",
                "recordingId": recording_id,
                "error": final_error,
            },
        )

    def _publish_recording_error(
        self,
        error: str,
        *,
        recording_id: str = "",
        source: str = "",
    ) -> None:
        trace_live_event(
            "testing.recording_error",
            recordingId=recording_id,
            source=source,
            error=error,
        )
        self.publish(
            "testing_recording_error",
            {
                "recordingId": recording_id,
                "source": source,
                "error": error,
            },
        )

    def _stt_chunks(
        self,
        source: str,
        chunks: Iterable[bytes],
        config: LiveAudioConfig,
        stop_event: threading.Event,
    ) -> Iterator[bytes]:
        gate = EnergyVadGate(config.sample_rate_hertz, config.vad)
        last_level_at = 0.0

        for chunk in chunks:
            if stop_event.is_set():
                return
            if not config.vad_enabled:
                self.record_audio_chunk(source, len(chunk))
                yield chunk
                continue

            decision = gate.process(chunk)
            now = time.monotonic()
            if now - last_level_at >= 0.5:
                self.publish(
                    "audio_level",
                    {
                        "source": source,
                        "rms": decision.rms,
                        "speech": decision.is_speech,
                        "speechThreshold": decision.speech_threshold,
                        "silenceThreshold": decision.silence_threshold,
                    },
                )
                last_level_at = now
            if decision.speech_started:
                self.publish("audio_status", {"source": source, "mode": self.mode, "status": "speech"})
                self.record_audio_speech_started(source)
            if decision.speech_ended:
                self.publish("audio_status", {"source": source, "mode": self.mode, "status": "silence"})
            for stt_chunk in decision.audio_chunks:
                self.record_audio_chunk(source, len(stt_chunk))
                yield stt_chunk

    def _finish_source(self, source: str) -> None:
        should_publish_idle = False
        with self._lock:
            self._active_sources.discard(source)
            self._threads.pop(source, None)
            if not self._threads and not self._running:
                self._stop_event = None
            if not self._active_sources and self._running:
                self._running = False
                self._stop_event = None
                should_publish_idle = True
        if should_publish_idle:
            self.publish("audio_status", {"status": "idle", "mode": self.mode, "running": False})

    def publish(self, event: str, payload: dict[str, object]) -> None:
        self.session.publish_status(event, payload)

    def record_audio_speech_started(self, source: str) -> None:
        recorder = getattr(self.session, "record_audio_speech_started", None)
        if callable(recorder):
            recorder(source)

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        recorder = getattr(self.session, "record_audio_chunk", None)
        if callable(recorder):
            recorder(source, byte_count)

    def record_stt_result(self, source: str, is_final: bool) -> None:
        recorder = getattr(self.session, "record_stt_result", None)
        if callable(recorder):
            recorder(source, is_final)

    def mark_degraded(self, phase: str, error: str) -> None:
        marker = getattr(self.session, "mark_degraded", None)
        if callable(marker):
            marker(phase, error)

    def _start_fallback(self, reason: str) -> bool:
        if self.fallback_starter is None:
            return False
        with self._lock:
            if self._fallback_started or self._config is None:
                return self._fallback_started
            self._fallback_started = True
            config = self._config

        def run_fallback() -> None:
            try:
                self.fallback_starter(config, reason)
            except Exception as error:
                trace_live_event("audio.fallback_error", mode="local_vosk", error=str(error))
                self.publish(
                    "audio_error",
                    {
                        "source": REMOTE_SOURCE,
                        "mode": "local_vosk",
                        "phase": "fallback",
                        "error": str(error),
                        "running": False,
                    },
                )
                self.mark_degraded("fallback", str(error))

        threading.Thread(target=run_fallback, name="mimir-local-audio-fallback", daemon=True).start()
        return True


def default_source_factory(source: str, config: AudioCaptureConfig) -> SoundcardPcmSource | ProcessLoopbackPcmSource:
    if source == REMOTE_SOURCE:
        if not config.process_id:
            raise AudioCaptureError("Выберите приложение созвона в настройках")
        return ProcessLoopbackPcmSource(config.process_id, config)
    return SoundcardPcmSource(source, config)


def normalize_sources(sources: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for source in sources:
        value = source.strip().lower()
        if value in {"remote", "them", "system", "loopback"}:
            clean = REMOTE_SOURCE
        elif value in {"mic", "me", "user"}:
            clean = MIC_SOURCE
        else:
            raise ValueError("audio source must be remote or mic")
        if clean not in normalized:
            normalized.append(clean)
    return tuple(normalized)
