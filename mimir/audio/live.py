from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .. import __version__
from ..dialogue import MIC_SOURCE, REMOTE_SOURCE
from ..live_trace import trace_live_event, trace_path_payload
from ..providers.base import ProviderError
from ..stt import AudioStreamConfig, SpeechKitStreamRunner, StreamingRecognizer
from .applications import ProcessLoopbackPcmSource
from .capture import AudioCaptureConfig, AudioCaptureError, SoundcardPcmSource
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

CAPTURE_BUFFER_DURATION_MS = 10_000


@dataclass(frozen=True)
class CapturedPcmChunk:
    pcm: bytes
    captured_at_ns: int


class _CaptureStopSignal:
    def __init__(self, parent: threading.Event) -> None:
        self._parent = parent
        self._local = threading.Event()

    def is_set(self) -> bool:
        return self._parent.is_set() or self._local.is_set()

    def set(self) -> None:
        self._local.set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._local.wait(min(remaining, 0.05))
            else:
                self._local.wait(0.05)
        return True


class _BufferedPcmCapture:
    def __init__(
        self,
        source: PcmSource,
        stop_event: threading.Event,
        *,
        capacity: int,
        on_capture: Callable[[CapturedPcmChunk], None] | None = None,
        on_drop: Callable[[int, int], None] | None = None,
        on_capture_drop: Callable[[int, int], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capture buffer capacity must be positive")
        self.source = source
        self.stop_event = stop_event
        self.on_capture = on_capture
        self.on_drop = on_drop
        self.on_capture_drop = on_capture_drop
        self._signal = _CaptureStopSignal(stop_event)
        self._queue: queue.Queue[CapturedPcmChunk] = queue.Queue(maxsize=capacity)
        self._capture_queue: queue.Queue[CapturedPcmChunk] | None = (
            queue.Queue(maxsize=capacity) if on_capture is not None else None
        )
        self._capture_drop_lock = threading.Lock()
        self._pending_capture_drop_frames = 0
        self._pending_capture_drop_at_ns = 0
        self._stream_drop_lock = threading.Lock()
        self._pending_stream_drop_frames = 0
        self._pending_stream_drop_at_ns = 0
        self._done = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._produce,
            name="mimir-pcm-capture",
            daemon=True,
        )
        self._capture_thread = threading.Thread(
            target=self._record_captured,
            name="mimir-pcm-recording",
            daemon=True,
        )
        self._started = False

    def chunks(self) -> Iterator[CapturedPcmChunk]:
        if self._started:
            raise RuntimeError("capture stream can only be consumed once")
        self._started = True
        if self._capture_queue is not None:
            self._capture_thread.start()
        self._thread.start()
        try:
            while not self.stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._done.is_set():
                        break
                    continue
                self._report_stream_drops()
                yield item
            self._report_stream_drops()
            if self._error is not None and not self.stop_event.is_set():
                raise self._error
        finally:
            self._signal.set()
            self._thread.join(timeout=0.5)
            if self._capture_queue is not None:
                self._capture_thread.join()
            self._report_stream_drops()

    def _produce(self) -> None:
        try:
            for pcm in self.source.chunks(self._signal):  # type: ignore[arg-type]
                if self._signal.is_set():
                    break
                item = CapturedPcmChunk(bytes(pcm), time.monotonic_ns())
                self._queue_for_recording(item)
                while not self._signal.is_set():
                    try:
                        self._queue.put_nowait(item)
                        break
                    except queue.Full:
                        try:
                            dropped = self._queue.get_nowait()
                        except queue.Empty:
                            continue
                        with self._stream_drop_lock:
                            self._pending_stream_drop_frames += len(dropped.pcm) // 2
                            self._pending_stream_drop_at_ns = dropped.captured_at_ns
        except Exception as error:
            self._error = error
        finally:
            self._done.set()

    def _queue_for_recording(self, item: CapturedPcmChunk) -> None:
        capture_queue = self._capture_queue
        if capture_queue is None:
            return
        while True:
            try:
                capture_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = capture_queue.get_nowait()
                except queue.Empty:
                    continue
                with self._capture_drop_lock:
                    self._pending_capture_drop_frames += len(dropped.pcm) // 2
                    self._pending_capture_drop_at_ns = dropped.captured_at_ns

    def _record_captured(self) -> None:
        capture_queue = self._capture_queue
        if capture_queue is None or self.on_capture is None:
            return
        while not (self._done.is_set() and capture_queue.empty()):
            try:
                item = capture_queue.get(timeout=0.05)
            except queue.Empty:
                self._report_capture_drops()
                continue
            self.on_capture(item)
            self._report_capture_drops()
        self._report_capture_drops()

    def _report_capture_drops(self) -> None:
        callback = self.on_capture_drop
        if callback is None:
            return
        with self._capture_drop_lock:
            frames = self._pending_capture_drop_frames
            captured_at_ns = self._pending_capture_drop_at_ns
            self._pending_capture_drop_frames = 0
            self._pending_capture_drop_at_ns = 0
        if frames:
            callback(frames, captured_at_ns)

    def _report_stream_drops(self) -> None:
        callback = self.on_drop
        if callback is None:
            return
        with self._stream_drop_lock:
            frames = self._pending_stream_drop_frames
            captured_at_ns = self._pending_stream_drop_at_ns
            self._pending_stream_drop_frames = 0
            self._pending_stream_drop_at_ns = 0
        if frames:
            callback(frames, captured_at_ns)


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
    mic_gain: float = 2.0


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
        self._stop_requested_at_ns = 0
        self._capture_drop_frames: dict[str, int] = {}
        self._recording_drop_frames: dict[str, int] = {}

    def start(self, config: LiveAudioConfig, api_key: str) -> dict[str, object]:
        key = api_key.strip()
        if self.requires_api_key and not key:
            raise ProviderError("Yandex SpeechKit API key is not configured")
        sources = normalize_sources(config.sources)
        if not sources:
            raise ValueError("at least one audio source is required")
        if not math.isfinite(config.mic_gain) or config.mic_gain <= 0:
            raise ValueError("microphone gain must be a positive finite number")

        with self._lock:
            if self._running:
                return self.snapshot_locked()
            if any(thread.is_alive() for thread in self._threads.values()):
                raise RuntimeError("Предыдущий звуковой поток еще завершается")
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
                mic_gain=float(config.mic_gain),
            )
            self._running = True
            self._fallback_started = False
            self._recording_id = ""
            self._recording_sources = set()
            self._recording_errors = []
            self._stop_requested_at_ns = 0
            self._capture_drop_frames = {source: 0 for source in sources}
            self._recording_drop_frames = {source: 0 for source in sources}

        self.session.start()
        recording_id = self._start_recording()
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
            micGain=self._config.mic_gain,
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
            if was_running and not self._stop_requested_at_ns:
                self._stop_requested_at_ns = time.monotonic_ns()
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
            "micGain": config.mic_gain if config else 2.0,
            "recordingId": self._recording_id,
            "captureDroppedFrames": dict(self._capture_drop_frames),
            "recordingDroppedFrames": dict(self._recording_drop_frames),
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
            diagnostic_setter = getattr(source_reader, "set_diagnostic_callback", None)
            if callable(diagnostic_setter):
                diagnostic_setter(
                    lambda event, payload: self._capture_diagnostic(
                        source,
                        recording_id,
                        event,
                        payload,
                    )
                )
            recognizer = self.recognizer_factory(api_key)
            with self._lock:
                self._recognizers[source] = recognizer
            stream_config = AudioStreamConfig(
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
            )
            recording_failed = False

            def record_chunk(item: CapturedPcmChunk) -> None:
                nonlocal recording_failed
                if not recording_id or recording_failed:
                    return
                store = self.recording_store
                if store is None:
                    recording_failed = True
                    return
                try:
                    written = store.write(
                        source,
                        item.pcm,
                        captured_at_ns=item.captured_at_ns,
                        recording_id=recording_id,
                    )
                    if not written:
                        raise RuntimeError("Хранилище отклонило звуковой фрагмент")
                except Exception as error:
                    recording_failed = True
                    message = str(error) or error.__class__.__name__
                    self._note_recording_error(recording_id, f"{source}: {message}")
                    self._publish_recording_error(
                        message,
                        recording_id=recording_id,
                        source=source,
                    )

            queue_capacity = max(
                4,
                (CAPTURE_BUFFER_DURATION_MS + config.chunk_duration_ms - 1)
                // config.chunk_duration_ms,
            )
            capture = _BufferedPcmCapture(
                source_reader,
                stop_event,
                capacity=queue_capacity,
                on_capture=record_chunk if recording_id else None,
                on_drop=lambda frames, captured_at_ns: self._capture_buffer_drop(
                    source,
                    recording_id,
                    frames,
                    captured_at_ns,
                ),
                on_capture_drop=(
                    (
                        lambda frames, captured_at_ns: self._recording_buffer_drop(
                            source,
                            recording_id,
                            frames,
                            captured_at_ns,
                        )
                    )
                    if recording_id
                    else None
                ),
            )
            captured_chunks = (item.pcm for item in capture.chunks())
            if source == MIC_SOURCE and config.mic_gain != 1.0:
                captured_chunks = (
                    apply_pcm16_gain(chunk, config.mic_gain) for chunk in captured_chunks
                )
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
                self.record_stt_result(
                    event.source,
                    event.is_final,
                    is_refinement=event.is_refinement,
                )
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
                app_revision=os.environ.get("MIMIR_APP_REVISION", "").strip() or __version__,
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

    def _capture_diagnostic(
        self,
        source: str,
        recording_id: str,
        event: str,
        payload: dict[str, object],
    ) -> None:
        clean_payload = {"source": source, "mode": self.mode, **payload}
        trace_live_event(event, **clean_payload)
        store = self.recording_store
        recorder = getattr(store, "record_event", None)
        if recording_id and callable(recorder):
            recorder(event, **clean_payload)
        status = "failed" if event.endswith(".failed") else "reconnecting"
        self.publish(
            "audio_status",
            {
                "source": source,
                "mode": self.mode,
                "status": status,
                "phase": "application_capture_refresh",
                "running": True,
                **payload,
            },
        )

    def _capture_buffer_drop(
        self,
        source: str,
        recording_id: str,
        frames: int,
        captured_at_ns: int,
    ) -> None:
        config = self._config
        sample_width = 2
        byte_count = frames * sample_width
        with self._lock:
            previous = self._capture_drop_frames.get(source, 0)
            total = previous + frames
            self._capture_drop_frames[source] = total
        payload = {
            "source": source,
            "mode": self.mode,
            "frames": frames,
            "bytes": byte_count,
            "totalDroppedFrames": total,
            "droppedAudioMs": (
                round(total * 1000 / config.sample_rate_hertz) if config is not None else 0
            ),
        }
        trace_live_event("audio.capture_buffer_overflow", **payload)
        store = self.recording_store
        recorder = getattr(store, "record_stream_drop", None)
        if recording_id and callable(recorder):
            recorder(
                source,
                frames,
                captured_at_ns=captured_at_ns,
            )
        if previous == 0:
            message = "Очередь захвата звука переполнилась; старый фрагмент пропущен"
            self.publish(
                "audio_error",
                {
                    **payload,
                    "phase": "capture_buffer_overflow",
                    "error": message,
                    "running": True,
                },
            )
            self.mark_degraded("capture_buffer_overflow", message)

    def _recording_buffer_drop(
        self,
        source: str,
        recording_id: str,
        frames: int,
        captured_at_ns: int,
    ) -> None:
        config = self._config
        with self._lock:
            previous = self._recording_drop_frames.get(source, 0)
            total = previous + frames
            self._recording_drop_frames[source] = total
        payload = {
            "source": source,
            "mode": self.mode,
            "frames": frames,
            "totalDroppedFrames": total,
            "droppedAudioMs": (
                round(total * 1000 / config.sample_rate_hertz) if config is not None else 0
            ),
        }
        trace_live_event("audio.recording_buffer_overflow", **payload)
        store = self.recording_store
        recorder = getattr(store, "record_capture_drop", None)
        if callable(recorder):
            recorder(
                source,
                frames,
                captured_at_ns=captured_at_ns,
            )
        if previous == 0:
            self._publish_recording_error(
                "Очередь записи звука переполнилась; в записи будет отмечен пропуск",
                recording_id=recording_id,
                source=source,
            )

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
            completed = store.finish(
                error=final_error,
                finished_monotonic_ns=self._stop_requested_at_ns or time.monotonic_ns(),
            )
        except Exception as finish_error:
            self._publish_recording_error(
                str(finish_error) or finish_error.__class__.__name__,
                recording_id=recording_id,
            )
            return
        recording_status = (
            str(completed.get("status") or "") if isinstance(completed, dict) else ""
        )
        public_status = {
            "complete": "completed",
            "incomplete": "incomplete",
            "failed": "failed",
        }.get(recording_status, "failed" if final_error else "completed")
        self.publish(
            "testing_recording_status",
            {
                "status": public_status,
                "recordingId": recording_id,
                "error": final_error,
                "recordingStatus": recording_status,
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
                self.record_audio_speech_ended(source, decision.trailing_silence_ms)
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

    def record_audio_speech_ended(self, source: str, trailing_silence_ms: int = 0) -> None:
        recorder = getattr(self.session, "record_audio_speech_ended", None)
        if callable(recorder):
            recorder(source, trailing_silence_ms=trailing_silence_ms)

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        recorder = getattr(self.session, "record_audio_chunk", None)
        if callable(recorder):
            recorder(source, byte_count)

    def record_stt_result(
        self,
        source: str,
        is_final: bool,
        *,
        is_refinement: bool = False,
    ) -> None:
        recorder = getattr(self.session, "record_stt_result", None)
        if callable(recorder):
            recorder(source, is_final, is_refinement=is_refinement)

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


def apply_pcm16_gain(pcm: bytes, gain: float) -> bytes:
    raw = bytes(pcm)
    if not raw or gain == 1.0:
        return raw
    if len(raw) % 2:
        raise ValueError("PCM block must contain complete 16-bit samples")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("PCM gain must be a positive finite number")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = max(-32_768, min(32_767, round(sample * gain)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()
