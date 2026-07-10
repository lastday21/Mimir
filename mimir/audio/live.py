from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from ..dialogue import MIC_SOURCE, REMOTE_SOURCE
from ..live_trace import trace_live_event, trace_path_payload
from ..providers.base import ProviderError
from ..stt import AudioStreamConfig, SpeechKitStreamRunner, StreamingRecognizer
from .capture import AudioCaptureConfig, SoundcardPcmSource
from .vad import EnergyVadConfig, EnergyVadGate


class SessionSink(Protocol):
    def start(self) -> dict[str, object]:
        ...

    def ingest_transcript(self, source: str, text: str, is_final: bool = True) -> dict[str, object]:
        ...

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        ...


class PcmSource(Protocol):
    def chunks(self, stop_event: threading.Event) -> Iterator[bytes]:
        ...


RecognizerFactory = Callable[[str], StreamingRecognizer]
PcmSourceFactory = Callable[[str, AudioCaptureConfig], PcmSource]


@dataclass(frozen=True)
class LiveAudioConfig:
    sources: tuple[str, ...] = (REMOTE_SOURCE, MIC_SOURCE)
    language: str = "ru-RU"
    sample_rate_hertz: int = 16_000
    chunk_duration_ms: int = 200
    vad_enabled: bool = True
    vad: EnergyVadConfig = field(default_factory=EnergyVadConfig)
    device_ids: dict[str, str] = field(default_factory=dict)


class LiveAudioController:
    def __init__(
        self,
        session: SessionSink,
        recognizer_factory: RecognizerFactory,
        source_factory: PcmSourceFactory | None = None,
        *,
        mode: str = "speechkit",
        requires_api_key: bool = True,
    ) -> None:
        self.session = session
        self.recognizer_factory = recognizer_factory
        self.source_factory = source_factory or default_source_factory
        self.mode = mode
        self.requires_api_key = requires_api_key
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._active_sources: set[str] = set()
        self._running = False
        self._config: LiveAudioConfig | None = None

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
            self._active_sources = set(sources)
            self._config = LiveAudioConfig(
                sources=sources,
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
                vad_enabled=config.vad_enabled,
                vad=config.vad,
                device_ids=dict(config.device_ids),
            )
            self._running = True

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
                args=(source, key),
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
            was_running = self._running
            self._running = False
            self._active_sources = set()
        if not was_running:
            return self.snapshot()

        trace_live_event("audio.stop", mode=self.mode)
        self.publish("audio_status", {"status": "stopping", "mode": self.mode, "running": False})
        if stop_event is not None:
            stop_event.set()
        for thread in threads:
            thread.join(timeout=1.5)
        with self._lock:
            self._threads = {
                source: thread for source, thread in self._threads.items() if thread.is_alive()
            }
            if not self._threads:
                self._stop_event = None
        self.publish("audio_status", {"status": "stopped", "mode": self.mode, "running": False})
        return self.snapshot()

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
            "tracePath": trace_path_payload(),
        }

    def _run_source(self, source: str, api_key: str) -> None:
        config = self._config
        stop_event = self._stop_event
        if config is None or stop_event is None:
            return

        capture_config = AudioCaptureConfig(
            sample_rate_hertz=config.sample_rate_hertz,
            chunk_duration_ms=config.chunk_duration_ms,
            device_id=config.device_ids.get(source),
        )
        try:
            source_reader = self.source_factory(source, capture_config)
            recognizer = self.recognizer_factory(api_key)
            stream_config = AudioStreamConfig(
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
            )
            chunks = self._stt_chunks(
                source,
                source_reader.chunks(stop_event),
                config,
                stop_event,
            )
            self.publish(
                "audio_status",
                {"source": source, "mode": self.mode, "status": "streaming", "running": True},
            )
            runner = SpeechKitStreamRunner(recognizer, stream_config)
            for event in runner.run(source, chunks):
                if stop_event.is_set():
                    break
                self.record_stt_result(event.source, event.is_final)
                trace_live_event(
                    "stt.transcript",
                    source=event.source,
                    mode=self.mode,
                    text=event.text,
                    isFinal=event.is_final,
                    endOfUtterance=event.end_of_utterance,
                )
                self.session.ingest_transcript(event.source, event.text, is_final=event.is_final)
            self.publish("audio_status", {"source": source, "mode": self.mode, "status": "done"})
        except Exception as error:
            self.publish(
                "audio_error",
                {"source": source, "mode": self.mode, "error": str(error), "running": self.snapshot()["running"]},
            )
            if source == REMOTE_SOURCE:
                self.mark_degraded("remote_producer", str(error))
        finally:
            self._finish_source(source)

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
                    },
                )
                last_level_at = now
            if decision.speech_started:
                self.publish("audio_status", {"source": source, "mode": self.mode, "status": "speech"})
                self.record_audio_speech_started(source)
            if decision.speech_ended:
                self.publish("audio_status", {"source": source, "mode": self.mode, "status": "silence"})
            if decision.send_to_stt:
                self.record_audio_chunk(source, len(chunk))
                yield chunk

    def _finish_source(self, source: str) -> None:
        should_publish_idle = False
        with self._lock:
            self._active_sources.discard(source)
            self._threads.pop(source, None)
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


def default_source_factory(source: str, config: AudioCaptureConfig) -> SoundcardPcmSource:
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
