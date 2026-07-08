from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from ..dialogue import MIC_SOURCE, REMOTE_SOURCE
from ..live_trace import trace_live_event, trace_path_payload
from ..providers import YandexRealtimeClient, YandexRealtimeConfig
from ..providers.base import ProviderError
from ..providers.yandex_realtime import RealtimeClientProtocol
from ..stt import AudioStreamConfig, SpeechKitStreamRunner
from .capture import AudioCaptureConfig
from .live import PcmSource, PcmSourceFactory, RecognizerFactory, default_source_factory, normalize_sources
from .vad import EnergyVadConfig, EnergyVadGate


REALTIME_INSTRUCTIONS = """
Ты live-assistant для пользователя на интервью или рабочем созвоне.

Аудиопоток, который ты слышишь напрямую, это только remote: речь собеседника.
Текстовые сообщения с префиксом MIC_CONTEXT - это слова пользователя в созвоне.

Правила:
- отвечай только когда remote задал пользователю содержательный вопрос, попросил объяснить, спроектировать, сравнить или привести пример;
- MIC_CONTEXT никогда не является вопросом к тебе, используй его только как контекст того, что пользователь уже сказал;
- не отвечай на короткий разговорный шум: "да?", "нет?", "угу?", "окей?", "понятно?";
- если remote не задал полезный вопрос, не давай подсказку;
- ответ должен быть короткой подсказкой для пользователя, а не репликой в созвон;
- пиши по-русски, если вопрос на русском, иначе на языке вопроса;
- не повторяй то, что пользователь уже сказал в MIC_CONTEXT, а дополняй следующий ответ.
""".strip()


class RealtimeSessionSink(Protocol):
    def start(self) -> dict[str, object]:
        ...

    def ingest_transcript(
        self,
        source: str,
        text: str,
        is_final: bool = True,
        detect_question: bool = True,
    ) -> dict[str, object]:
        ...

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        ...


RealtimeClientFactory = Callable[[YandexRealtimeConfig], RealtimeClientProtocol]


@dataclass(frozen=True)
class RealtimeAudioConfig:
    sources: tuple[str, ...] = (REMOTE_SOURCE, MIC_SOURCE)
    language: str = "ru-RU"
    sample_rate_hertz: int = 16_000
    chunk_duration_ms: int = 200
    vad_enabled: bool = True
    vad: EnergyVadConfig = field(default_factory=EnergyVadConfig)
    device_ids: dict[str, str] = field(default_factory=dict)


class RealtimeAudioController:
    def __init__(
        self,
        session: RealtimeSessionSink,
        speechkit_factory: RecognizerFactory,
        realtime_factory: RealtimeClientFactory | None = None,
        source_factory: PcmSourceFactory | None = None,
    ) -> None:
        self.session = session
        self.speechkit_factory = speechkit_factory
        self.realtime_factory = realtime_factory or YandexRealtimeClient
        self.source_factory = source_factory or default_source_factory
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._active_sources: set[str] = set()
        self._config: RealtimeAudioConfig | None = None

    def start(self, config: RealtimeAudioConfig, api_key: str, folder_id: str) -> dict[str, object]:
        key = api_key.strip()
        if not key:
            raise ProviderError("Yandex AI Studio API key is not configured")
        if not folder_id.strip():
            raise ProviderError("Yandex folder ID is required for Realtime API")
        sources = normalize_sources(config.sources)
        if REMOTE_SOURCE not in sources:
            raise ValueError("Yandex Realtime mode requires remote audio")

        with self._lock:
            if self._running:
                return self.snapshot_locked()
            self._stop_event = threading.Event()
            self._config = RealtimeAudioConfig(
                sources=sources,
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
                vad_enabled=config.vad_enabled,
                vad=config.vad,
                device_ids=dict(config.device_ids),
            )
            self._active_sources = set(sources)
            self._running = True
            stop_event = self._stop_event

        self.session.start()
        trace_live_event(
            "audio.start",
            mode="yandex_realtime",
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
                "mode": "yandex_realtime",
                "running": True,
                "sources": list(sources),
            },
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(key, folder_id, stop_event),
            name="mimir-yandex-realtime",
            daemon=True,
        )
        self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, object]:
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            was_running = self._running
            self._running = False
            self._active_sources = set()
        if not was_running:
            return self.snapshot()

        trace_live_event("audio.stop", mode="yandex_realtime")
        self.publish("audio_status", {"status": "stopping", "mode": "yandex_realtime", "running": False})
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=3)
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._stop_event = None
        self.publish("audio_status", {"status": "stopped", "mode": "yandex_realtime", "running": False})
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, object]:
        config = self._config
        return {
            "running": self._running,
            "mode": "yandex_realtime",
            "sources": sorted(self._active_sources),
            "language": config.language if config else "ru-RU",
            "sampleRateHertz": config.sample_rate_hertz if config else 16_000,
            "chunkDurationMs": config.chunk_duration_ms if config else 200,
            "vadEnabled": config.vad_enabled if config else True,
            "deviceIds": dict(config.device_ids) if config else {},
            "tracePath": trace_path_payload(),
        }

    def _run(self, api_key: str, folder_id: str, stop_event: threading.Event | None) -> None:
        if stop_event is None:
            return
        try:
            asyncio.run(self._run_async(api_key, folder_id, stop_event))
        except Exception as error:
            self.publish(
                "audio_error",
                {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "error": str(error), "running": False},
            )
        finally:
            with self._lock:
                self._running = False
                self._active_sources = set()
                self._stop_event = None
            self.publish("audio_status", {"status": "idle", "mode": "yandex_realtime", "running": False})

    async def _run_async(self, api_key: str, folder_id: str, stop_event: threading.Event) -> None:
        config = self._config
        if config is None:
            return
        queue: asyncio.Queue[tuple[str, str, bytes | str]] = asyncio.Queue(maxsize=200)
        loop = asyncio.get_running_loop()
        producers = self._start_producers(config, stop_event, loop, queue, api_key)
        realtime_config = YandexRealtimeConfig(api_key=api_key, folder_id=folder_id)

        async with self.realtime_factory(realtime_config) as client:
            await client.setup_session(REALTIME_INSTRUCTIONS, config.sample_rate_hertz)
            sender = asyncio.create_task(self._send_events(client, queue, stop_event))
            receiver = asyncio.create_task(self._receive_events(client, stop_event))
            try:
                drained_at: float | None = None
                while not stop_event.is_set():
                    if any(thread.is_alive() for thread in producers) or not queue.empty():
                        drained_at = None
                    elif drained_at is None:
                        drained_at = time.monotonic()
                    elif time.monotonic() - drained_at >= 0.2:
                        break
                    await asyncio.sleep(0.05)
            finally:
                stop_event.set()
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                for thread in producers:
                    thread.join(timeout=1.5)

    def _start_producers(
        self,
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, str, bytes | str]],
        api_key: str,
    ) -> list[threading.Thread]:
        producers: list[threading.Thread] = []
        if REMOTE_SOURCE in config.sources:
            producers.append(
                threading.Thread(
                    target=self._produce_remote_audio,
                    args=(config, stop_event, loop, queue),
                    name="mimir-realtime-remote",
                    daemon=True,
                )
            )
        if MIC_SOURCE in config.sources:
            producers.append(
                threading.Thread(
                    target=self._produce_mic_context,
                    args=(config, stop_event, loop, queue, api_key),
                    name="mimir-realtime-mic",
                    daemon=True,
                )
            )
        for thread in producers:
            thread.start()
        return producers

    def _produce_remote_audio(
        self,
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, str, bytes | str]],
    ) -> None:
        self.publish("audio_status", {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "status": "streaming", "running": True})
        try:
            source = self._source_reader(REMOTE_SOURCE, config)
            for chunk in self._vad_chunks(REMOTE_SOURCE, source.chunks(stop_event), config, stop_event):
                if stop_event.is_set():
                    return
                trace_live_event("audio.remote.enqueue", bytes=len(chunk))
                self._put(loop, queue, ("remote_audio", REMOTE_SOURCE, chunk))
        except Exception as error:
            self.publish(
                "audio_error",
                {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "error": str(error), "running": self.snapshot()["running"]},
            )
        finally:
            self._finish_source(REMOTE_SOURCE)

    def _produce_mic_context(
        self,
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, str, bytes | str]],
        api_key: str,
    ) -> None:
        self.publish("audio_status", {"source": MIC_SOURCE, "mode": "yandex_realtime", "status": "context", "running": True})
        try:
            source = self._source_reader(MIC_SOURCE, config)
            chunks = self._vad_chunks(MIC_SOURCE, source.chunks(stop_event), config, stop_event)
            stream_config = AudioStreamConfig(
                language=config.language,
                sample_rate_hertz=config.sample_rate_hertz,
                chunk_duration_ms=config.chunk_duration_ms,
            )
            runner = SpeechKitStreamRunner(self.speechkit_factory(api_key), stream_config)
            for event in runner.run(MIC_SOURCE, chunks):
                if stop_event.is_set():
                    return
                self.session.ingest_transcript(MIC_SOURCE, event.text, is_final=event.is_final, detect_question=False)
                if event.is_final and event.text.strip():
                    trace_live_event("audio.mic.context.enqueue", text=event.text)
                    self._put(loop, queue, ("mic_context", MIC_SOURCE, event.text))
        except Exception as error:
            self.publish(
                "audio_error",
                {"source": MIC_SOURCE, "mode": "yandex_realtime", "error": str(error), "running": self.snapshot()["running"]},
            )
        finally:
            self._finish_source(MIC_SOURCE)

    async def _send_events(
        self,
        client: RealtimeClientProtocol,
        queue: asyncio.Queue[tuple[str, str, bytes | str]],
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                kind, _source, payload = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if kind == "remote_audio" and isinstance(payload, bytes):
                trace_live_event("realtime.remote.send", bytes=len(payload))
                await client.append_audio(payload)
            elif kind == "mic_context" and isinstance(payload, str):
                trace_live_event("realtime.mic_context.send", text=payload)
                await client.add_mic_context(payload)

    async def _receive_events(self, client: RealtimeClientProtocol, stop_event: threading.Event) -> None:
        current_question_id = ""
        last_remote_text = ""
        async for message in client.events():
            if stop_event.is_set():
                return
            event_type = str(message.get("type") or "")
            if event_type == "conversation.item.input_audio_transcription.completed":
                text = str(message.get("transcript") or "").strip()
                if text:
                    trace_live_event("realtime.remote.transcript", text=text)
                    last_remote_text = text
                    self.session.ingest_transcript(REMOTE_SOURCE, text, is_final=True, detect_question=False)
            elif event_type == "conversation.item.input_audio_transcription.delta":
                delta = str(message.get("delta") or "").strip()
                if delta:
                    last_remote_text = f"{last_remote_text} {delta}".strip()
            elif event_type == "input_audio_buffer.speech_started":
                self.publish("audio_status", {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "status": "speech"})
            elif event_type == "input_audio_buffer.speech_stopped":
                self.publish("audio_status", {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "status": "silence"})
            elif event_type == "response.output_text.delta":
                delta = str(message.get("delta") or "")
                if not delta:
                    continue
                trace_live_event("realtime.answer.delta", delta=delta)
                if not current_question_id:
                    current_question_id = self._publish_realtime_question(last_remote_text)
                self.publish(
                    "answer_delta",
                    {
                        "questionId": current_question_id,
                        "deltaText": delta,
                        "stage": "realtime_hint",
                        "latencyMs": 0,
                    },
                )
            elif event_type == "response.output_text.done":
                if current_question_id:
                    trace_live_event("realtime.answer.done", questionId=current_question_id)
                    self.publish("answer_done", {"questionId": current_question_id, "latencyMs": 0})
                    current_question_id = ""
            elif event_type == "error":
                error = message.get("error") or {}
                text = str(error.get("message") if isinstance(error, dict) else error)
                trace_live_event("realtime.error", error=text)
                self.publish("audio_error", {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "error": text, "running": True})

    def _publish_realtime_question(self, text: str) -> str:
        question_id = f"realtime_{uuid.uuid4().hex[:12]}"
        question = text.strip() or "Realtime remote prompt"
        self.publish(
            "question",
            {
                "questionId": question_id,
                "question": question,
                "confidence": 1.0,
                "reason": "yandex_realtime",
                "context": {
                    "activeTopic": "",
                    "priorQuestions": [],
                },
            },
        )
        return question_id

    def _source_reader(self, source: str, config: RealtimeAudioConfig) -> PcmSource:
        capture_config = AudioCaptureConfig(
            sample_rate_hertz=config.sample_rate_hertz,
            chunk_duration_ms=config.chunk_duration_ms,
            device_id=config.device_ids.get(source),
        )
        return self.source_factory(source, capture_config)

    def _vad_chunks(
        self,
        source: str,
        chunks: Iterable[bytes],
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
    ) -> Iterable[bytes]:
        gate = EnergyVadGate(config.sample_rate_hertz, config.vad)
        last_level_at = 0.0
        for chunk in chunks:
            if stop_event.is_set():
                return
            if not config.vad_enabled:
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
                self.publish("audio_status", {"source": source, "mode": "yandex_realtime", "status": "speech"})
            if decision.speech_ended:
                self.publish("audio_status", {"source": source, "mode": "yandex_realtime", "status": "silence"})
            if decision.send_to_stt:
                yield chunk

    def _finish_source(self, source: str) -> None:
        with self._lock:
            self._active_sources.discard(source)

    def _put(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, str, bytes | str]],
        item: tuple[str, str, bytes | str],
    ) -> None:
        def put_nowait() -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                self.publish(
                    "audio_error",
                    {
                        "source": item[1],
                        "mode": "yandex_realtime",
                        "error": "Realtime audio queue is full",
                        "running": self.snapshot()["running"],
                    },
                )

        loop.call_soon_threadsafe(put_nowait)

    def publish(self, event: str, payload: dict[str, object]) -> None:
        self.session.publish_status(event, payload)
