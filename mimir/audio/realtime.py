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


REALTIME_RECONNECT_LIMIT = 3
REALTIME_RECONNECT_BASE_DELAY_SECONDS = 0.4
REALTIME_DRAIN_IDLE_SECONDS = 0.2
REALTIME_CONTEXT_TURNS = 12
REALTIME_CONTEXT_CHARS = 1800

REALTIME_INSTRUCTIONS = """
Ты скрытый realtime-помощник пользователя на интервью или рабочем созвоне.
Ты не участник созвона и не отвечаешь собеседнику напрямую.

Ты слышишь напрямую только remote: речь собеседника.
DIALOGUE_CONTEXT - это фоновая история последних реплик с ролями. Это не вопрос к тебе.

Отвечай только если remote задал пользователю содержательный вопрос, попросил объяснить, сравнить,
спроектировать, привести пример или уточнил предыдущую тему.

Если remote не задал полезный вопрос, не генерируй подсказку.

Используй DIALOGUE_CONTEXT, чтобы понять:
- текущую тему;
- что пользователь уже сказал;
- ограничения собеседника;
- является ли вопрос продолжением предыдущей темы.

Не повторяй ответ пользователя. Если пользователь уже начал отвечать, дай только недостающую мысль
или улучшенную формулировку.

Не выдумывай опыт, компании, цифры, проекты и факты. Если факта нет в контексте, дай нейтральную
формулировку, которую пользователь может адаптировать.

Формат:
- коротко, без вступления;
- если вопрос про опыт, пиши от первого лица;
- если вопрос технический, сначала суть, потом максимум 2-3 опорных пункта.
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
        is_refinement: bool = False,
    ) -> dict[str, object]:
        ...

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        ...

    def realtime_context(self, max_turns: int = 12, max_chars: int = 1800) -> str:
        ...


RealtimeClientFactory = Callable[[YandexRealtimeConfig], RealtimeClientProtocol]
RealtimeQueueItem = tuple[str, str, bytes | str]


@dataclass(frozen=True)
class RealtimeAudioConfig:
    sources: tuple[str, ...] = (REMOTE_SOURCE, MIC_SOURCE)
    language: str = "ru-RU"
    sample_rate_hertz: int = 16_000
    chunk_duration_ms: int = 200
    vad_enabled: bool = True
    vad: EnergyVadConfig = field(default_factory=EnergyVadConfig)
    device_ids: dict[str, str] = field(default_factory=dict)


RealtimeFallbackStarter = Callable[[RealtimeAudioConfig, str], object]


class RealtimeAudioController:
    def __init__(
        self,
        session: RealtimeSessionSink,
        speechkit_factory: RecognizerFactory,
        realtime_factory: RealtimeClientFactory | None = None,
        source_factory: PcmSourceFactory | None = None,
        fallback_starter: RealtimeFallbackStarter | None = None,
    ) -> None:
        self.session = session
        self.speechkit_factory = speechkit_factory
        self.realtime_factory = realtime_factory or YandexRealtimeClient
        self.source_factory = source_factory or default_source_factory
        self.fallback_starter = fallback_starter
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._active_sources: set[str] = set()
        self._config: RealtimeAudioConfig | None = None
        self._last_error = ""
        self._last_dialogue_context = ""
        self._fallback_started = False

    def start(self, config: RealtimeAudioConfig, api_key: str, folder_id: str) -> dict[str, object]:
        key = api_key.strip()
        if not key:
            raise ProviderError("Yandex AI Studio API key is not configured")
        if not folder_id.strip():
            raise ProviderError("Yandex folder ID is required for Realtime API")
        sources = normalize_sources(config.sources)
        if REMOTE_SOURCE not in sources:
            raise ValueError("Yandex Realtime mode requires remote audio")

        stale_thread = self._stale_thread()
        if stale_thread is not None:
            stale_thread.join(timeout=3)
            if stale_thread.is_alive():
                raise ProviderError("Previous Yandex Realtime audio session is still stopping")

        with self._lock:
            self._prune_thread_locked()
            if self._running or self._thread is not None:
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
            self._last_error = ""
            self._last_dialogue_context = ""
            self._fallback_started = False
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
            was_running = self._running or bool(thread and thread.is_alive())
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
            if thread.is_alive():
                self._publish_error(REMOTE_SOURCE, "Yandex Realtime audio did not stop within timeout", running=False, phase="stop")
                return self.snapshot()
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
            "lastError": self._last_error,
        }

    def _run(self, api_key: str, folder_id: str, stop_event: threading.Event | None) -> None:
        if stop_event is None:
            return
        try:
            asyncio.run(self._run_async(api_key, folder_id, stop_event))
        except Exception as error:
            self._publish_error(REMOTE_SOURCE, str(error), running=False, phase="run")
        finally:
            with self._lock:
                fallback_started = self._fallback_started
            with self._lock:
                self._running = False
                self._active_sources = set()
                if self._stop_event is stop_event:
                    self._stop_event = None
                if self._thread is threading.current_thread():
                    self._thread = None
            if not fallback_started:
                self.publish("audio_status", {"status": "idle", "mode": "yandex_realtime", "running": False})

    async def _run_async(self, api_key: str, folder_id: str, stop_event: threading.Event) -> None:
        config = self._config
        if config is None:
            return
        queue: asyncio.Queue[RealtimeQueueItem] = asyncio.Queue(maxsize=200)
        loop = asyncio.get_running_loop()
        producers = self._start_producers(config, stop_event, loop, queue, api_key)
        realtime_config = YandexRealtimeConfig(api_key=api_key, folder_id=folder_id)
        reconnects = 0

        try:
            while not stop_event.is_set():
                try:
                    await self._run_realtime_connection(realtime_config, config, queue, producers, stop_event)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if stop_event.is_set():
                        return
                    reconnects += 1
                    self._publish_error(
                        REMOTE_SOURCE,
                        f"Yandex Realtime connection lost: {error}",
                        running=True,
                        phase="reconnect",
                    )
                    if reconnects > REALTIME_RECONNECT_LIMIT:
                        raise ProviderError("Yandex Realtime reconnect limit exceeded") from error
                    delay = min(2.0, REALTIME_RECONNECT_BASE_DELAY_SECONDS * reconnects)
                    trace_live_event("realtime.reconnect", attempt=reconnects, delaySeconds=delay, error=str(error))
                    self.publish(
                        "audio_status",
                        {
                            "source": REMOTE_SOURCE,
                            "mode": "yandex_realtime",
                            "status": "reconnecting",
                            "running": True,
                            "attempt": reconnects,
                        },
                    )
                    await asyncio.sleep(delay)
        finally:
            stop_event.set()
            for thread in producers:
                thread.join(timeout=1.5)

    async def _run_realtime_connection(
        self,
        realtime_config: YandexRealtimeConfig,
        config: RealtimeAudioConfig,
        queue: asyncio.Queue[RealtimeQueueItem],
        producers: list[threading.Thread],
        stop_event: threading.Event,
    ) -> None:
        async with self.realtime_factory(realtime_config) as client:
            await client.setup_session(REALTIME_INSTRUCTIONS, config.sample_rate_hertz)
            with self._lock:
                self._last_dialogue_context = ""
            self._enqueue_dialogue_context_now(queue, "session")
            self.publish(
                "audio_status",
                {"source": REMOTE_SOURCE, "mode": "yandex_realtime", "status": "connected", "running": True},
            )
            sender = asyncio.create_task(self._send_events(client, queue, stop_event), name="mimir-realtime-send")
            receiver = asyncio.create_task(self._receive_events(client, queue, stop_event), name="mimir-realtime-receive")
            try:
                drained_at: float | None = None
                while not stop_event.is_set():
                    if any(thread.is_alive() for thread in producers) or not queue.empty():
                        drained_at = None
                    elif drained_at is None:
                        drained_at = time.monotonic()
                    elif time.monotonic() - drained_at >= REALTIME_DRAIN_IDLE_SECONDS:
                        break
                    self._raise_task_error(sender, "Realtime sender failed")
                    self._raise_task_error(receiver, "Realtime receiver failed")
                    if receiver.done() and any(thread.is_alive() for thread in producers):
                        raise ProviderError("Yandex Realtime websocket closed")
                    await asyncio.sleep(0.05)
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    def _start_producers(
        self,
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[RealtimeQueueItem],
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
        queue: asyncio.Queue[RealtimeQueueItem],
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
            self._publish_error(REMOTE_SOURCE, str(error), running=bool(self.snapshot()["running"]), phase="remote_producer")
        finally:
            self._finish_source(REMOTE_SOURCE)

    def _produce_mic_context(
        self,
        config: RealtimeAudioConfig,
        stop_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[RealtimeQueueItem],
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
                self.record_stt_result(MIC_SOURCE, event.is_final)
                self.session.ingest_transcript(
                    MIC_SOURCE,
                    event.text,
                    is_final=event.is_final,
                    detect_question=False,
                    is_refinement=event.is_refinement,
                )
                if event.is_final and event.text.strip():
                    self._enqueue_dialogue_context(loop, queue, MIC_SOURCE)
        except Exception as error:
            self._publish_error(MIC_SOURCE, str(error), running=bool(self.snapshot()["running"]), phase="mic_producer")
        finally:
            self._finish_source(MIC_SOURCE)

    async def _send_events(
        self,
        client: RealtimeClientProtocol,
        queue: asyncio.Queue[RealtimeQueueItem],
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
            elif kind == "dialogue_context" and isinstance(payload, str):
                trace_live_event("realtime.dialogue_context.send", source=_source, chars=len(payload))
                await client.add_dialogue_context(payload)

    async def _receive_events(
        self,
        client: RealtimeClientProtocol,
        queue: asyncio.Queue[RealtimeQueueItem],
        stop_event: threading.Event,
    ) -> None:
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
                    self.record_stt_result(REMOTE_SOURCE, True)
                    self.session.ingest_transcript(REMOTE_SOURCE, text, is_final=True, detect_question=False)
                    self._enqueue_dialogue_context_now(queue, REMOTE_SOURCE)
            elif event_type == "conversation.item.input_audio_transcription.delta":
                delta = str(message.get("delta") or "").strip()
                if delta:
                    self.record_stt_result(REMOTE_SOURCE, False)
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
                self.record_answer_delta(current_question_id, delta)
                self.record_answer_first_hint(current_question_id, provider="yandex_realtime")
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
                    self.record_answer_done(current_question_id)
                    self.publish("answer_done", {"questionId": current_question_id, "latencyMs": 0})
                    current_question_id = ""
            elif event_type == "error":
                error = message.get("error") or {}
                text = str(error.get("message") if isinstance(error, dict) else error)
                self._publish_error(REMOTE_SOURCE, text, running=True, phase="server_event")

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
        self.record_external_question(question_id, question, provider="yandex_realtime")
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
                self.publish("audio_status", {"source": source, "mode": "yandex_realtime", "status": "speech"})
                self.record_audio_speech_started(source)
            if decision.speech_ended:
                self.publish("audio_status", {"source": source, "mode": "yandex_realtime", "status": "silence"})
            if decision.send_to_stt:
                self.record_audio_chunk(source, len(chunk))
                yield chunk

    def _finish_source(self, source: str) -> None:
        with self._lock:
            self._active_sources.discard(source)

    def _enqueue_dialogue_context(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[RealtimeQueueItem],
        source: str,
    ) -> None:
        context = self._take_dialogue_context_update()
        if not context:
            return
        trace_live_event("audio.dialogue_context.enqueue", source=source, chars=len(context))
        self._put(loop, queue, ("dialogue_context", source, context))

    def _enqueue_dialogue_context_now(self, queue: asyncio.Queue[RealtimeQueueItem], source: str) -> None:
        context = self._take_dialogue_context_update()
        if not context:
            return
        trace_live_event("audio.dialogue_context.enqueue", source=source, chars=len(context))
        self._put_nowait(queue, ("dialogue_context", source, context))

    def _take_dialogue_context_update(self) -> str:
        context = self._latest_dialogue_context()
        if not context:
            return ""
        with self._lock:
            if context == self._last_dialogue_context:
                return ""
            self._last_dialogue_context = context
        return context

    def _latest_dialogue_context(self) -> str:
        builder = getattr(self.session, "realtime_context", None)
        if not callable(builder):
            return ""
        return str(builder(max_turns=REALTIME_CONTEXT_TURNS, max_chars=REALTIME_CONTEXT_CHARS)).strip()

    def _put(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[RealtimeQueueItem],
        item: RealtimeQueueItem,
    ) -> None:
        def put_nowait() -> None:
            self._put_nowait(queue, item)

        loop.call_soon_threadsafe(put_nowait)

    def _put_nowait(self, queue: asyncio.Queue[RealtimeQueueItem], item: RealtimeQueueItem) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            self._publish_error(item[1], "Realtime audio queue is full", running=bool(self.snapshot()["running"]), phase="queue")

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

    def record_external_question(self, question_id: str, question: str, *, provider: str) -> None:
        recorder = getattr(self.session, "record_external_question", None)
        if callable(recorder):
            recorder(question_id, question, provider=provider, source=REMOTE_SOURCE)

    def record_answer_first_hint(self, question_id: str, *, provider: str) -> None:
        recorder = getattr(self.session, "record_answer_first_hint", None)
        if callable(recorder):
            recorder(question_id, provider=provider)

    def record_answer_delta(self, question_id: str, text: str) -> None:
        recorder = getattr(self.session, "record_answer_delta", None)
        if callable(recorder):
            recorder(question_id, text)

    def record_answer_done(self, question_id: str) -> None:
        recorder = getattr(self.session, "record_answer_done", None)
        if callable(recorder):
            recorder(question_id)

    def _publish_error(self, source: str, error: str, *, running: bool, phase: str) -> None:
        with self._lock:
            self._last_error = error
        trace_live_event("audio.error", source=source, mode="yandex_realtime", phase=phase, error=error, running=running)
        self.publish(
            "audio_error",
            {
                "source": source,
                "mode": "yandex_realtime",
                "phase": phase,
                "error": error,
                "running": running,
            },
        )
        if phase in {"server_event", "remote_producer"}:
            with self._lock:
                stop_event = self._stop_event
            if stop_event is not None:
                stop_event.set()
            if not self._start_fallback(error):
                self.mark_degraded(phase, error)
        elif not running and phase == "run":
            if not self._start_fallback(error):
                self.mark_degraded(phase, error)

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

    def mark_degraded(self, phase: str, error: str) -> None:
        marker = getattr(self.session, "mark_degraded", None)
        if callable(marker):
            marker(phase, error)

    def _stale_thread(self) -> threading.Thread | None:
        with self._lock:
            thread = self._thread
            if thread is not None and not thread.is_alive():
                self._thread = None
                self._stop_event = None
                return None
            if thread is not None and thread.is_alive() and not self._running:
                return thread
            return None

    def _prune_thread_locked(self) -> None:
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._stop_event = None

    @staticmethod
    def _raise_task_error(task: asyncio.Task[object], message: str) -> None:
        if not task.done():
            return
        if task.cancelled():
            raise ProviderError(message)
        error = task.exception()
        if error is not None:
            raise ProviderError(f"{message}: {error}") from error
