from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .config import load_config
from .credentials import read_secret
from .dialogue import MIC_SOURCE, REMOTE_SOURCE, DialogueMemory, DialogueTurn
from .models import ChatMessage
from .prompts import build_realtime_messages
from .providers import OllamaClient, YandexAIStudioClient
from .providers.base import ProviderError
from .question_detector import detect_questions


@dataclass(frozen=True)
class SessionEvent:
    sequence: int
    event: str
    payload: dict[str, Any]


class SessionManager:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: list[SessionEvent] = []
        self._sequence = 0
        self._session_id = new_id("session")
        self._state = "idle"
        self._memory = DialogueMemory()
        self._answer_thread: threading.Thread | None = None
        self._cancel_answer = threading.Event()
        self._last_question_key = ""
        self._last_question_at = 0.0
        self._metrics: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._state in {"listening", "answering"}:
                return self.snapshot_locked()
            self._session_id = new_id("session")
            self._memory = DialogueMemory()
            self._cancel_answer = threading.Event()
            self._state = "listening"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            return payload

    def stop(self) -> dict[str, Any]:
        self._cancel_answer.set()
        with self._condition:
            self._state = "stopped"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            return payload

    def ingest_transcript(self, source: str, text: str, is_final: bool = True) -> dict[str, Any]:
        source = normalize_source(source)
        turn = DialogueTurn(source=source, text=text, is_final=is_final)
        with self._condition:
            if self._state == "idle":
                self._state = "listening"
                self.publish_locked("session_state", self.snapshot_locked())
            self._memory.append(turn)
            payload = {
                "sessionId": self._session_id,
                "source": turn.source,
                "text": turn.text,
                "isFinal": turn.is_final,
                "timestampMs": turn.timestamp_ms,
            }
            self.publish_locked("transcript", payload)

        if turn.source == REMOTE_SOURCE and turn.is_final:
            self._maybe_trigger_question(turn.text, turn.timestamp_ms)
        return payload

    def manual_question(self, question: str) -> dict[str, Any]:
        text = question.strip()
        if not text:
            raise ValueError("question is required")
        return self.trigger_question(text, confidence=1.0, reason="manual")

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._metrics)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self.snapshot_locked()

    def listen(self, after: int = 0) -> Iterator[SessionEvent]:
        cursor = after
        while True:
            with self._condition:
                event = self.next_event_locked(cursor)
                if event is None:
                    self._condition.wait(timeout=15)
                    event = self.next_event_locked(cursor)
                if event is None:
                    self._sequence += 1
                    event = SessionEvent(self._sequence, "heartbeat", {"state": self._state})
                    self._events.append(event)
                    self._events = self._events[-300:]
                    self._condition.notify_all()
                cursor = event.sequence
            yield event

    def _maybe_trigger_question(self, text: str, timestamp_ms: int) -> None:
        questions = detect_questions(text, timestamp_ms=timestamp_ms, source=REMOTE_SOURCE)
        if not questions:
            return
        best = max(questions, key=lambda item: item.confidence)
        self.trigger_question(best.text, confidence=best.confidence, reason="auto")

    def trigger_question(self, question: str, confidence: float, reason: str) -> dict[str, Any]:
        key = normalize_question_key(question)
        now = time.monotonic()
        with self._condition:
            if key and key == self._last_question_key and now - self._last_question_at < 8:
                return {
                    "sessionId": self._session_id,
                    "question": question,
                    "skipped": True,
                    "reason": "duplicate",
                }
            question_id = new_id("question")
            self._last_question_key = key
            self._last_question_at = now
            self._memory.remember_question(question)
            context = self._memory.build_context(self._session_id, question_id, question, confidence)
            payload = {
                "sessionId": self._session_id,
                "questionId": question_id,
                "question": question,
                "confidence": confidence,
                "reason": reason,
                "context": {
                    "activeTopic": context.active_topic,
                    "priorQuestions": context.relevant_prior_questions,
                },
            }
            self.publish_locked("question", payload)
            self._cancel_answer.set()
            self._cancel_answer = threading.Event()
            cancel = self._cancel_answer
            self._state = "answering"
            self.publish_locked("session_state", self.snapshot_locked())

        thread = threading.Thread(
            target=self._run_answer,
            args=(question_id, question, confidence, cancel),
            name=f"mimir-answer-{question_id}",
            daemon=True,
        )
        self._answer_thread = thread
        thread.start()
        return payload

    def _run_answer(self, question_id: str, question: str, confidence: float, cancel: threading.Event) -> None:
        started = time.monotonic()
        first_delta_at: float | None = None
        try:
            with self._condition:
                context = self._memory.build_context(self._session_id, question_id, question, confidence)
            messages = build_realtime_messages(question, context.to_prompt_text())
            for chunk in self._stream_answer(messages):
                if cancel.is_set():
                    self._publish("answer_cancelled", {"questionId": question_id})
                    return
                if not chunk:
                    continue
                if first_delta_at is None:
                    first_delta_at = time.monotonic()
                    self._record_metric("llmTtfbMs", elapsed_ms(started, first_delta_at))
                self._publish(
                    "answer_delta",
                    {
                        "questionId": question_id,
                        "deltaText": chunk,
                        "stage": "full_hint",
                        "latencyMs": elapsed_ms(started, time.monotonic()),
                    },
                )
            self._publish("answer_done", {"questionId": question_id, "latencyMs": elapsed_ms(started, time.monotonic())})
        except ProviderError as error:
            self._publish("answer_error", {"questionId": question_id, "error": str(error)})
            with self._condition:
                self._state = "degraded"
                self.publish_locked("session_state", self.snapshot_locked())
            return
        finally:
            if not cancel.is_set():
                with self._condition:
                    if self._state == "answering":
                        self._state = "listening"
                        self.publish_locked("session_state", self.snapshot_locked())

    def _stream_answer(self, messages: list[ChatMessage]) -> Iterator[str]:
        config = load_config()
        if config.llm_provider == "ollama":
            yield from OllamaClient(config.ollama_base_url).stream_chat(config.llm_model, messages)
            return
        key = read_secret("yandex_ai_studio") or ""
        yield from YandexAIStudioClient(key, config.yandex_folder_id).stream_chat(config.llm_model, messages)

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            self.publish_locked(event, {"sessionId": self._session_id, **payload})

    def _record_metric(self, key: str, value: Any) -> None:
        with self._condition:
            self._metrics[key] = value
            self._metrics["updatedAt"] = int(time.time() * 1000)

    def snapshot_locked(self) -> dict[str, Any]:
        return {
            "sessionId": self._session_id,
            "state": self._state,
            "memory": self._memory.payload(),
            "metrics": dict(self._metrics),
        }

    def publish_locked(self, event: str, payload: dict[str, Any]) -> None:
        self._sequence += 1
        item = SessionEvent(self._sequence, event, payload)
        self._events.append(item)
        self._events = self._events[-300:]
        self._condition.notify_all()

    def next_event_locked(self, after: int) -> SessionEvent | None:
        for event in self._events:
            if event.sequence > after:
                return event
        return None


def normalize_source(source: str) -> str:
    value = source.strip().lower()
    if value in {"remote", "them", "system"}:
        return REMOTE_SOURCE
    if value in {"mic", "me", "user"}:
        return MIC_SOURCE
    raise ValueError("source must be remote or mic")


def normalize_question_key(text: str) -> str:
    return " ".join(text.lower().strip().rstrip("?!.").split())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def elapsed_ms(started: float, ended: float) -> int:
    return int((ended - started) * 1000)


def sse_payload(event: SessionEvent) -> bytes:
    data = json.dumps(event.payload, ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {data}\n\n".encode("utf-8")
