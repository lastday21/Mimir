from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from .answer_provider import AnswerProviderGateway
from .config import load_config
from .credentials import read_secret
from .dialogue import MIC_SOURCE, REMOTE_SOURCE, DialogueMemory, DialogueTurn
from .live_trace import trace_live_event
from .models import ChatMessage
from .providers import OllamaClient, YandexAIStudioClient
from .session_answer_flow import SessionAnswerFlow
from .session_metrics import SessionMetrics
from .session_types import (
    AnswerStreamChunk,
    SessionEvent,
    new_id,
    normalize_source,
    parse_model_decision,
    sse_payload,
    wall_ms,
)


class SessionManager(SessionAnswerFlow):
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
        self._last_candidate_key = ""
        self._last_candidate_at = 0.0
        self._candidate_sequence = 0
        self._metric_store = SessionMetrics(trace_live_event)
        self._active_question_id = ""
        self._cancelled_streams = 0
        self._generation = 0
        self._answer_provider_override: str | None = None
        self._current_question: dict[str, Any] | None = None
        self._current_answer_text = ""

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._state in {"listening", "answering"}:
                return self.snapshot_locked()
            self._generation += 1
            self._session_id = new_id("session")
            self._memory = DialogueMemory()
            self._cancel_answer = threading.Event()
            self._last_question_key = ""
            self._last_question_at = 0.0
            self._last_candidate_key = ""
            self._last_candidate_at = 0.0
            self._candidate_sequence = 0
            self._current_question = None
            self._current_answer_text = ""
            self.reset_metrics_locked()
            self._state = "listening"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            trace_live_event("session.start", sessionId=self._session_id, state=self._state)
            return payload

    def stop(self) -> dict[str, Any]:
        self._cancel_answer.set()
        with self._condition:
            self._generation += 1
            self._state = "stopped"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            trace_live_event("session.stop", sessionId=self._session_id, state=self._state)
            return payload

    def pause(self) -> dict[str, Any]:
        self._cancel_answer.set()
        with self._condition:
            if self._state not in {"listening", "answering", "degraded"}:
                return self.snapshot_locked()
            self._generation += 1
            self._memory = DialogueMemory()
            self._last_question_key = ""
            self._last_question_at = 0.0
            self._last_candidate_key = ""
            self._last_candidate_at = 0.0
            self._candidate_sequence = 0
            self._current_question = None
            self._current_answer_text = ""
            self.reset_metrics_locked()
            self._state = "paused"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            trace_live_event("session.pause", sessionId=self._session_id, state=self._state)
            return payload

    def ingest_transcript(
        self,
        source: str,
        text: str,
        is_final: bool = True,
        detect_question: bool = True,
        is_refinement: bool = False,
    ) -> dict[str, Any]:
        source = normalize_source(source)
        turn = DialogueTurn(source=source, text=text, is_final=is_final)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return {
                    "sessionId": self._session_id,
                    "source": turn.source,
                    "text": turn.text,
                    "isFinal": turn.is_final,
                    "timestampMs": turn.timestamp_ms,
                    "skipped": True,
                    "reason": self._state,
                }
            if self._state == "idle":
                self._state = "listening"
                self.publish_locked("session_state", self.snapshot_locked())
            update = self._memory.append(turn, refine_latest=is_refinement)
            if update is None:
                return {
                    "sessionId": self._session_id,
                    "source": turn.source,
                    "text": turn.text,
                    "isFinal": turn.is_final,
                    "timestampMs": turn.timestamp_ms,
                    "skipped": True,
                    "reason": "empty",
                }
            turn = update.turn
            if turn.source == MIC_SOURCE and turn.is_final:
                self._memory.record_user_answer(self._active_question_id, turn)
            payload = {
                "sessionId": self._session_id,
                "turnId": turn.turn_id,
                "source": turn.source,
                "text": turn.text,
                "isFinal": turn.is_final,
                "timestampMs": turn.timestamp_ms,
                "operation": update.operation,
                "memoryWindowMs": self._memory.retention_ms,
            }
            self.publish_locked("transcript", payload)
            trace_live_event(
                "session.transcript",
                source=turn.source,
                text=turn.text,
                isFinal=turn.is_final,
                operation=update.operation,
                isRefinement=is_refinement,
                detectQuestion=detect_question,
                timestampMs=turn.timestamp_ms,
            )

        if detect_question and turn.source == REMOTE_SOURCE and turn.is_final and not is_refinement:
            self._consider_remote_utterance(turn.text, turn.timestamp_ms)
        return payload

    def publish_status(self, event: str, payload: dict[str, Any]) -> None:
        self._publish(event, payload)

    def mark_degraded(self, phase: str, error: str) -> None:
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            self._state = "degraded"
            self._metric_store.mark_error(phase, error)
            self.publish_locked("session_state", self.snapshot_locked())
            trace_live_event("session.degraded", sessionId=self._session_id, phase=phase, error=error)

    def set_answer_provider_override(self, provider: str | None) -> None:
        clean = provider.strip().lower() if provider else ""
        with self._condition:
            self._answer_provider_override = clean or None
            self._metric_store.set_value("answerProviderOverride", self._answer_provider_override)
            trace_live_event(
                "session.answer_provider_override",
                sessionId=self._session_id,
                provider=self._answer_provider_override or "",
            )

    def record_audio_speech_started(self, source: str) -> None:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            self._metric_store.record_speech_started(self._session_id, source)

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            self._metric_store.record_audio_chunk(self._session_id, source, byte_count)

    def record_stt_result(self, source: str, is_final: bool) -> None:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            self._metric_store.record_stt_result(self._session_id, source, is_final)

    def record_external_question(
        self,
        question_id: str,
        question: str,
        *,
        confidence: float = 1.0,
        provider: str = "external",
        source: str = REMOTE_SOURCE,
    ) -> None:
        source = normalize_source(source)
        now = time.monotonic()
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            metric = self.build_question_metric_locked(
                question_id=question_id,
                confidence=confidence,
                reason=provider,
                source=source,
                detected_started=now,
                detected_done=now,
                context_started=now,
                context_done=now,
                provider=provider,
            )
            self.add_question_metric_locked(metric, question_ready_at=now)
            self._memory.remember_question(question_id, question, wall_ms())
            self._current_question = {
                "sessionId": self._session_id,
                "questionId": question_id,
                "question": question,
                "confidence": confidence,
                "reason": provider,
                "context": {
                    "activeTopic": self._memory.active_topic,
                    "priorQuestions": self._memory.recent_questions(exclude_id=question_id),
                },
            }
            self._current_answer_text = ""

    def record_answer_delta(self, question_id: str, text: str) -> None:
        if not text:
            return
        with self._condition:
            if question_id != self._active_question_id:
                return
            self._current_answer_text += text
            self._memory.record_hint_delta(question_id, text, wall_ms())

    def record_answer_first_hint(self, question_id: str, *, provider: str | None = None) -> None:
        with self._condition:
            self._metric_store.record_first_hint(question_id, provider=provider)

    def record_answer_done(self, question_id: str) -> None:
        with self._condition:
            self._metric_store.record_answer_done(question_id)

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            return self.metrics_locked()

    def realtime_context(self, max_turns: int = 12, max_chars: int = 1800) -> str:
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return ""
            return self._memory.realtime_context(max_turns=max_turns, max_chars=max_chars)

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

    def _stream_answer(self, messages: list[ChatMessage]) -> Iterator[AnswerStreamChunk | str]:
        with self._condition:
            override = self._answer_provider_override
        yield from self.answer_provider().stream(messages, override)

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            self.publish_locked(event, {"sessionId": self._session_id, **payload})

    def _publish_if_current(self, generation: int, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self.publish_locked(event, {"sessionId": self._session_id, **payload})

    def _record_metric_if_current(self, generation: int, key: str, value: Any) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._metric_store.set_value(key, value)

    def _record_question_field_if_current(self, generation: int, question_id: str, key: str, value: Any) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._metric_store.set_question_field(question_id, key, value)

    def _generation_matches(self, generation: int) -> bool:
        with self._condition:
            return generation == self._generation

    def reset_metrics_locked(self) -> None:
        self._metric_store.reset()
        self._active_question_id = ""
        self._cancelled_streams = 0

    def metrics_locked(self) -> dict[str, Any]:
        return self._metric_store.payload(
            current_question_id=self._active_question_id,
            cancelled_streams=self._cancelled_streams,
            provider_override=self._answer_provider_override,
        )

    def build_question_metric_locked(
        self,
        *,
        question_id: str,
        confidence: float,
        reason: str,
        source: str,
        detected_started: float,
        detected_done: float,
        context_started: float,
        context_done: float,
        provider: str,
    ) -> dict[str, Any]:
        return self._metric_store.build_question(
            session_id=self._session_id,
            question_id=question_id,
            confidence=confidence,
            reason=reason,
            source=source,
            detected_started=detected_started,
            detected_done=detected_done,
            context_started=context_started,
            context_done=context_done,
            provider=provider,
            cancelled_streams=self._cancelled_streams,
        )

    def add_question_metric_locked(self, metric: dict[str, Any], *, question_ready_at: float) -> None:
        self._metric_store.add_question(self._session_id, metric, question_ready_at=question_ready_at)
        self._active_question_id = str(metric["questionId"])

    def provider_name(self, config: Any | None = None) -> str:
        with self._condition:
            override = self._answer_provider_override
        return self.answer_provider().provider_name(override, config)

    @staticmethod
    def answer_provider() -> AnswerProviderGateway:
        return AnswerProviderGateway(
            load_config,
            read_secret,
            YandexAIStudioClient,
            OllamaClient,
            trace_live_event,
        )

    def snapshot_locked(self) -> dict[str, Any]:
        return {
            "sessionId": self._session_id,
            "state": self._state,
            "memory": self._memory.payload(),
            "metrics": self.metrics_locked(),
            "eventSequence": self._sequence,
            "currentQuestion": dict(self._current_question) if self._current_question else None,
            "currentAnswer": {
                "questionId": self._active_question_id,
                "text": self._current_answer_text,
            },
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
