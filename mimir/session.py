from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from collections.abc import Callable
from typing import Any

from .answer_provider import AnswerProviderGateway
from .config import load_config
from .credentials import read_secret
from .dialogue import MIC_SOURCE, REMOTE_SOURCE, DialogueMemory, DialogueTurn
from .live_trace import trace_live_event
from .models import ChatMessage
from .prompts import build_realtime_session_instructions
from .providers import OllamaClient, YandexAIStudioClient
from .session_answer_flow import SessionAnswerFlow
from .session_metrics import SessionMetrics
from .session_summary import DialogueSummaryCoordinator
from .session_types import (
    AnswerStreamChunk,
    SessionEvent,
    new_id,
    normalize_source,
    parse_model_decision,
    sse_payload,
    wall_ms,
)


SessionEventSink = Callable[[str, dict[str, Any]], None]
MISSING_STT_WARNING_DELAY_SECONDS = 4.0


class SessionManager(SessionAnswerFlow):
    def __init__(self, event_sink: SessionEventSink | None = None) -> None:
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
        self._pending_candidate_timer: threading.Timer | None = None
        self._metric_store = SessionMetrics(trace_live_event)
        self._active_question_id = ""
        self._cancelled_streams = 0
        self._generation = 0
        self._answer_provider_override: str | None = None
        self._current_question: dict[str, Any] | None = None
        self._current_answer_text = ""
        self._speech_sequences: dict[str, int] = {}
        self._ended_speech_sequences: dict[str, list[int]] = {}
        self._finalized_speech_sequences: dict[str, set[int]] = {}
        self._warned_speech_sequences: dict[str, set[int]] = {}
        self._last_final_transcript_sequences: dict[str, int] = {}
        self._active_stt_sequences: dict[str, int] = {}
        self._event_sinks: list[SessionEventSink] = []
        if event_sink is not None:
            self._event_sinks.append(event_sink)
        self._summary = DialogueSummaryCoordinator(
            self._stream_answer,
            self.prompt_config,
            self._apply_summary,
            trace_live_event,
        )
        self._summary.reset(self._generation, self._session_id)

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._state in {"listening", "answering"}:
                return self.snapshot_locked()
            pending = self._pending_candidate_timer
            self._pending_candidate_timer = None
            if pending is not None:
                pending.cancel()
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
            self._speech_sequences = {}
            self._ended_speech_sequences = {}
            self._finalized_speech_sequences = {}
            self._warned_speech_sequences = {}
            self._last_final_transcript_sequences = {}
            self._active_stt_sequences = {}
            self._summary.reset(self._generation, self._session_id)
            self.reset_metrics_locked()
            self._state = "listening"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            trace_live_event("session.start", sessionId=self._session_id, state=self._state)
            return payload

    def stop(self) -> dict[str, Any]:
        self._cancel_answer.set()
        self._cancel_pending_remote_utterance()
        with self._condition:
            self._generation += 1
            self._summary.reset(self._generation, self._session_id)
            self._state = "stopped"
            payload = self.snapshot_locked()
            self.publish_locked("session_state", payload)
            trace_live_event("session.stop", sessionId=self._session_id, state=self._state)
            return payload

    def pause(self) -> dict[str, Any]:
        self._cancel_answer.set()
        self._cancel_pending_remote_utterance()
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
            self._summary.reset(self._generation, self._session_id)
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
        timestamp_ms: int | None = None,
        started_at_ms: int | None = None,
    ) -> dict[str, Any]:
        source = normalize_source(source)
        turn = DialogueTurn(
            source=source,
            text=text,
            is_final=is_final,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else wall_ms(),
            started_at_ms=started_at_ms or 0,
        )
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return {
                    "sessionId": self._session_id,
                    "source": turn.source,
                    "text": turn.text,
                    "isFinal": turn.is_final,
                    "startedAtMs": turn.started_at_ms,
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
                    "startedAtMs": turn.started_at_ms,
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
                "uncertain": turn.uncertain,
                "startedAtMs": turn.started_at_ms,
                "timestampMs": turn.timestamp_ms,
                "operation": update.operation,
                "memoryWindowMs": self._memory.retention_ms,
            }
            self.publish_locked("transcript", payload)
            trace_live_event(
                "session.transcript",
                source=turn.source,
                turnId=turn.turn_id,
                text=turn.text,
                isFinal=turn.is_final,
                operation=update.operation,
                isRefinement=is_refinement,
                detectQuestion=detect_question,
                timestampMs=turn.timestamp_ms,
                startedAtMs=turn.started_at_ms,
            )
            summary_source = (
                self._generation,
                self._session_id,
                self._memory.summary,
                *self._memory.summary_source(),
            ) if turn.is_final else None
            question_utterance_sequence = (
                self._last_final_transcript_sequences.get(turn.source, 0)
                if turn.is_final
                else 0
            )

        if detect_question and turn.source == REMOTE_SOURCE and turn.is_final:
            self._schedule_remote_utterance(
                turn.text,
                turn.timestamp_ms,
                question_utterance_sequence,
            )
        if summary_source is not None:
            self._summary.observe(*summary_source, force=is_refinement)
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
            self._speech_sequences[source] = self._speech_sequences.get(source, 0) + 1
            self._metric_store.record_speech_started(
                self._session_id,
                source,
                self._speech_sequences[source],
            )

    def record_audio_speech_ended(self, source: str, *, trailing_silence_ms: int = 0) -> None:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            speech_sequence = self._speech_sequences.get(source, 0)
            self._metric_store.record_speech_ended(
                self._session_id,
                source,
                trailing_silence_ms,
                speech_sequence,
            )
            generation = self._generation
            session_id = self._session_id
            if speech_sequence > 0:
                ended = self._ended_speech_sequences.setdefault(source, [])
                if speech_sequence not in ended:
                    ended.append(speech_sequence)
        if speech_sequence <= 0:
            return
        timer = threading.Timer(
            MISSING_STT_WARNING_DELAY_SECONDS,
            self._check_missing_stt_utterance,
            args=(generation, session_id, source, speech_sequence),
        )
        timer.daemon = True
        timer.start()

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            self._metric_store.record_audio_chunk(
                self._session_id,
                source,
                byte_count,
                self._speech_sequences.get(source, 0),
            )

    def record_stt_result(
        self,
        source: str,
        is_final: bool,
        *,
        is_refinement: bool = False,
    ) -> int:
        source = normalize_source(source)
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return 0
            if is_refinement:
                return 0
            speech_sequence = self._speech_sequences.get(source, 0)
            if not is_final:
                active_sequence = self._active_stt_sequences.get(source, 0)
                ended = self._ended_speech_sequences.get(source, [])
                finalized = self._finalized_speech_sequences.get(source, set())
                if speech_sequence > 0 and (
                    active_sequence <= 0
                    or active_sequence in finalized
                    or (
                        speech_sequence > active_sequence
                        and active_sequence in ended
                    )
                ):
                    self._active_stt_sequences[source] = speech_sequence
            if is_final:
                ended = self._ended_speech_sequences.get(source, [])
                finalized = self._finalized_speech_sequences.setdefault(source, set())
                warned = self._warned_speech_sequences.setdefault(source, set())
                candidates = [
                    sequence
                    for sequence in ended
                    if sequence not in finalized
                ]
                if speech_sequence > 0 and speech_sequence not in finalized:
                    if speech_sequence not in candidates:
                        candidates.append(speech_sequence)
                active_sequence = self._active_stt_sequences.get(source, 0)
                if active_sequence in warned:
                    self._active_stt_sequences.pop(source, None)
                    active_sequence = 0
                if active_sequence > 0 and active_sequence not in finalized:
                    speech_sequence = active_sequence
                else:
                    speech_sequence = next(
                        (sequence for sequence in candidates if sequence not in warned),
                        candidates[0] if candidates else speech_sequence,
                    )
            self._metric_store.record_stt_result(
                self._session_id,
                source,
                is_final,
                speech_sequence,
            )
            if is_final:
                if speech_sequence > 0:
                    finalized.add(speech_sequence)
                    if self._active_stt_sequences.get(source) == speech_sequence:
                        self._active_stt_sequences.pop(source, None)
                    self._last_final_transcript_sequences[source] = speech_sequence
                    if speech_sequence in warned:
                        warned.remove(speech_sequence)
                        self._metric_store.decrement("missingSttUtterances")
                        self._metric_store.increment("lateSttRecoveries")
                        self.publish_locked(
                            "stt_recovered",
                            {
                                "sessionId": self._session_id,
                                "source": source,
                                "message": "Распознавание вернуло запоздавший текст",
                            },
                        )
            return speech_sequence

    def _check_missing_stt_utterance(
        self,
        generation: int,
        session_id: str,
        source: str,
        speech_sequence: int,
    ) -> None:
        with self._condition:
            if (
                generation != self._generation
                or session_id != self._session_id
                or self._state in {"paused", "stopped"}
                or speech_sequence in self._finalized_speech_sequences.get(source, set())
                or speech_sequence in self._warned_speech_sequences.get(source, set())
            ):
                return
            self._warned_speech_sequences.setdefault(source, set()).add(speech_sequence)
            self._metric_store.increment("missingSttUtterances")
            payload = {
                "sessionId": self._session_id,
                "source": source,
                "message": "Речь была слышна, но распознавание не вернуло текст",
            }
            self.publish_locked("stt_warning", payload)
        trace_live_event(
            "stt.missing_utterance",
            sessionId=session_id,
            source=source,
            speechSequence=speech_sequence,
        )

    def record_external_question(
        self,
        question_id: str,
        question: str,
        *,
        confidence: float = 1.0,
        provider: str = "external",
        source: str = REMOTE_SOURCE,
        utterance_sequence: int = 0,
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
                utterance_sequence=utterance_sequence,
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

    def realtime_instructions(self, base: str) -> str:
        config = self.prompt_config()
        with self._condition:
            dialogue_context = self._memory.realtime_context(max_turns=16, max_chars=3600)
        return build_realtime_session_instructions(
            base,
            config,
            relevance_text=dialogue_context,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self.snapshot_locked()

    def is_processing(self) -> bool:
        with self._condition:
            answer_running = self._answer_thread is not None and self._answer_thread.is_alive()
            candidate_pending = (
                self._pending_candidate_timer is not None
                and self._pending_candidate_timer.is_alive()
            )
            return self._state == "answering" or answer_running or candidate_pending

    def listen(self, after: int = 0, *, recover_gap: bool = False) -> Iterator[SessionEvent]:
        cursor = after
        while True:
            with self._condition:
                snapshot = self._snapshot_if_event_gap_locked(cursor) if recover_gap else None
                if snapshot is not None:
                    event = SessionEvent(
                        int(snapshot["eventSequence"]),
                        "session_snapshot",
                        snapshot,
                    )
                else:
                    event = self.next_event_locked(cursor)
                    if event is None:
                        self._condition.wait(timeout=15)
                        snapshot = (
                            self._snapshot_if_event_gap_locked(cursor)
                            if recover_gap
                            else None
                        )
                        event = (
                            SessionEvent(
                                int(snapshot["eventSequence"]),
                                "session_snapshot",
                                snapshot,
                            )
                            if snapshot is not None
                            else self.next_event_locked(cursor)
                        )
                    if event is None:
                        self._sequence += 1
                        event = SessionEvent(self._sequence, "heartbeat", {"state": self._state})
                        self._events.append(event)
                        self._events = self._events[-300:]
                        self._condition.notify_all()
                cursor = event.sequence
            yield event

    def snapshot_if_event_gap(self, after: int) -> dict[str, Any] | None:
        with self._condition:
            return self._snapshot_if_event_gap_locked(after)

    def _snapshot_if_event_gap_locked(self, after: int) -> dict[str, Any] | None:
        if after > self._sequence:
            return self.snapshot_locked()
        if not self._events:
            return self.snapshot_locked() if after < self._sequence else None
        earliest_sequence = self._events[0].sequence
        if after < earliest_sequence - 1:
            return self.snapshot_locked()
        return None

    def add_event_sink(self, sink: SessionEventSink) -> None:
        with self._condition:
            if sink not in self._event_sinks:
                self._event_sinks.append(sink)

    def remove_event_sink(self, sink: SessionEventSink) -> None:
        with self._condition:
            self._event_sinks = [item for item in self._event_sinks if item is not sink]

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

    def _apply_summary(
        self,
        generation: int,
        session_id: str,
        summary: str,
        through_turn_id: str,
        through_timestamp_ms: int,
        _requested_revision: int,
    ) -> bool:
        with self._condition:
            if generation != self._generation or session_id != self._session_id:
                return False
            if self._state in {"paused", "stopped"}:
                return False
            if self._memory.turn_is_uncertain(through_turn_id):
                trace_live_event(
                    "session.context_summary_skipped",
                    sessionId=self._session_id,
                    reason="uncertain_turn",
                    throughTurnId=through_turn_id,
                )
                return False
            self._memory.set_summary(summary, through_turn_id, through_timestamp_ms)
            self.publish_locked(
                "context_summary",
                {
                    "sessionId": self._session_id,
                    "text": summary,
                    "throughTurnId": through_turn_id,
                },
            )
            return True

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
        utterance_sequence: int = 0,
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
            utterance_sequence=utterance_sequence,
        )

    def add_question_metric_locked(self, metric: dict[str, Any], *, question_ready_at: float) -> None:
        self._metric_store.add_question(self._session_id, metric, question_ready_at=question_ready_at)
        self._active_question_id = str(metric["questionId"])

    def provider_name(self, config: Any | None = None) -> str:
        with self._condition:
            override = self._answer_provider_override
        return self.answer_provider().provider_name(override, config)

    @staticmethod
    def prompt_config() -> Any:
        return load_config()

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
        for sink in tuple(self._event_sinks):
            try:
                sink(event, payload)
            except Exception:
                continue

    def next_event_locked(self, after: int) -> SessionEvent | None:
        for event in self._events:
            if event.sequence > after:
                return event
        return None
