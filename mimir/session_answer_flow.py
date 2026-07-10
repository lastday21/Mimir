from __future__ import annotations

import threading
import time
from typing import Any

from .dialogue import REMOTE_SOURCE, ContextSnapshot
from .live_trace import trace_live_event
from .prompts import build_realtime_messages, build_transcript_decision_messages
from .session_types import (
    AnswerStreamChunk,
    answer_chunk,
    elapsed_ms,
    new_id,
    normalize_question_key,
    parse_model_decision,
)


class SessionAnswerFlow:
    def _consider_remote_utterance(self, text: str, timestamp_ms: int) -> None:
        key = normalize_question_key(text)
        now = time.monotonic()
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            if key and key == self._last_candidate_key and now - self._last_candidate_at < 8:
                trace_live_event("session.utterance_skipped", reason="duplicate", text=text)
                return
            self._last_candidate_key = key
            self._last_candidate_at = now
            self._candidate_sequence += 1
            candidate_sequence = self._candidate_sequence
            generation = self._generation
            context_started = time.monotonic()
            context = self._memory.build_context(
                self._session_id,
                f"candidate_{candidate_sequence}",
                text,
                1.0,
            )
            context_done = time.monotonic()

        cancel = threading.Event()
        thread = threading.Thread(
            target=self._run_model_decision,
            args=(
                text,
                timestamp_ms,
                cancel,
                generation,
                candidate_sequence,
                now,
                context_started,
                context_done,
                context,
            ),
            name=f"mimir-decision-{candidate_sequence}",
            daemon=True,
        )
        self._answer_thread = thread
        thread.start()

    def _run_model_decision(
        self,
        utterance: str,
        timestamp_ms: int,
        cancel: threading.Event,
        generation: int,
        candidate_sequence: int,
        detected_started: float,
        context_started: float,
        context_done: float,
        context: ContextSnapshot,
    ) -> None:
        messages = build_transcript_decision_messages(utterance, context.to_background_text())
        decision_buffer = ""
        question_id = ""
        first_delta_at: float | None = None
        started = time.monotonic()
        last_chunk = AnswerStreamChunk("", self.provider_name())
        try:
            for item in self._stream_answer(messages):
                chunk = answer_chunk(item, self.provider_name())
                last_chunk = chunk
                if not question_id:
                    if not self._candidate_is_current(generation, candidate_sequence):
                        return
                    decision_buffer += chunk.text
                    decision = parse_model_decision(decision_buffer)
                    if decision is None:
                        continue
                    if decision.action == "skip":
                        self._record_skipped_utterance(generation, candidate_sequence, utterance)
                        return
                    question_id = self._activate_model_question(
                        utterance,
                        timestamp_ms,
                        cancel,
                        generation,
                        candidate_sequence,
                        detected_started,
                        time.monotonic(),
                        context_started,
                        context_done,
                        context,
                        chunk.provider,
                    )
                    if not question_id:
                        return
                    decision_buffer = ""
                    chunk = AnswerStreamChunk(
                        decision.text,
                        provider=chunk.provider,
                        fallback_used=chunk.fallback_used,
                        fallback_reason=chunk.fallback_reason,
                    )
                elif cancel.is_set() or not self._generation_matches(generation):
                    self._publish_if_current(generation, "answer_cancelled", {"questionId": question_id})
                    return

                first_delta_at = self._publish_answer_chunk(
                    generation,
                    question_id,
                    chunk,
                    started,
                    first_delta_at,
                )

            if not question_id:
                if not self._candidate_is_current(generation, candidate_sequence):
                    return
                decision = parse_model_decision(decision_buffer, final=True)
                if decision is None or decision.action == "skip":
                    self._record_skipped_utterance(generation, candidate_sequence, utterance)
                    return
                question_id = self._activate_model_question(
                    utterance,
                    timestamp_ms,
                    cancel,
                    generation,
                    candidate_sequence,
                    detected_started,
                    time.monotonic(),
                    context_started,
                    context_done,
                    context,
                    last_chunk.provider,
                )
                if not question_id:
                    return
                first_delta_at = self._publish_answer_chunk(
                    generation,
                    question_id,
                    AnswerStreamChunk(
                        decision.text,
                        provider=last_chunk.provider,
                        fallback_used=last_chunk.fallback_used,
                        fallback_reason=last_chunk.fallback_reason,
                    ),
                    started,
                    first_delta_at,
                )

            self._publish_if_current(
                generation,
                "answer_done",
                {"questionId": question_id, "latencyMs": elapsed_ms(started, time.monotonic())},
            )
            self.record_answer_done(question_id)
        except Exception as error:
            message = str(error) or error.__class__.__name__
            trace_live_event("answer.error", questionId=question_id, error=message)
            if question_id:
                self._publish_if_current(generation, "answer_error", {"questionId": question_id, "error": message})
            with self._condition:
                if generation == self._generation:
                    self._state = "degraded"
                    self._metric_store.mark_error("answer_decision", message)
                    self.publish_locked("session_state", self.snapshot_locked())
            return
        finally:
            if question_id and not cancel.is_set():
                with self._condition:
                    if (
                        generation == self._generation
                        and self._state == "answering"
                        and self._active_question_id == question_id
                    ):
                        self._state = "listening"
                        self.publish_locked("session_state", self.snapshot_locked())

    def _activate_model_question(
        self,
        question: str,
        timestamp_ms: int,
        cancel: threading.Event,
        generation: int,
        candidate_sequence: int,
        detected_started: float,
        detected_done: float,
        context_started: float,
        context_done: float,
        context: ContextSnapshot,
        provider: str,
    ) -> str:
        with self._condition:
            if (
                generation != self._generation
                or candidate_sequence != self._candidate_sequence
                or self._state in {"paused", "stopped"}
            ):
                return ""
            question_id = new_id("question")
            if self._state == "answering":
                self._cancelled_streams += 1
            self._memory.remember_question(question_id, question, timestamp_ms)
            metric = self.build_question_metric_locked(
                question_id=question_id,
                confidence=1.0,
                reason="model",
                source=REMOTE_SOURCE,
                detected_started=detected_started,
                detected_done=detected_done,
                context_started=context_started,
                context_done=context_done,
                provider=provider,
            )
            self.add_question_metric_locked(metric, question_ready_at=detected_done)
            payload = {
                "sessionId": self._session_id,
                "questionId": question_id,
                "question": question,
                "confidence": 1.0,
                "reason": "model",
                "context": {
                    "activeTopic": context.active_topic,
                    "priorQuestions": context.relevant_prior_questions,
                },
            }
            self._current_question = dict(payload)
            self._current_answer_text = ""
            self._last_question_key = normalize_question_key(question)
            self._last_question_at = time.monotonic()
            self.publish_locked("question", payload)
            trace_live_event(
                "session.question",
                questionId=question_id,
                question=question,
                confidence=1.0,
                reason="model",
            )
            self._cancel_answer.set()
            self._cancel_answer = cancel
            self._state = "answering"
            self.publish_locked("session_state", self.snapshot_locked())
            return question_id

    def _publish_answer_chunk(
        self,
        generation: int,
        question_id: str,
        chunk: AnswerStreamChunk,
        started: float,
        first_delta_at: float | None,
    ) -> float | None:
        if not chunk.text:
            return first_delta_at
        self.record_answer_delta(question_id, chunk.text)
        if first_delta_at is None:
            first_delta_at = time.monotonic()
            self._record_metric_if_current(generation, "llmTtfbMs", elapsed_ms(started, first_delta_at))
            self._record_question_field_if_current(
                generation,
                question_id,
                "tLlmTtfbMs",
                elapsed_ms(started, first_delta_at),
            )
            self._record_question_field_if_current(generation, question_id, "provider", chunk.provider)
            if chunk.fallback_used:
                self._record_question_field_if_current(generation, question_id, "fallbackUsed", True)
                if chunk.fallback_reason:
                    self._record_question_field_if_current(
                        generation,
                        question_id,
                        "fallbackReason",
                        chunk.fallback_reason,
                    )
            self.record_answer_first_hint(question_id)
        self._publish_if_current(
            generation,
            "answer_delta",
            {
                "questionId": question_id,
                "deltaText": chunk.text,
                "stage": "full_hint",
                "latencyMs": elapsed_ms(started, time.monotonic()),
                "provider": chunk.provider,
                "fallbackUsed": chunk.fallback_used,
            },
        )
        return first_delta_at

    def _candidate_is_current(self, generation: int, candidate_sequence: int) -> bool:
        with self._condition:
            return generation == self._generation and candidate_sequence == self._candidate_sequence

    def _record_skipped_utterance(self, generation: int, candidate_sequence: int, text: str) -> None:
        with self._condition:
            if generation != self._generation or candidate_sequence != self._candidate_sequence:
                return
            self._metric_store.increment("skippedUtterances")
        trace_live_event("session.utterance_skipped", reason="model", text=text)

    def trigger_question(
        self,
        question: str,
        confidence: float,
        reason: str,
        detected_started: float | None = None,
        detected_done: float | None = None,
        question_timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        key = normalize_question_key(question)
        now = time.monotonic()
        detect_started_at = detected_started if detected_started is not None else now
        detect_done_at = detected_done if detected_done is not None else now
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
            if self._state == "answering":
                self._cancelled_streams += 1
            self._memory.remember_question(question_id, question, question_timestamp_ms)
            context_started = time.monotonic()
            context = self._memory.build_context(self._session_id, question_id, question, confidence)
            context_done = time.monotonic()
            metric = self.build_question_metric_locked(
                question_id=question_id,
                confidence=confidence,
                reason=reason,
                source=REMOTE_SOURCE,
                detected_started=detect_started_at,
                detected_done=detect_done_at,
                context_started=context_started,
                context_done=context_done,
                provider=self.provider_name(),
            )
            self.add_question_metric_locked(metric, question_ready_at=detect_done_at)
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
            self._current_question = dict(payload)
            self._current_answer_text = ""
            self.publish_locked("question", payload)
            trace_live_event(
                "session.question",
                questionId=question_id,
                question=question,
                confidence=confidence,
                reason=reason,
            )
            self._cancel_answer.set()
            self._cancel_answer = threading.Event()
            cancel = self._cancel_answer
            self._state = "answering"
            self._active_question_id = question_id
            self.publish_locked("session_state", self.snapshot_locked())
            generation = self._generation

        thread = threading.Thread(
            target=self._run_answer,
            args=(question_id, question, confidence, cancel, generation),
            name=f"mimir-answer-{question_id}",
            daemon=True,
        )
        self._answer_thread = thread
        thread.start()
        return payload

    def _run_answer(
        self,
        question_id: str,
        question: str,
        confidence: float,
        cancel: threading.Event,
        generation: int,
    ) -> None:
        started = time.monotonic()
        first_delta_at: float | None = None
        try:
            with self._condition:
                if generation != self._generation:
                    return
                context = self._memory.build_context(self._session_id, question_id, question, confidence)
            messages = build_realtime_messages(question, context.to_prompt_text())
            for item in self._stream_answer(messages):
                chunk = answer_chunk(item, self.provider_name())
                if cancel.is_set() or not self._generation_matches(generation):
                    self._publish_if_current(generation, "answer_cancelled", {"questionId": question_id})
                    return
                first_delta_at = self._publish_answer_chunk(
                    generation,
                    question_id,
                    chunk,
                    started,
                    first_delta_at,
                )
            self._publish_if_current(
                generation,
                "answer_done",
                {"questionId": question_id, "latencyMs": elapsed_ms(started, time.monotonic())},
            )
            self.record_answer_done(question_id)
        except Exception as error:
            message = str(error) or error.__class__.__name__
            trace_live_event("answer.error", questionId=question_id, error=message)
            self._publish_if_current(generation, "answer_error", {"questionId": question_id, "error": message})
            with self._condition:
                if generation == self._generation:
                    self._state = "degraded"
                    self._metric_store.mark_error("answer", message)
                    self.publish_locked("session_state", self.snapshot_locked())
            return
        finally:
            if not cancel.is_set():
                with self._condition:
                    if generation == self._generation and self._state == "answering":
                        self._state = "listening"
                        self.publish_locked("session_state", self.snapshot_locked())
