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
from .dialogue import MIC_SOURCE, REMOTE_SOURCE, ContextSnapshot, DialogueMemory, DialogueTurn
from .live_trace import trace_live_event
from .models import ChatMessage
from .ollama_fallback import select_preferred_model
from .prompts import build_realtime_messages, build_transcript_decision_messages
from .providers import OllamaClient, YandexAIStudioClient
from .providers.base import ProviderError


@dataclass(frozen=True)
class SessionEvent:
    sequence: int
    event: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AnswerStreamChunk:
    text: str
    provider: str
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class ModelDecision:
    action: str
    text: str = ""


MODEL_SKIP_MARKER = "[[SKIP]]"
MODEL_ANSWER_MARKER = "[[ANSWER]]"


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
        self._last_candidate_key = ""
        self._last_candidate_at = 0.0
        self._candidate_sequence = 0
        self._metrics: dict[str, Any] = {}
        self._source_metrics: dict[str, dict[str, Any]] = {}
        self._question_metrics: list[dict[str, Any]] = []
        self._question_metric_by_id: dict[str, dict[str, Any]] = {}
        self._question_runtime: dict[str, dict[str, float]] = {}
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
            self._metrics["lastError"] = error
            self._metrics["errorPhase"] = phase
            self._metrics["updatedAt"] = int(time.time() * 1000)
            self.publish_locked("session_state", self.snapshot_locked())
            trace_live_event("session.degraded", sessionId=self._session_id, phase=phase, error=error)

    def set_answer_provider_override(self, provider: str | None) -> None:
        clean = provider.strip().lower() if provider else ""
        with self._condition:
            self._answer_provider_override = clean or None
            self._metrics["answerProviderOverride"] = self._answer_provider_override
            self._metrics["updatedAt"] = int(time.time() * 1000)
            trace_live_event(
                "session.answer_provider_override",
                sessionId=self._session_id,
                provider=self._answer_provider_override or "",
            )

    def record_audio_speech_started(self, source: str) -> None:
        source = normalize_source(source)
        now = time.monotonic()
        now_ms = wall_ms()
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            metric = {
                "source": source,
                "speechStartedAtMs": now_ms,
                "_speechStartedAt": now,
            }
            self._source_metrics[source] = metric
            self.trace_metric_stage_locked(source=source, stage="speech_started", elapsed_ms=0)

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        source = normalize_source(source)
        now = time.monotonic()
        now_ms = wall_ms()
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            metric = self._source_metrics.setdefault(
                source,
                {
                    "source": source,
                    "speechStartedAtMs": now_ms,
                    "_speechStartedAt": now,
                },
            )
            if "audioChunkAtMs" in metric:
                return
            started = float(metric.get("_speechStartedAt", now))
            elapsed = elapsed_ms(started, now)
            metric["audioChunkAtMs"] = now_ms
            metric["_audioChunkAt"] = now
            metric["audioChunkBytes"] = byte_count
            metric["tAudioChunkMs"] = elapsed
            self.trace_metric_stage_locked(source=source, stage="audio_chunk", elapsed_ms=elapsed, bytes=byte_count)

    def record_stt_result(self, source: str, is_final: bool) -> None:
        source = normalize_source(source)
        now = time.monotonic()
        now_ms = wall_ms()
        with self._condition:
            if self._state in {"paused", "stopped"}:
                return
            metric = self._source_metrics.setdefault(
                source,
                {
                    "source": source,
                    "speechStartedAtMs": now_ms,
                    "_speechStartedAt": now,
                    "audioChunkAtMs": now_ms,
                    "_audioChunkAt": now,
                    "tAudioChunkMs": 0,
                },
            )
            audio_at = float(metric.get("_audioChunkAt", now))
            if not is_final and "tSttInterimMs" not in metric:
                elapsed = elapsed_ms(audio_at, now)
                metric["sttInterimAtMs"] = now_ms
                metric["tSttInterimMs"] = elapsed
                self.trace_metric_stage_locked(source=source, stage="stt_interim", elapsed_ms=elapsed)
                return
            if is_final:
                elapsed = elapsed_ms(audio_at, now)
                metric["sttFinalAtMs"] = now_ms
                metric["_sttFinalAt"] = now
                metric["tSttFinalMs"] = elapsed
                self.trace_metric_stage_locked(source=source, stage="stt_final", elapsed_ms=elapsed)

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
                question=question,
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
        now = time.monotonic()
        with self._condition:
            metric = self._question_metric_by_id.get(question_id)
            runtime = self._question_runtime.get(question_id)
            if metric is None or runtime is None or "tFirstHintMs" in metric:
                return
            metric["firstHintAtMs"] = wall_ms()
            metric["tFirstHintMs"] = elapsed_ms(runtime["questionReadyAt"], now)
            if provider:
                metric["provider"] = provider
            self.trace_question_metric_locked(metric)

    def record_answer_done(self, question_id: str) -> None:
        now = time.monotonic()
        with self._condition:
            metric = self._question_metric_by_id.get(question_id)
            runtime = self._question_runtime.get(question_id)
            if metric is None or runtime is None:
                return
            metric["answerDoneAtMs"] = wall_ms()
            metric["tAnswerDoneMs"] = elapsed_ms(runtime["questionReadyAt"], now)
            self.trace_question_metric_locked(metric)

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
                    self._metrics["lastError"] = message
                    self._metrics["errorPhase"] = "answer_decision"
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
                question=question,
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
            self._metrics["skippedUtterances"] = int(self._metrics.get("skippedUtterances", 0)) + 1
            self._metrics["updatedAt"] = wall_ms()
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
                question=question,
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
                    self._metrics["lastError"] = message
                    self._metrics["errorPhase"] = "answer"
                    self.publish_locked("session_state", self.snapshot_locked())
            return
        finally:
            if not cancel.is_set():
                with self._condition:
                    if generation == self._generation and self._state == "answering":
                        self._state = "listening"
                        self.publish_locked("session_state", self.snapshot_locked())

    def _stream_answer(self, messages: list[ChatMessage]) -> Iterator[AnswerStreamChunk | str]:
        config = load_config()
        provider = self.provider_name(config)
        if provider == "ollama":
            client = OllamaClient(config.ollama_base_url)
            model = self.ollama_model(config, client)
            for chunk in client.stream_chat(model, messages):
                yield AnswerStreamChunk(chunk, provider="ollama")
            return
        key = read_secret("yandex_ai_studio") or ""
        primary_started = False
        try:
            for chunk in YandexAIStudioClient(key, config.yandex_folder_id).stream_chat(config.llm_model, messages):
                if chunk:
                    primary_started = True
                yield AnswerStreamChunk(chunk, provider="yandex_ai_studio")
        except ProviderError as error:
            if primary_started:
                raise
            yield from self._stream_ollama_fallback(messages, error)

    def _stream_ollama_fallback(
        self,
        messages: list[ChatMessage],
        primary_error: ProviderError,
    ) -> Iterator[AnswerStreamChunk]:
        config = load_config()
        client = OllamaClient(config.ollama_base_url)
        try:
            preferred = select_preferred_model(client.list_models())
        except ProviderError as error:
            raise ProviderError(f"Yandex AI Studio failed: {primary_error}. Ollama fallback failed: {error}") from error
        if preferred is None:
            raise ProviderError(f"Yandex AI Studio failed: {primary_error}. Ollama fallback has no local models")
        reason = str(primary_error)
        trace_live_event(
            "answer.fallback",
            fromProvider="yandex_ai_studio",
            toProvider="ollama",
            model=preferred.id,
            reason=reason,
        )
        try:
            for chunk in client.stream_chat(preferred.id, messages):
                yield AnswerStreamChunk(chunk, provider="ollama", fallback_used=True, fallback_reason=reason)
        except ProviderError as error:
            raise ProviderError(f"Yandex AI Studio failed: {primary_error}. Ollama fallback failed: {error}") from error

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            self.publish_locked(event, {"sessionId": self._session_id, **payload})

    def _publish_if_current(self, generation: int, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self.publish_locked(event, {"sessionId": self._session_id, **payload})

    def _record_metric(self, key: str, value: Any) -> None:
        with self._condition:
            self._metrics[key] = value
            self._metrics["updatedAt"] = int(time.time() * 1000)

    def _record_metric_if_current(self, generation: int, key: str, value: Any) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._metrics[key] = value
            self._metrics["updatedAt"] = int(time.time() * 1000)

    def _record_question_field_if_current(self, generation: int, question_id: str, key: str, value: Any) -> None:
        with self._condition:
            if generation != self._generation:
                return
            metric = self._question_metric_by_id.get(question_id)
            if metric is not None:
                metric[key] = value

    def _generation_matches(self, generation: int) -> bool:
        with self._condition:
            return generation == self._generation

    def reset_metrics_locked(self) -> None:
        self._metrics = {}
        self._source_metrics = {}
        self._question_metrics = []
        self._question_metric_by_id = {}
        self._question_runtime = {}
        self._active_question_id = ""
        self._cancelled_streams = 0

    def metrics_locked(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["sources"] = {
            source: public_metric(metric)
            for source, metric in self._source_metrics.items()
        }
        payload["questions"] = [public_metric(metric) for metric in self._question_metrics[-20:]]
        payload["currentQuestionId"] = self._active_question_id
        payload["cancelledStreams"] = self._cancelled_streams
        payload["answerProviderOverride"] = self._answer_provider_override
        return payload

    def build_question_metric_locked(
        self,
        *,
        question_id: str,
        question: str,
        confidence: float,
        reason: str,
        source: str,
        detected_started: float,
        detected_done: float,
        context_started: float,
        context_done: float,
        provider: str,
    ) -> dict[str, Any]:
        source_metric = self._source_metrics.get(source, {})
        metric = {
            "sessionId": self._session_id,
            "questionId": question_id,
            "source": source,
            "reason": reason,
            "provider": provider,
            "fallbackUsed": False,
            "questionConfidence": confidence,
            "contextConfidence": confidence,
            "cancelledStreams": self._cancelled_streams,
            "createdAtMs": wall_ms(),
            "tDetectMs": elapsed_ms(detected_started, detected_done),
            "tContextBuildMs": elapsed_ms(context_started, context_done),
        }
        for source_key, target_key in (
            ("tAudioChunkMs", "tAudioChunkMs"),
            ("tSttInterimMs", "tSttInterimMs"),
            ("tSttFinalMs", "tSttFinalMs"),
        ):
            if source_key in source_metric:
                metric[target_key] = source_metric[source_key]
        return metric

    def add_question_metric_locked(self, metric: dict[str, Any], *, question_ready_at: float) -> None:
        question_id = str(metric["questionId"])
        self._question_metrics.append(metric)
        self._question_metrics = self._question_metrics[-50:]
        self._question_metric_by_id[question_id] = metric
        self._question_runtime[question_id] = {"questionReadyAt": question_ready_at}
        self._active_question_id = question_id
        self.trace_metric_stage_locked(
            source=str(metric["source"]),
            stage="question_detected",
            elapsed_ms=int(metric.get("tDetectMs", 0)),
            questionId=question_id,
        )
        self.trace_metric_stage_locked(
            source=str(metric["source"]),
            stage="context_built",
            elapsed_ms=int(metric.get("tContextBuildMs", 0)),
            questionId=question_id,
        )

    def trace_metric_stage_locked(self, *, source: str, stage: str, elapsed_ms: int, **payload: Any) -> None:
        trace_live_event(
            "metric.stage",
            sessionId=self._session_id,
            source=source,
            stage=stage,
            elapsedMs=elapsed_ms,
            **payload,
        )

    def trace_question_metric_locked(self, metric: dict[str, Any]) -> None:
        trace_live_event("metric.question", **public_metric(metric))

    def provider_name(self, config: Any | None = None) -> str:
        with self._condition:
            override = self._answer_provider_override
        if override:
            return override
        try:
            return (config or load_config()).llm_provider
        except Exception:
            return "unknown"

    def ollama_model(self, config: Any, client: OllamaClient) -> str:
        with self._condition:
            forced_ollama = self._answer_provider_override == "ollama" and config.llm_provider != "ollama"
        if not forced_ollama:
            return str(config.llm_model)
        preferred = select_preferred_model(client.list_models())
        if preferred is None:
            raise ProviderError("Ollama fallback has no local models")
        return preferred.id

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


def normalize_source(source: str) -> str:
    value = source.strip().lower()
    if value in {"remote", "them", "system"}:
        return REMOTE_SOURCE
    if value in {"mic", "me", "user"}:
        return MIC_SOURCE
    raise ValueError("source must be remote or mic")


def normalize_question_key(text: str) -> str:
    return " ".join(text.lower().strip().rstrip("?!.").split())


def answer_chunk(item: AnswerStreamChunk | str, default_provider: str) -> AnswerStreamChunk:
    if isinstance(item, AnswerStreamChunk):
        return item
    return AnswerStreamChunk(str(item), provider=default_provider)


def parse_model_decision(text: str, *, final: bool = False) -> ModelDecision | None:
    cleaned = text.lstrip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].lstrip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    if upper.startswith(MODEL_SKIP_MARKER):
        return ModelDecision("skip")
    if upper.startswith(MODEL_ANSWER_MARKER):
        return ModelDecision("answer", cleaned[len(MODEL_ANSWER_MARKER) :].lstrip())
    if MODEL_SKIP_MARKER.startswith(upper) or MODEL_ANSWER_MARKER.startswith(upper):
        return None
    if final:
        plain = upper.strip(" \t\r\n.!:;[]")
        if plain in {"SKIP", "ПРОПУСТИТЬ", "НЕ ОТВЕЧАТЬ"}:
            return ModelDecision("skip")
        return ModelDecision("answer", cleaned)
    if len(cleaned) >= len(MODEL_ANSWER_MARKER):
        return ModelDecision("answer", cleaned)
    return None


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def elapsed_ms(started: float, ended: float) -> int:
    return int((ended - started) * 1000)


def wall_ms() -> int:
    return int(time.time() * 1000)


def public_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if not key.startswith("_")}


def sse_payload(event: SessionEvent) -> bytes:
    data = json.dumps(event.payload, ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {data}\n\n".encode("utf-8")
