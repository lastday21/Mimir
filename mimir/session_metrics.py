from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .session_types import elapsed_ms, public_metric, wall_ms


class SessionMetrics:
    def __init__(self, trace_event: Callable[..., None]) -> None:
        self.trace_event = trace_event
        self.reset()

    def reset(self) -> None:
        self.values: dict[str, Any] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.questions: list[dict[str, Any]] = []
        self.questions_by_id: dict[str, dict[str, Any]] = {}
        self.question_runtime: dict[str, dict[str, float]] = {}

    def set_value(self, key: str, value: Any) -> None:
        self.values[key] = value
        self.values["updatedAt"] = wall_ms()

    def mark_error(self, phase: str, error: str) -> None:
        self.values["lastError"] = error
        self.values["errorPhase"] = phase
        self.values["updatedAt"] = wall_ms()

    def increment(self, key: str) -> None:
        self.values[key] = int(self.values.get(key, 0)) + 1
        self.values["updatedAt"] = wall_ms()

    def record_speech_started(self, session_id: str, source: str) -> None:
        now = time.monotonic()
        self.sources[source] = {
            "source": source,
            "speechStartedAtMs": wall_ms(),
            "_speechStartedAt": now,
        }
        self.trace_stage(session_id, source=source, stage="speech_started", elapsed_ms=0)

    def record_audio_chunk(self, session_id: str, source: str, byte_count: int) -> None:
        now = time.monotonic()
        now_ms = wall_ms()
        metric = self.sources.setdefault(
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
        self.trace_stage(
            session_id,
            source=source,
            stage="audio_chunk",
            elapsed_ms=elapsed,
            bytes=byte_count,
        )

    def record_stt_result(self, session_id: str, source: str, is_final: bool) -> None:
        now = time.monotonic()
        now_ms = wall_ms()
        metric = self.sources.setdefault(
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
            self.trace_stage(session_id, source=source, stage="stt_interim", elapsed_ms=elapsed)
            return
        if is_final:
            elapsed = elapsed_ms(audio_at, now)
            metric["sttFinalAtMs"] = now_ms
            metric["_sttFinalAt"] = now
            metric["tSttFinalMs"] = elapsed
            self.trace_stage(session_id, source=source, stage="stt_final", elapsed_ms=elapsed)

    def build_question(
        self,
        *,
        session_id: str,
        question_id: str,
        confidence: float,
        reason: str,
        source: str,
        detected_started: float,
        detected_done: float,
        context_started: float,
        context_done: float,
        provider: str,
        cancelled_streams: int,
    ) -> dict[str, Any]:
        source_metric = self.sources.get(source, {})
        metric = {
            "sessionId": session_id,
            "questionId": question_id,
            "source": source,
            "reason": reason,
            "provider": provider,
            "fallbackUsed": False,
            "questionConfidence": confidence,
            "contextConfidence": confidence,
            "cancelledStreams": cancelled_streams,
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

    def add_question(self, session_id: str, metric: dict[str, Any], *, question_ready_at: float) -> None:
        question_id = str(metric["questionId"])
        self.questions.append(metric)
        self.questions = self.questions[-50:]
        self.questions_by_id[question_id] = metric
        self.question_runtime[question_id] = {"questionReadyAt": question_ready_at}
        self.trace_stage(
            session_id,
            source=str(metric["source"]),
            stage="question_detected",
            elapsed_ms=int(metric.get("tDetectMs", 0)),
            questionId=question_id,
        )
        self.trace_stage(
            session_id,
            source=str(metric["source"]),
            stage="context_built",
            elapsed_ms=int(metric.get("tContextBuildMs", 0)),
            questionId=question_id,
        )

    def record_first_hint(self, question_id: str, *, provider: str | None = None) -> None:
        now = time.monotonic()
        metric = self.questions_by_id.get(question_id)
        runtime = self.question_runtime.get(question_id)
        if metric is None or runtime is None or "tFirstHintMs" in metric:
            return
        metric["firstHintAtMs"] = wall_ms()
        metric["tFirstHintMs"] = elapsed_ms(runtime["questionReadyAt"], now)
        if provider:
            metric["provider"] = provider
        self.trace_question(metric)

    def record_answer_done(self, question_id: str) -> None:
        now = time.monotonic()
        metric = self.questions_by_id.get(question_id)
        runtime = self.question_runtime.get(question_id)
        if metric is None or runtime is None:
            return
        metric["answerDoneAtMs"] = wall_ms()
        metric["tAnswerDoneMs"] = elapsed_ms(runtime["questionReadyAt"], now)
        self.trace_question(metric)

    def set_question_field(self, question_id: str, key: str, value: Any) -> None:
        metric = self.questions_by_id.get(question_id)
        if metric is not None:
            metric[key] = value

    def payload(
        self,
        *,
        current_question_id: str,
        cancelled_streams: int,
        provider_override: str | None,
    ) -> dict[str, Any]:
        payload = dict(self.values)
        payload["sources"] = {
            source: public_metric(metric)
            for source, metric in self.sources.items()
        }
        payload["questions"] = [public_metric(metric) for metric in self.questions[-20:]]
        payload["currentQuestionId"] = current_question_id
        payload["cancelledStreams"] = cancelled_streams
        payload["answerProviderOverride"] = provider_override
        return payload

    def trace_stage(self, session_id: str, *, source: str, stage: str, elapsed_ms: int, **payload: Any) -> None:
        self.trace_event(
            "metric.stage",
            sessionId=session_id,
            source=source,
            stage=stage,
            elapsedMs=elapsed_ms,
            **payload,
        )

    def trace_question(self, metric: dict[str, Any]) -> None:
        self.trace_event("metric.question", **public_metric(metric))
