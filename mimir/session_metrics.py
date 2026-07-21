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
        self.utterances: dict[tuple[str, int], dict[str, Any]] = {}
        self.questions: list[dict[str, Any]] = []
        self.questions_by_id: dict[str, dict[str, Any]] = {}
        self.question_runtime: dict[str, dict[str, float]] = {}
        self.finalized_utterances: dict[str, dict[str, Any]] = {}

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

    def decrement(self, key: str) -> None:
        self.values[key] = max(0, int(self.values.get(key, 0)) - 1)
        self.values["updatedAt"] = wall_ms()

    def record_speech_started(self, session_id: str, source: str, sequence: int = 0) -> None:
        now = time.monotonic()
        metric = {
            "source": source,
            "utteranceSequence": sequence,
            "speechStartedAtMs": wall_ms(),
            "_speechStartedAt": now,
        }
        self.sources[source] = metric
        if sequence > 0:
            self.utterances[(source, sequence)] = metric
        self.trace_stage(session_id, source=source, stage="speech_started", elapsed_ms=0)

    def record_audio_chunk(
        self,
        session_id: str,
        source: str,
        byte_count: int,
        sequence: int = 0,
    ) -> None:
        now = time.monotonic()
        now_ms = wall_ms()
        metric = self.utterances.get((source, sequence)) if sequence > 0 else None
        if metric is None:
            metric = self.sources.setdefault(
                source,
                {
                    "source": source,
                    "utteranceSequence": sequence,
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

    def record_speech_ended(
        self,
        session_id: str,
        source: str,
        trailing_silence_ms: int = 0,
        sequence: int = 0,
    ) -> None:
        now = time.monotonic()
        requested_trailing_ms = max(0, int(trailing_silence_ms))
        metric = self.utterances.get((source, sequence)) if sequence > 0 else None
        if metric is None:
            metric = self.sources.setdefault(
                source,
                {"source": source, "utteranceSequence": sequence},
            )
        started = float(metric.get("_speechStartedAt", now))
        speech_ended_at = max(started, now - requested_trailing_ms / 1000)
        trailing_ms = max(0, round((now - speech_ended_at) * 1000))
        closed_at_ms = wall_ms()
        started_at_ms = int(metric.get("speechStartedAtMs", closed_at_ms))
        metric["speechEndedAtMs"] = started_at_ms + elapsed_ms(started, speech_ended_at)
        metric["vadClosedAtMs"] = closed_at_ms
        metric["vadTailMs"] = requested_trailing_ms
        metric["trailingSilenceMs"] = trailing_ms
        metric["_speechEndedAt"] = speech_ended_at
        for question in self.questions:
            if (
                question.get("source") != source
                or int(question.get("utteranceSequence", 0)) != sequence
            ):
                continue
            question_id = str(question.get("questionId", ""))
            runtime = self.question_runtime.get(question_id)
            question["utteranceEndedAtMs"] = metric["speechEndedAtMs"]
            question["_utteranceEndedAt"] = speech_ended_at
            question["latencyBaseline"] = "speech_end"
            question["vadTailMs"] = requested_trailing_ms
            question["trailingSilenceMs"] = trailing_ms
            if runtime is None:
                continue
            runtime["utteranceEndedAt"] = speech_ended_at
            first_hint_at = runtime.get("firstHintAt")
            if first_hint_at is not None:
                question["tFirstHintMs"] = max(
                    0,
                    elapsed_ms(speech_ended_at, float(first_hint_at)),
                )
            answer_done_at = runtime.get("answerDoneAt")
            if answer_done_at is not None:
                question["tAnswerDoneMs"] = max(
                    0,
                    elapsed_ms(speech_ended_at, float(answer_done_at)),
                )
            self.trace_question(question)
        finalized = self.finalized_utterances.get(source)
        if finalized is not None and int(finalized.get("utteranceSequence", 0)) == sequence:
            self.finalized_utterances[source] = dict(metric)
        self.trace_stage(
            session_id,
            source=source,
            stage="speech_ended",
            elapsed_ms=elapsed_ms(started, speech_ended_at),
            trailingSilenceMs=trailing_ms,
            vadTailMs=requested_trailing_ms,
        )

    def record_stt_result(
        self,
        session_id: str,
        source: str,
        is_final: bool,
        sequence: int = 0,
    ) -> None:
        now = time.monotonic()
        now_ms = wall_ms()
        metric = self.utterances.get((source, sequence)) if sequence > 0 else None
        if metric is None:
            metric = self.sources.setdefault(
                source,
                {
                    "source": source,
                    "utteranceSequence": sequence,
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
            self.finalized_utterances[source] = dict(metric)
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
        utterance_sequence: int = 0,
    ) -> dict[str, Any]:
        source_metric = (
            self.utterances.get((source, utterance_sequence))
            if utterance_sequence > 0
            else None
        )
        if source_metric is None:
            source_metric = self.finalized_utterances.get(source, self.sources.get(source, {}))
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
            "latencySchemaVersion": 2,
        }
        for source_key, target_key in (
            ("utteranceSequence", "utteranceSequence"),
            ("tAudioChunkMs", "tAudioChunkMs"),
            ("tSttInterimMs", "tSttInterimMs"),
            ("tSttFinalMs", "tSttFinalMs"),
            ("vadTailMs", "vadTailMs"),
            ("trailingSilenceMs", "trailingSilenceMs"),
        ):
            if source_key in source_metric:
                metric[target_key] = source_metric[source_key]
        if "speechEndedAtMs" in source_metric:
            metric["utteranceEndedAtMs"] = source_metric["speechEndedAtMs"]
            metric["_utteranceEndedAt"] = source_metric.get("_speechEndedAt")
            metric["latencyBaseline"] = "speech_end"
        elif "sttFinalAtMs" in source_metric:
            metric["utteranceEndedAtMs"] = source_metric["sttFinalAtMs"]
            metric["_utteranceEndedAt"] = source_metric.get("_sttFinalAt")
            metric["latencyBaseline"] = "stt_final"
        else:
            metric["latencyBaseline"] = "question_ready"
        return metric

    def add_question(self, session_id: str, metric: dict[str, Any], *, question_ready_at: float) -> None:
        question_id = str(metric["questionId"])
        self.questions.append(metric)
        self.questions = self.questions[-50:]
        self.questions_by_id[question_id] = metric
        self.question_runtime[question_id] = {
            "questionReadyAt": question_ready_at,
            "utteranceEndedAt": float(metric.get("_utteranceEndedAt") or question_ready_at),
        }
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
        runtime["firstHintAt"] = now
        metric["tQuestionToFirstHintMs"] = elapsed_ms(runtime["questionReadyAt"], now)
        metric["tFirstHintMs"] = elapsed_ms(runtime["utteranceEndedAt"], now)
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
        runtime["answerDoneAt"] = now
        metric["tQuestionToAnswerDoneMs"] = elapsed_ms(runtime["questionReadyAt"], now)
        metric["tAnswerDoneMs"] = elapsed_ms(runtime["utteranceEndedAt"], now)
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
