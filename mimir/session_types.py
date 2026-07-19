from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .dialogue import MIC_SOURCE, REMOTE_SOURCE


MODEL_SKIP_MARKER = "[[SKIP]]"
MODEL_ANSWER_MARKER = "[[ANSWER]]"
MODEL_UNCLEAR_MARKER = "[[UNCLEAR]]"


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
    if upper.startswith(MODEL_UNCLEAR_MARKER):
        return ModelDecision("unclear", cleaned[len(MODEL_UNCLEAR_MARKER) :].lstrip())
    if any(
        marker.startswith(upper)
        for marker in (MODEL_SKIP_MARKER, MODEL_ANSWER_MARKER, MODEL_UNCLEAR_MARKER)
    ):
        return None
    if final:
        plain = upper.strip(" \t\r\n.!:;[]")
        if plain in {"SKIP", "ПРОПУСТИТЬ", "НЕ ОТВЕЧАТЬ"}:
            return ModelDecision("skip")
        if plain in {"UNCLEAR", "НЕРАЗБОРЧИВО", "НЕ УВЕРЕН"}:
            return ModelDecision("unclear")
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
