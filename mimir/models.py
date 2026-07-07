from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    provider: str
    context_window: int | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class SpeechRecognitionResult:
    text: str
    is_final: bool
    end_of_utterance: bool = False
    confidence: float | None = None
