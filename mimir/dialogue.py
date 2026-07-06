from __future__ import annotations

import time
from dataclasses import dataclass, field


REMOTE_SOURCE = "remote"
MIC_SOURCE = "mic"


@dataclass(frozen=True)
class DialogueTurn:
    source: str
    text: str
    is_final: bool = True
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(frozen=True)
class ContextSnapshot:
    session_id: str
    question_id: str
    question: str
    active_topic: str
    latest_remote_turns: list[str]
    latest_user_turns: list[str]
    relevant_prior_questions: list[str]
    transcript_excerpt: str
    answer_mode: str = "interview"
    language: str = "ru"
    confidence: float = 0.0

    def to_prompt_text(self) -> str:
        blocks = [
            f"Текущий вопрос собеседника:\n{self.question}",
            f"Текущая тема:\n{self.active_topic or 'не определена'}",
        ]
        if self.relevant_prior_questions:
            blocks.append("Предыдущие вопросы:\n" + "\n".join(f"- {item}" for item in self.relevant_prior_questions))
        if self.latest_user_turns:
            blocks.append("Что уже сказал пользователь:\n" + "\n".join(f"- {item}" for item in self.latest_user_turns))
        if self.latest_remote_turns:
            blocks.append("Последние реплики собеседника:\n" + "\n".join(f"- {item}" for item in self.latest_remote_turns))
        if self.transcript_excerpt:
            blocks.append("Краткий фрагмент диалога:\n" + self.transcript_excerpt)
        return "\n\n".join(blocks)


class DialogueMemory:
    def __init__(self, max_turns: int = 80) -> None:
        self.max_turns = max_turns
        self.turns: list[DialogueTurn] = []
        self.questions: list[str] = []
        self.active_topic = ""

    def append(self, turn: DialogueTurn) -> None:
        text = turn.text.strip()
        if not text:
            return
        self.turns.append(DialogueTurn(turn.source, text, turn.is_final, turn.timestamp_ms))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        if turn.source == REMOTE_SOURCE and turn.is_final:
            self.active_topic = infer_topic(text, self.active_topic)

    def remember_question(self, question: str) -> None:
        normalized = question.strip()
        if not normalized:
            return
        if normalized not in self.questions:
            self.questions.append(normalized)
        self.questions = self.questions[-10:]

    def recent_questions(self, limit: int = 5) -> list[str]:
        return self.questions[-limit:]

    def build_context(self, session_id: str, question_id: str, question: str, confidence: float) -> ContextSnapshot:
        remote = [turn.text for turn in self.turns if turn.source == REMOTE_SOURCE and turn.is_final][-8:]
        user = [turn.text for turn in self.turns if turn.source == MIC_SOURCE and turn.is_final][-6:]
        excerpt = "\n".join(format_turn(turn) for turn in self.turns[-16:] if turn.is_final)
        return ContextSnapshot(
            session_id=session_id,
            question_id=question_id,
            question=question.strip(),
            active_topic=self.active_topic,
            latest_remote_turns=remote,
            latest_user_turns=user,
            relevant_prior_questions=self.recent_questions(),
            transcript_excerpt=excerpt,
            confidence=confidence,
        )

    def payload(self) -> dict[str, object]:
        return {
            "activeTopic": self.active_topic,
            "turns": [
                {
                    "source": turn.source,
                    "text": turn.text,
                    "isFinal": turn.is_final,
                    "timestampMs": turn.timestamp_ms,
                }
                for turn in self.turns[-30:]
            ],
            "questions": self.recent_questions(),
        }


def format_turn(turn: DialogueTurn) -> str:
    speaker = "Собеседник" if turn.source == REMOTE_SOURCE else "Пользователь"
    return f"{speaker}: {turn.text}"


def infer_topic(text: str, current: str) -> str:
    words = [
        clean_topic_word(word)
        for word in text.split()
        if len(clean_topic_word(word)) >= 4
    ]
    if not words:
        return current
    return " ".join(words[:6])


def clean_topic_word(word: str) -> str:
    return word.strip(" \t\r\n.,!?;:()[]{}\"'«»“”").lower()
