from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


REMOTE_SOURCE = "remote"
MIC_SOURCE = "mic"
MIN_MEMORY_WINDOW_MS = 5 * 60 * 1000


@dataclass(frozen=True)
class DialogueTurn:
    source: str
    text: str
    is_final: bool = True
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    turn_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex}")


@dataclass(frozen=True)
class TranscriptUpdate:
    turn: DialogueTurn
    operation: str


@dataclass(frozen=True)
class UserAnswerTurn:
    turn_id: str
    text: str
    timestamp_ms: int


@dataclass
class DialogueExchange:
    question_id: str
    question: str
    timestamp_ms: int
    updated_at_ms: int
    hint: str = ""
    user_turns: list[UserAnswerTurn] = field(default_factory=list)

    @property
    def user_answer(self) -> str:
        return " ".join(turn.text for turn in self.user_turns).strip()


@dataclass(frozen=True)
class ContextSnapshot:
    session_id: str
    question_id: str
    question: str
    active_topic: str
    latest_remote_turns: list[str]
    latest_user_turns: list[str]
    relevant_prior_questions: list[str]
    related_exchanges: list[str]
    transcript_excerpt: str
    answer_mode: str = "interview"
    language: str = "ru"
    confidence: float = 0.0

    def to_prompt_text(self) -> str:
        return f"Текущий вопрос собеседника:\n{self.question}\n\n{self.to_background_text()}"

    def to_background_text(self) -> str:
        blocks = [f"Текущая тема:\n{self.active_topic or 'не определена'}"]
        if self.related_exchanges:
            blocks.append("Связанные вопросы, подсказки и ответы пользователя:\n" + "\n\n".join(self.related_exchanges))
        elif self.relevant_prior_questions:
            blocks.append("Предыдущие вопросы:\n" + "\n".join(f"- {item}" for item in self.relevant_prior_questions))
        if self.latest_user_turns:
            blocks.append("Что уже сказал пользователь:\n" + "\n".join(f"- {item}" for item in self.latest_user_turns))
        if self.latest_remote_turns:
            blocks.append("Последние реплики собеседника:\n" + "\n".join(f"- {item}" for item in self.latest_remote_turns))
        if self.transcript_excerpt:
            blocks.append("Краткий фрагмент диалога:\n" + self.transcript_excerpt)
        return "\n\n".join(blocks)


class DialogueMemory:
    def __init__(
        self,
        retention_ms: int = MIN_MEMORY_WINDOW_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.retention_ms = max(MIN_MEMORY_WINDOW_MS, int(retention_ms))
        self.turns: list[DialogueTurn] = []
        self.exchanges: list[DialogueExchange] = []
        self.active_topic = ""
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._pending_interim: dict[str, str] = {}
        self._latest_final: dict[str, str] = {}

    def append(self, turn: DialogueTurn, *, refine_latest: bool = False) -> TranscriptUpdate | None:
        text = turn.text.strip()
        if not text:
            return None
        self._prune()

        if refine_latest:
            latest_id = self._latest_final.get(turn.source)
            index = self._turn_index(latest_id)
            if index is not None:
                stored = DialogueTurn(turn.source, text, True, turn.timestamp_ms, latest_id)
                self.turns[index] = stored
                self._pending_interim.pop(turn.source, None)
                self._refresh_active_topic()
                return TranscriptUpdate(stored, "replace")

        pending_id = self._pending_interim.get(turn.source)
        index = self._turn_index(pending_id)
        if index is not None:
            stored = DialogueTurn(turn.source, text, turn.is_final, turn.timestamp_ms, pending_id)
            self.turns[index] = stored
            if turn.is_final:
                self._pending_interim.pop(turn.source, None)
                self._latest_final[turn.source] = pending_id
            self._refresh_active_topic()
            return TranscriptUpdate(stored, "replace")
        self._pending_interim.pop(turn.source, None)

        stored = DialogueTurn(turn.source, text, turn.is_final, turn.timestamp_ms, turn.turn_id)
        self.turns.append(stored)
        if turn.is_final:
            self._latest_final[turn.source] = stored.turn_id
        else:
            self._pending_interim[turn.source] = stored.turn_id
        self._refresh_active_topic()
        return TranscriptUpdate(stored, "append")

    def remember_question(self, question_id: str, question: str, timestamp_ms: int | None = None) -> None:
        normalized = question.strip()
        if not question_id or not normalized:
            return
        now_ms = timestamp_ms if timestamp_ms is not None else self._clock_ms()
        self._prune(now_ms)
        exchange = self._find_exchange(question_id)
        if exchange is not None:
            exchange.question = normalized
            exchange.updated_at_ms = max(exchange.updated_at_ms, now_ms)
            return
        self.exchanges.append(DialogueExchange(question_id, normalized, now_ms, now_ms))

    def record_hint_delta(self, question_id: str, text: str, timestamp_ms: int | None = None) -> None:
        if not text.strip():
            return
        now_ms = timestamp_ms if timestamp_ms is not None else self._clock_ms()
        self._prune(now_ms)
        exchange = self._find_exchange(question_id)
        if exchange is None:
            return
        exchange.hint += text
        exchange.updated_at_ms = max(exchange.updated_at_ms, now_ms)

    def record_user_answer(self, question_id: str, turn: DialogueTurn) -> None:
        if not question_id or not turn.is_final or not turn.text.strip():
            return
        self._prune(turn.timestamp_ms)
        exchange = self._find_exchange(question_id)
        if exchange is None:
            return
        answer = UserAnswerTurn(turn.turn_id, turn.text.strip(), turn.timestamp_ms)
        for index, current in enumerate(exchange.user_turns):
            if current.turn_id == turn.turn_id:
                exchange.user_turns[index] = answer
                break
        else:
            exchange.user_turns.append(answer)
        exchange.updated_at_ms = max(exchange.updated_at_ms, turn.timestamp_ms)

    def recent_questions(self, limit: int = 5, *, exclude_id: str = "") -> list[str]:
        self._prune()
        questions = [exchange.question for exchange in self.exchanges if exchange.question_id != exclude_id]
        return questions[-limit:]

    def build_context(self, session_id: str, question_id: str, question: str, confidence: float) -> ContextSnapshot:
        self._prune()
        remote = [turn.text for turn in self.turns if turn.source == REMOTE_SOURCE and turn.is_final][-8:]
        user = [turn.text for turn in self.turns if turn.source == MIC_SOURCE and turn.is_final][-6:]
        excerpt = "\n".join(format_turn(turn) for turn in self.turns[-16:] if turn.is_final)
        prior_exchanges = [exchange for exchange in self.exchanges if exchange.question_id != question_id][-5:]
        return ContextSnapshot(
            session_id=session_id,
            question_id=question_id,
            question=question.strip(),
            active_topic=self.active_topic,
            latest_remote_turns=remote,
            latest_user_turns=user,
            relevant_prior_questions=[exchange.question for exchange in prior_exchanges],
            related_exchanges=[format_exchange(exchange) for exchange in prior_exchanges],
            transcript_excerpt=excerpt,
            confidence=confidence,
        )

    def realtime_context(self, max_turns: int = 12, max_chars: int = 1800) -> str:
        self._prune()
        lines = [format_turn(turn) for turn in self.turns if turn.is_final][-max_turns:]
        if not lines:
            return ""

        selected: list[str] = []
        current_len = 0
        for line in reversed(lines):
            extra = 1 if selected else 0
            if not selected and len(line) > max_chars:
                selected.append(line[:max_chars].rstrip())
                break
            if selected and current_len + extra + len(line) > max_chars:
                break
            selected.append(line)
            current_len += extra + len(line)
        return "\n".join(reversed(selected))

    def payload(self) -> dict[str, object]:
        self._prune()
        return {
            "activeTopic": self.active_topic,
            "windowMs": self.retention_ms,
            "turns": [
                {
                    "turnId": turn.turn_id,
                    "source": turn.source,
                    "text": turn.text,
                    "isFinal": turn.is_final,
                    "timestampMs": turn.timestamp_ms,
                }
                for turn in self.turns
            ],
            "questions": [exchange.question for exchange in self.exchanges],
            "exchanges": [
                {
                    "questionId": exchange.question_id,
                    "question": exchange.question,
                    "hint": exchange.hint.strip(),
                    "userAnswer": exchange.user_answer,
                    "timestampMs": exchange.timestamp_ms,
                    "updatedAtMs": exchange.updated_at_ms,
                }
                for exchange in self.exchanges
            ],
        }

    def _prune(self, now_ms: int | None = None) -> None:
        cutoff = (now_ms if now_ms is not None else self._clock_ms()) - self.retention_ms
        previous_turn_count = len(self.turns)
        self.turns = [turn for turn in self.turns if turn.timestamp_ms >= cutoff]
        self.exchanges = [exchange for exchange in self.exchanges if exchange.updated_at_ms >= cutoff]

        turn_ids = {turn.turn_id for turn in self.turns}
        self._pending_interim = {
            source: turn_id
            for source, turn_id in self._pending_interim.items()
            if turn_id in turn_ids
        }
        self._latest_final = {
            source: turn_id
            for source, turn_id in self._latest_final.items()
            if turn_id in turn_ids
        }
        if len(self.turns) != previous_turn_count:
            self._refresh_active_topic()

    def _turn_index(self, turn_id: str | None) -> int | None:
        if not turn_id:
            return None
        for index, turn in enumerate(self.turns):
            if turn.turn_id == turn_id:
                return index
        return None

    def _find_exchange(self, question_id: str) -> DialogueExchange | None:
        for exchange in reversed(self.exchanges):
            if exchange.question_id == question_id:
                return exchange
        return None

    def _refresh_active_topic(self) -> None:
        topic = ""
        for turn in self.turns:
            if turn.source == REMOTE_SOURCE and turn.is_final:
                topic = infer_topic(turn.text, topic)
        self.active_topic = topic


def format_turn(turn: DialogueTurn) -> str:
    speaker = "Собеседник" if turn.source == REMOTE_SOURCE else "Пользователь"
    return f"{speaker}: {turn.text}"


def format_exchange(exchange: DialogueExchange) -> str:
    lines = [f"Вопрос: {exchange.question}"]
    if exchange.hint.strip():
        lines.append(f"Подсказка Mimir: {exchange.hint.strip()}")
    if exchange.user_answer:
        lines.append(f"Ответ пользователя: {exchange.user_answer}")
    return "\n".join(lines)


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
