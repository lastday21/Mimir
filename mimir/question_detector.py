from __future__ import annotations

from dataclasses import dataclass


INTERROGATIVE_STARTERS = {
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "do", "does", "is", "are",
    "will", "have", "has", "tell",
    "что", "почему", "как", "когда", "где", "куда", "откуда", "кто",
    "какой", "какая", "какое", "какие", "сколько", "зачем",
    "можно", "можешь", "можете", "могли",
    "расскажи", "расскажите", "объясни", "объясните",
    "подскажи", "подскажите", "покажи", "покажите",
}

LEADING_FILLER_WORDS = {
    "so", "well", "okay", "ok", "and", "but",
    "а", "и", "ну", "так", "то", "ладно", "хорошо",
}

INTERVIEW_PATTERNS = {
    "walk me through",
    "describe a time",
    "tell me about",
    "what would you do",
    "how do you handle",
    "give me an example",
    "explain how",
    "what's your experience with",
    "what is your experience with",
    "can you walk me through",
    "can you describe",
    "can you explain",
    "can you tell me",
    "talk about a time",
    "share an example",
    "how would you",
    "how have you",
    "what approach would you",
    "what's your approach to",
    "what is your approach to",
    "расскажите о",
    "расскажи о",
    "опишите случай",
    "опиши случай",
    "приведите пример",
    "приведи пример",
    "объясните как",
    "объясни как",
    "как бы вы",
    "как бы ты",
    "что бы вы",
    "что бы ты",
    "какой у вас опыт",
    "какой у тебя опыт",
    "можете рассказать",
    "можешь рассказать",
    "можете объяснить",
    "можешь объяснить",
}


@dataclass(frozen=True)
class DetectedQuestion:
    text: str
    confidence: float
    timestamp_ms: int
    source: str


def detect_questions(text: str, timestamp_ms: int = 0, source: str = "Them") -> list[DetectedQuestion]:
    questions: list[DetectedQuestion] = []
    for sentence in split_sentences(text):
        trimmed = sentence.strip()
        if len(trimmed) < 5:
            continue

        lower = trimmed.lower()
        confidence = 0.0
        if trimmed.endswith("?"):
            confidence = 0.95
        elif starts_with_interrogative_word(lower):
            confidence = max(confidence, 0.6)

        if any(pattern in lower for pattern in INTERVIEW_PATTERNS):
            confidence = max(confidence, 0.85)

        if confidence >= 0.5:
            questions.append(DetectedQuestion(trimmed, confidence, timestamp_ms, source))
    return questions


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in ".?!؟？":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current.clear()
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def starts_with_interrogative_word(text: str) -> bool:
    for word in text.split():
        cleaned = clean_word(word)
        if not cleaned or cleaned in LEADING_FILLER_WORDS:
            continue
        return cleaned in INTERROGATIVE_STARTERS
    return False


def clean_word(word: str) -> str:
    return word.strip(" \t\r\n.,!?;:()[]{}\"'«»“”")
