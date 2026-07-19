from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .config import AppConfig, UserProfile


PROFILE_FACT_LIMIT = 5
PROFILE_CONTEXT_CHAR_LIMIT = 1800
PROFILE_FACT_CHAR_LIMIT = 420

STOP_WORDS = {
    "без",
    "был",
    "была",
    "были",
    "быть",
    "вам",
    "вас",
    "весь",
    "для",
    "его",
    "если",
    "есть",
    "или",
    "как",
    "когда",
    "который",
    "мне",
    "можно",
    "мой",
    "надо",
    "наш",
    "нужно",
    "она",
    "они",
    "про",
    "свой",
    "так",
    "также",
    "там",
    "тебя",
    "тебе",
    "только",
    "уже",
    "чем",
    "что",
    "чтобы",
    "этот",
    "это",
    "this",
    "that",
    "the",
    "with",
    "from",
    "have",
}

FIELD_ALIASES = {
    "Опыт и резюме": "опыт навыки стек технологии резюме работал специализация обязанности",
    "Проекты": "проект продукт система сервис задача архитектура разработка внедрение",
    "Подготовленные истории": "пример ситуация достижение результат проблема решение случай star",
}


@dataclass(frozen=True)
class ProfileFact:
    label: str
    text: str
    order: int


def select_profile_facts(
    config: AppConfig,
    relevance_text: str = "",
    *,
    max_facts: int = PROFILE_FACT_LIMIT,
    max_chars: int = PROFILE_CONTEXT_CHAR_LIMIT,
) -> list[ProfileFact]:
    facts = profile_facts(config.profile)
    if not facts or max_facts <= 0 or max_chars <= 0:
        return []

    query = "\n".join(
        part
        for part in (
            config.conversation.goal,
            config.conversation.context,
            relevance_text,
        )
        if part.strip()
    )
    query_terms = searchable_terms(query)
    scored: list[tuple[int, int, ProfileFact]] = []
    for fact in facts:
        content_overlap = query_terms & searchable_terms(fact.text)
        field_overlap = query_terms & searchable_terms(f"{fact.label} {FIELD_ALIASES.get(fact.label, '')}")
        content_score = sum(4 if len(term) >= 6 else 3 for term in content_overlap)
        score = content_score + len(field_overlap)
        if score:
            scored.append((content_score, score, fact))

    has_content_match = any(content_score for content_score, _score, _fact in scored)
    if has_content_match:
        selected = [
            fact
            for content_score, _score, fact in sorted(
                scored,
                key=lambda item: (-item[1], item[2].order),
            )
            if content_score
        ]
    else:
        selected = first_fact_per_label(
            fact
            for _content_score, _score, fact in sorted(
                scored,
                key=lambda item: (-item[1], item[2].order),
            )
        )
    if not selected:
        selected = fallback_profile_facts(facts, config.conversation.mode)
    else:
        selected = add_general_background(selected, facts)

    result: list[ProfileFact] = []
    used_chars = 0
    for fact in selected:
        if len(result) >= max_facts:
            break
        rendered_length = len(fact.label) + len(fact.text) + 4
        if result and used_chars + rendered_length > max_chars:
            continue
        if not result and rendered_length > max_chars:
            text_limit = max(1, max_chars - len(fact.label) - 4)
            result.append(ProfileFact(fact.label, fact.text[:text_limit].rstrip(), fact.order))
            break
        result.append(fact)
        used_chars += rendered_length
    return result


def profile_facts(profile: UserProfile) -> list[ProfileFact]:
    fields = (
        ("Опыт и резюме", profile.background),
        ("Проекты", profile.projects),
        ("Подготовленные истории", profile.stories),
    )
    facts: list[ProfileFact] = []
    order = 0
    for label, value in fields:
        for chunk in split_profile_text(value):
            facts.append(ProfileFact(label, chunk, order))
            order += 1
    return facts


def split_profile_text(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    pieces = re.split(r"\n+|(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    for piece in pieces:
        clean = piece.strip(" \t-•*")
        while len(clean) > PROFILE_FACT_CHAR_LIMIT:
            boundary = clean.rfind(" ", 0, PROFILE_FACT_CHAR_LIMIT + 1)
            if boundary < PROFILE_FACT_CHAR_LIMIT // 2:
                boundary = PROFILE_FACT_CHAR_LIMIT
            chunks.append(clean[:boundary].rstrip())
            clean = clean[boundary:].strip()
        if clean:
            chunks.append(clean)
    return chunks


def searchable_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9+#.-]+", text.lower().replace("ё", "е")):
        normalized = token.strip(".-")
        if len(normalized) < 3 or normalized in STOP_WORDS:
            continue
        terms.add(normalized if len(normalized) <= 5 else normalized[:6])
    return terms


def fallback_profile_facts(facts: list[ProfileFact], mode: str) -> list[ProfileFact]:
    labels = ("Опыт и резюме", "Проекты") if mode in {"interview", "technical"} else ("Опыт и резюме",)
    selected: list[ProfileFact] = []
    for label in labels:
        fact = next((item for item in facts if item.label == label), None)
        if fact is not None:
            selected.append(fact)
    return selected or facts[:1]


def add_general_background(selected: list[ProfileFact], facts: list[ProfileFact]) -> list[ProfileFact]:
    background = next((item for item in facts if item.label == "Опыт и резюме"), None)
    if background is None or background in selected:
        return selected
    return [background, *selected]


def first_fact_per_label(facts: Iterable[ProfileFact]) -> list[ProfileFact]:
    result: list[ProfileFact] = []
    seen_labels: set[str] = set()
    for fact in facts:
        if fact.label in seen_labels:
            continue
        seen_labels.add(fact.label)
        result.append(fact)
    return result
