from __future__ import annotations

from .models import ChatMessage


SYSTEM_PROMPT = (
    "Ты Mimir, помощник для встреч и интервью. "
    "Отвечай кратко, по делу, на русском языке, если пользователь не просит иначе. "
    "Если вопрос технический, давай конкретные шаги и проверяемые команды."
)

REALTIME_SYSTEM_PROMPT = (
    "Ты Mimir, realtime-помощник на собеседовании. "
    "Давай короткую подсказку, которую можно произнести вслух. "
    "Отвечай от первого лица, если вопрос про опыт пользователя. "
    "Не выдумывай опыт, компании, цифры и факты, которых нет в контексте. "
    "Сначала дай прямой ответ, затем максимум 1-3 опорных пункта, если они нужны."
)


def build_messages(user_text: str, transcript: str = "") -> list[ChatMessage]:
    context = transcript.strip()
    if context:
        user_text = f"Контекст встречи:\n{context}\n\nВопрос:\n{user_text.strip()}"
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", user_text.strip()),
    ]


def build_realtime_messages(question: str, context: str) -> list[ChatMessage]:
    prompt = (
        f"{context.strip()}\n\n"
        f"Ответь на текущий вопрос коротко и пригодно для живого ответа:\n{question.strip()}"
    )
    return [
        ChatMessage("system", REALTIME_SYSTEM_PROMPT),
        ChatMessage("user", prompt.strip()),
    ]
