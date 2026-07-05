from __future__ import annotations

from .models import ChatMessage


SYSTEM_PROMPT = (
    "Ты Mimir, помощник для встреч и интервью. "
    "Отвечай кратко, по делу, на русском языке, если пользователь не просит иначе. "
    "Если вопрос технический, давай конкретные шаги и проверяемые команды."
)


def build_messages(user_text: str, transcript: str = "") -> list[ChatMessage]:
    context = transcript.strip()
    if context:
        user_text = f"Контекст встречи:\n{context}\n\nВопрос:\n{user_text.strip()}"
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", user_text.strip()),
    ]
