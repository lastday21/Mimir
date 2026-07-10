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

TRANSCRIPT_DECISION_SYSTEM_PROMPT = (
    "Ты Mimir, скрытый помощник пользователя на интервью или рабочем созвоне. "
    "Определи по итоговой реплике собеседника и истории диалога, нужна ли пользователю подсказка. "
    "Содержательный вопрос, просьба объяснить, сравнить, спроектировать, привести пример или уточнение "
    "требуют ответа. Приветствие, обычное утверждение, служебная фраза, повтор и незавершенный шум ответа не требуют. "
    "Если подсказка не нужна, верни только [[SKIP]]. "
    "Если нужна, начни ответ с [[ANSWER]] и сразу дай короткую формулировку, которую можно произнести вслух. "
    "Не выдумывай опыт, компании, цифры и факты, которых нет в контексте. "
    "Не объясняй свое решение и не используй другие служебные метки."
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


def build_transcript_decision_messages(utterance: str, context: str) -> list[ChatMessage]:
    prompt = (
        f"История диалога:\n{context.strip() or 'нет предыдущего контекста'}\n\n"
        f"Новая итоговая реплика собеседника:\n{utterance.strip()}"
    )
    return [
        ChatMessage("system", TRANSCRIPT_DECISION_SYSTEM_PROMPT),
        ChatMessage("user", prompt),
    ]
