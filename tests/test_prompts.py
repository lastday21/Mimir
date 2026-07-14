import unittest

from mimir.config import AppConfig, ConversationSettings, UserProfile
from mimir.prompts import (
    build_dialogue_summary_messages,
    build_realtime_messages,
    build_realtime_session_instructions,
    build_transcript_decision_messages,
)


class PromptContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            profile=UserProfile(
                name="Тимур",
                role="Разработчик",
                background="Разрабатывает серверные приложения на Python",
                projects="Сервис обработки заказов",
                stories="Сократил время обработки очереди",
            ),
            conversation=ConversationSettings(
                mode="meeting",
                goal="Не погружаться в детали и понимать, что нужно сделать",
                context="Еженедельная встреча команды продукта",
            ),
        )

    def test_realtime_prompt_contains_profile_and_goal(self) -> None:
        messages = build_realtime_messages(
            "Когда будет готово?",
            "Коллега рассказал о новой задаче.",
            self.config,
        )

        prompt = messages[-1].content
        self.assertIn("Обычная рабочая встреча", prompt)
        self.assertIn("Не погружаться в детали", prompt)
        self.assertIn("Разрабатывает серверные приложения на Python", prompt)
        self.assertIn("Сервис обработки заказов", prompt)
        self.assertIn("Когда будет готово?", prompt)

    def test_decision_prompt_uses_conversation_context(self) -> None:
        messages = build_transcript_decision_messages(
            "Тимур, возьмешь эту задачу?",
            "Обсуждается выпуск новой версии.",
            self.config,
        )

        prompt = messages[-1].content
        self.assertIn("Еженедельная встреча команды продукта", prompt)
        self.assertIn("что нужно сделать", prompt)
        self.assertIn("Тимур, возьмешь эту задачу?", prompt)

    def test_direct_realtime_instructions_contain_profile_goal_and_mode(self) -> None:
        instructions = build_realtime_session_instructions("Основные правила", self.config)

        self.assertIn("Основные правила", instructions)
        self.assertIn("Обычная рабочая встреча", instructions)
        self.assertIn("Не погружаться в детали", instructions)
        self.assertIn("Еженедельная встреча команды продукта", instructions)
        self.assertIn("Разработчик", instructions)
        self.assertIn("Сервис обработки заказов", instructions)

    def test_summary_prompt_keeps_previous_summary_and_final_transcript(self) -> None:
        messages = build_dialogue_summary_messages(
            "Команда обсуждает выпуск.",
            "Собеседник: Тимур, проверь сборку.\nПользователь: Проверю сегодня.",
            self.config,
        )

        prompt = messages[-1].content
        self.assertIn("Команда обсуждает выпуск", prompt)
        self.assertIn("Тимур, проверь сборку", prompt)
        self.assertIn("Проверю сегодня", prompt)
        self.assertIn("Не погружаться в детали", prompt)


if __name__ == "__main__":
    unittest.main()
