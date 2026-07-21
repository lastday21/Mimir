import unittest

from mimir.config import AppConfig, ConversationSettings, UserProfile
from mimir.prompts import (
    DIALOGUE_SUMMARY_SYSTEM_PROMPT,
    REALTIME_SYSTEM_PROMPT,
    TRANSCRIPT_DECISION_SYSTEM_PROMPT,
    build_dialogue_summary_messages,
    build_realtime_messages,
    build_realtime_session_instructions,
    build_transcript_decision_messages,
    requires_transcript_clarification,
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

    def test_live_prompts_do_not_guess_garbled_terms_or_endorse_fragments(self) -> None:
        self.assertIn("не выдавай предположение", REALTIME_SYSTEM_PROMPT.casefold())
        self.assertIn("что такое когда восстающий вентиляционный", TRANSCRIPT_DECISION_SYSTEM_PROMPT.casefold())
        self.assertIn("[[unclear]]", TRANSCRIPT_DECISION_SYSTEM_PROMPT.casefold())
        self.assertIn("не знает ответ", DIALOGUE_SUMMARY_SYSTEM_PROMPT.casefold())
        self.assertIn("слова конкретного участника", DIALOGUE_SUMMARY_SYSTEM_PROMPT.casefold())

    def test_known_garbled_mining_question_requires_clarification(self) -> None:
        self.assertTrue(
            requires_transcript_clarification(
                "Что такое когда восстающий вентиляционный"
            )
        )
        self.assertTrue(
            requires_transcript_clarification(
                "Что такое тогда восстающий вентиляционный"
            )
        )
        self.assertFalse(
            requires_transcript_clarification(
                "Что такое вентиляционный восстающий?"
            )
        )

    def test_mining_question_gets_verified_terms_reference(self) -> None:
        messages = build_transcript_decision_messages(
            "Чем вертикальная горная выработка отличается от горизонтальной?",
            "",
            self.config,
        )

        self.assertIn("ГОСТ Р 57719-2017", messages[0].content)
        self.assertIn("нельзя выводить направление вдоль падения", messages[0].content)

    def test_ordinary_work_question_does_not_get_mining_reference(self) -> None:
        for question in (
            "С какими трудностями столкнулся сотрудник?",
            "Как вы выработали решение и оценили пластичность архитектуры?",
            "Как прошло восстановление после сбоя?",
            "Как рассчитывается выработка электроэнергии?",
            "Каковы обязанности горничной?",
            "Как проявляется горная болезнь?",
            "Где начинается эта горная река?",
            "Почему восстающий народ поддержал реформу?",
        ):
            with self.subTest(question=question):
                messages = build_transcript_decision_messages(
                    question,
                    "",
                    self.config,
                )

                self.assertNotIn("ГОСТ Р 57719-2017", messages[0].content)


if __name__ == "__main__":
    unittest.main()
