import unittest

from mimir.config import AppConfig, ConversationSettings, UserProfile
from mimir.profile_context import select_profile_facts


class ProfileContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            profile=UserProfile(
                name="Тимур",
                role="Разработчик",
                background="Разрабатываю серверные приложения на Python.",
                projects=(
                    "Построил обработку событий через Kafka и отдельные воркеры.\n"
                    "Разработал мобильное приложение на Flutter для курьеров."
                ),
                stories=(
                    "Устранил отставание очереди Kafka с помощью пакетной обработки.\n"
                    "Увеличил продажи интернет-магазина после изменения каталога."
                ),
            ),
            conversation=ConversationSettings(mode="interview"),
        )

    def test_selects_only_facts_related_to_current_topic(self) -> None:
        facts = select_profile_facts(
            self.config,
            "Как вы решали отставание очереди Kafka и масштабировали воркеры?",
        )
        text = "\n".join(fact.text for fact in facts)

        self.assertIn("серверные приложения на Python", text)
        self.assertIn("событий через Kafka", text)
        self.assertIn("отставание очереди Kafka", text)
        self.assertNotIn("Flutter", text)
        self.assertNotIn("продажи интернет-магазина", text)

    def test_generic_project_question_uses_one_project_instead_of_full_profile(self) -> None:
        facts = select_profile_facts(self.config, "Расскажите о вашем проекте")
        text = "\n".join(fact.text for fact in facts)

        self.assertIn("событий через Kafka", text)
        self.assertNotIn("Flutter", text)
        self.assertNotIn("продажи интернет-магазина", text)


if __name__ == "__main__":
    unittest.main()
