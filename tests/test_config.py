import unittest

from mimir.config import (
    AppConfig,
    AudioApplicationSettings,
    ConversationSettings,
    TestingSettings as AppTestingSettings,
    UserProfile,
)


class AppConfigTests(unittest.TestCase):
    def test_ollama_uses_local_audio(self) -> None:
        config = AppConfig.from_dict(
            {
                "llmProvider": "ollama",
                "llmModel": "qwen3:8b",
                "audioMode": "yandex_realtime",
            }
        )

        self.assertEqual(config.audio_mode, "local_vosk")

    def test_yandex_does_not_keep_local_audio(self) -> None:
        config = AppConfig.from_dict(
            {
                "llmProvider": "yandex_ai_studio",
                "llmModel": "yandexgpt/latest",
                "audioMode": "local_vosk",
            }
        )

        self.assertEqual(config.audio_mode, "speechkit")

    def test_old_realtime_mode_migrates_to_speechkit(self) -> None:
        config = AppConfig.from_dict(
            {
                "llmProvider": "yandex_ai_studio",
                "audioMode": "yandex_realtime",
            }
        )

        self.assertEqual(config.audio_mode, "speechkit")

    def test_speechkit_mode_is_saved_as_user_choice(self) -> None:
        config = AppConfig(audio_mode="speechkit")

        self.assertEqual(config.to_dict()["audioMode"], "speechkit")

    def test_profile_and_conversation_round_trip(self) -> None:
        config = AppConfig(
            profile=UserProfile(
                name="Тимур",
                role="Разработчик",
                background="Пять лет в серверной разработке",
                projects="Сервис обработки заказов",
                stories="Ускорил очередь задач",
            ),
            conversation=ConversationSettings(
                mode="meeting",
                goal="Понимать, что от меня требуется",
                context="Еженедельная встреча команды",
            ),
            audio_application=AudioApplicationSettings(
                process_id=42,
                executable="meeting.exe",
                title="Рабочий созвон",
            ),
            testing=AppTestingSettings(enabled=True),
            setup_completed=True,
        )

        restored = AppConfig.from_dict(config.to_dict())

        self.assertEqual(restored.profile.name, "Тимур")
        self.assertEqual(restored.profile.projects, "Сервис обработки заказов")
        self.assertEqual(restored.conversation.mode, "meeting")
        self.assertEqual(restored.conversation.goal, "Понимать, что от меня требуется")
        self.assertEqual(restored.audio_application.process_id, 42)
        self.assertEqual(restored.audio_application.executable, "meeting.exe")
        self.assertTrue(restored.testing.enabled)
        self.assertTrue(restored.setup_completed)

    def test_testing_is_disabled_by_default(self) -> None:
        config = AppConfig.from_dict({})

        self.assertFalse(config.testing.enabled)
        self.assertEqual(config.to_dict()["testing"], {"enabled": False})

    def test_unknown_conversation_mode_uses_interview(self) -> None:
        config = AppConfig.from_dict({"conversation": {"mode": "unknown"}})

        self.assertEqual(config.conversation.mode, "interview")


if __name__ == "__main__":
    unittest.main()
