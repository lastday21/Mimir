import unittest

from mimir.config import AppConfig


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

        self.assertEqual(config.audio_mode, "yandex_realtime")

    def test_old_speechkit_mode_migrates_to_realtime(self) -> None:
        config = AppConfig.from_dict(
            {
                "llmProvider": "yandex_ai_studio",
                "audioMode": "speechkit",
            }
        )

        self.assertEqual(config.audio_mode, "yandex_realtime")

    def test_internal_speechkit_mode_is_not_saved_as_user_choice(self) -> None:
        config = AppConfig(audio_mode="speechkit")

        self.assertEqual(config.to_dict()["audioMode"], "yandex_realtime")


if __name__ == "__main__":
    unittest.main()
