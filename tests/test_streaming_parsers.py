import unittest
from unittest.mock import patch

from mimir.models import ChatMessage
from mimir.providers.base import ProviderError
from mimir.providers.yandex_ai import YandexAIStudioClient
from mimir.providers.ollama import parse_ollama_delta
from mimir.providers.yandex_ai import parse_openai_delta


class StreamingParserTests(unittest.TestCase):
    def test_yandex_stream_wraps_timeout_as_provider_error(self) -> None:
        client = YandexAIStudioClient("test-key", "folder-id")

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(ProviderError, "timed out"):
                list(client.stream_chat("yandexgpt/latest", [ChatMessage("user", "test")]))

    def test_parses_openai_compatible_delta(self) -> None:
        payload = '{"choices":[{"delta":{"content":"ответ"}}]}'

        self.assertEqual(parse_openai_delta(payload), "ответ")

    def test_parses_ollama_delta(self) -> None:
        payload = '{"message":{"content":"ответ"},"done":false}'

        chunk, done = parse_ollama_delta(payload)

        self.assertEqual(chunk, "ответ")
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
