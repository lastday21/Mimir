import unittest

from mimir.providers.ollama import parse_ollama_delta
from mimir.providers.yandex_ai import parse_openai_delta


class StreamingParserTests(unittest.TestCase):
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
