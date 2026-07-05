import unittest

from mimir.models import ModelInfo
from mimir.ollama_fallback import select_preferred_model


class OllamaFallbackTests(unittest.TestCase):
    def test_prefers_qwen3_8b_for_local_fallback(self) -> None:
        models = [
            ModelInfo("llama3.2:3b", "llama3.2:3b", "ollama", 131_072),
            ModelInfo("qwen3:4b", "qwen3:4b", "ollama", 32_768),
            ModelInfo("qwen3:8b", "qwen3:8b", "ollama", 32_768),
        ]
        self.assertEqual(select_preferred_model(models).id, "qwen3:8b")

    def test_keeps_qwen_family_ahead_of_larger_general_models(self) -> None:
        models = [
            ModelInfo("llama3.1:70b", "llama3.1:70b", "ollama", 131_072),
            ModelInfo("qwen2.5:7b-instruct", "qwen2.5:7b-instruct", "ollama", 32_768),
        ]
        self.assertEqual(select_preferred_model(models).id, "qwen2.5:7b-instruct")


if __name__ == "__main__":
    unittest.main()
