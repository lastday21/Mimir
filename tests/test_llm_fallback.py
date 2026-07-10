import time
import unittest

import mimir.session as session_module
from mimir.config import AppConfig
from mimir.models import ChatMessage, ModelInfo
from mimir.providers.base import ProviderError
from mimir.session import SessionManager


class LlmFallbackTests(unittest.TestCase):
    def test_unexpected_answer_error_marks_session_degraded(self) -> None:
        class FailingSessionManager(SessionManager):
            def _stream_answer(self, _messages: list[ChatMessage]):
                raise TimeoutError("request timed out")

        manager = FailingSessionManager()
        manager.start()
        manager.trigger_question("Как работает очередь?", confidence=0.9, reason="test")
        self.wait_for_error(manager)
        events = manager.listen(after=0)
        answer_error = None
        for _ in range(8):
            event = next(events)
            if event.event == "answer_error":
                answer_error = event
                break

        self.assertIsNotNone(answer_error)
        self.assertEqual(answer_error.payload["error"], "request timed out")
        self.assertEqual(manager.snapshot()["metrics"]["errorPhase"], "answer")

    def test_yandex_failure_falls_back_to_preferred_ollama_model(self) -> None:
        original_load_config = session_module.load_config
        original_read_secret = session_module.read_secret
        original_yandex = session_module.YandexAIStudioClient
        original_ollama = session_module.OllamaClient

        class FailingYandexClient:
            def __init__(self, _key: str, _folder_id: str) -> None:
                pass

            def stream_chat(self, _model: str, _messages: list[ChatMessage]):
                raise ProviderError("yandex down")

        class FakeOllamaClient:
            selected_model = ""

            def __init__(self, _base_url: str) -> None:
                pass

            def list_models(self) -> list[ModelInfo]:
                return [
                    ModelInfo("llama3.2:3b", "llama3.2:3b", "ollama", 131_072),
                    ModelInfo("qwen3:8b", "qwen3:8b", "ollama", 32_768),
                ]

            def stream_chat(self, model: str, _messages: list[ChatMessage]):
                FakeOllamaClient.selected_model = model
                yield "Локальная подсказка."

        try:
            session_module.load_config = lambda: AppConfig(
                yandex_folder_id="folder-id",
                llm_provider="yandex_ai_studio",
                llm_model="yandexgpt/latest",
                ollama_base_url="http://localhost:11434",
            )
            session_module.read_secret = lambda _name: "test-key"
            session_module.YandexAIStudioClient = FailingYandexClient
            session_module.OllamaClient = FakeOllamaClient

            manager = SessionManager()
            manager.start()
            manager.trigger_question("Как ускорить обработку очереди?", confidence=0.9, reason="test")
            question = self.wait_for_answer(manager)

            self.assertEqual(FakeOllamaClient.selected_model, "qwen3:8b")
            self.assertEqual(question["provider"], "ollama")
            self.assertTrue(question["fallbackUsed"])
            self.assertEqual(question["fallbackReason"], "yandex down")
        finally:
            session_module.load_config = original_load_config
            session_module.read_secret = original_read_secret
            session_module.YandexAIStudioClient = original_yandex
            session_module.OllamaClient = original_ollama

    def test_does_not_mix_ollama_after_partial_yandex_answer(self) -> None:
        original_load_config = session_module.load_config
        original_read_secret = session_module.read_secret
        original_yandex = session_module.YandexAIStudioClient
        original_ollama = session_module.OllamaClient

        class PartialYandexClient:
            def __init__(self, _key: str, _folder_id: str) -> None:
                pass

            def stream_chat(self, _model: str, _messages: list[ChatMessage]):
                yield "Начало ответа."
                raise ProviderError("stream lost")

        class FakeOllamaClient:
            called = False

            def __init__(self, _base_url: str) -> None:
                pass

            def list_models(self) -> list[ModelInfo]:
                FakeOllamaClient.called = True
                return [ModelInfo("qwen3:8b", "qwen3:8b", "ollama", 32_768)]

            def stream_chat(self, _model: str, _messages: list[ChatMessage]):
                FakeOllamaClient.called = True
                yield "Не должен использоваться."

        try:
            session_module.load_config = lambda: AppConfig(
                yandex_folder_id="folder-id",
                llm_provider="yandex_ai_studio",
                llm_model="yandexgpt/latest",
                ollama_base_url="http://localhost:11434",
            )
            session_module.read_secret = lambda _name: "test-key"
            session_module.YandexAIStudioClient = PartialYandexClient
            session_module.OllamaClient = FakeOllamaClient

            manager = SessionManager()
            manager.start()
            manager.trigger_question("Как ускорить обработку очереди?", confidence=0.9, reason="test")
            self.wait_for_error(manager)

            self.assertFalse(FakeOllamaClient.called)
            self.assertFalse(manager.metrics()["questions"][-1]["fallbackUsed"])
        finally:
            session_module.load_config = original_load_config
            session_module.read_secret = original_read_secret
            session_module.YandexAIStudioClient = original_yandex
            session_module.OllamaClient = original_ollama

    def test_forced_ollama_override_selects_preferred_local_model(self) -> None:
        original_load_config = session_module.load_config
        original_ollama = session_module.OllamaClient

        class FakeOllamaClient:
            selected_model = ""

            def __init__(self, _base_url: str) -> None:
                pass

            def list_models(self) -> list[ModelInfo]:
                return [
                    ModelInfo("llama3.2:3b", "llama3.2:3b", "ollama", 131_072),
                    ModelInfo("qwen3:8b", "qwen3:8b", "ollama", 32_768),
                ]

            def stream_chat(self, model: str, _messages: list[ChatMessage]):
                FakeOllamaClient.selected_model = model
                yield "Локальный ответ."

        try:
            session_module.load_config = lambda: AppConfig(
                yandex_folder_id="folder-id",
                llm_provider="yandex_ai_studio",
                llm_model="yandexgpt/latest",
                ollama_base_url="http://localhost:11434",
            )
            session_module.OllamaClient = FakeOllamaClient

            manager = SessionManager()
            manager.set_answer_provider_override("ollama")
            manager.start()
            manager.trigger_question("Как ускорить обработку очереди?", confidence=0.9, reason="test")
            question = self.wait_for_answer(manager)

            self.assertEqual(FakeOllamaClient.selected_model, "qwen3:8b")
            self.assertEqual(question["provider"], "ollama")
        finally:
            session_module.load_config = original_load_config
            session_module.OllamaClient = original_ollama

    def wait_for_answer(self, manager: SessionManager) -> dict[str, object]:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            questions = manager.metrics()["questions"]
            if questions and "tAnswerDoneMs" in questions[-1]:
                return questions[-1]
            time.sleep(0.01)
        self.fail("answer metrics were not completed")

    def wait_for_error(self, manager: SessionManager) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if manager.snapshot()["state"] == "degraded":
                return
            time.sleep(0.01)
        self.fail("session did not enter degraded state")


if __name__ == "__main__":
    unittest.main()
