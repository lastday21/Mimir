import time
import unittest

from mimir.models import ChatMessage
from mimir.session import SessionManager, parse_model_decision


class DecisionSessionManager(SessionManager):
    def __init__(self, response: list[str]) -> None:
        super().__init__()
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    def _stream_answer(self, messages: list[ChatMessage]):
        self.calls.append(messages)
        yield from self.response


class ModelDecisionTests(unittest.TestCase):
    def test_model_skips_ordinary_remote_statement(self) -> None:
        manager = DecisionSessionManager(["[[SK", "IP]]"])
        manager.start()

        manager.ingest_transcript("remote", "Сегодня мы обсуждаем устройство команды")
        self.wait_until(lambda: manager.metrics().get("skippedUtterances") == 1)
        snapshot = manager.snapshot()

        self.assertEqual(len(manager.calls), 1)
        self.assertIn("Если подсказка не нужна", manager.calls[0][0].content)
        self.assertIsNone(snapshot["currentQuestion"])
        self.assertEqual(snapshot["metrics"]["questions"], [])

    def test_model_decides_and_streams_answer_in_one_request(self) -> None:
        manager = DecisionSessionManager(["[[ANS", "WER]]", "Сначала обозначьте цель", ", затем ограничения."])
        manager.start()

        manager.ingest_transcript("remote", "Расскажите, как вы проектировали очередь задач")
        self.wait_until(lambda: self.answer_is_done(manager))
        snapshot = manager.snapshot()
        exchange = snapshot["memory"]["exchanges"][-1]

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(snapshot["currentQuestion"]["reason"], "model")
        self.assertEqual(snapshot["currentAnswer"]["text"], "Сначала обозначьте цель, затем ограничения.")
        self.assertEqual(exchange["question"], "Расскажите, как вы проектировали очередь задач")
        self.assertEqual(exchange["hint"], "Сначала обозначьте цель, затем ограничения.")

    def test_model_is_not_called_for_interim_or_user_speech(self) -> None:
        manager = DecisionSessionManager(["[[SKIP]]"])
        manager.start()

        manager.ingest_transcript("remote", "Расскажите", is_final=False)
        manager.ingest_transcript("mic", "Я отвечаю", is_final=True)
        time.sleep(0.05)

        self.assertEqual(manager.calls, [])

    def test_exact_repeated_remote_utterance_is_checked_once(self) -> None:
        manager = DecisionSessionManager(["[[SKIP]]"])
        manager.start()

        manager.ingest_transcript("remote", "Мы закончили этот раздел")
        self.wait_until(lambda: manager.metrics().get("skippedUtterances") == 1)
        manager.ingest_transcript("remote", "Мы закончили этот раздел")
        time.sleep(0.05)

        self.assertEqual(len(manager.calls), 1)

    def test_short_unmarked_answer_is_kept_when_stream_finishes(self) -> None:
        decision = parse_model_decision("Да.", final=True)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.text, "Да.")

    def test_skip_marker_inside_code_fence_is_not_shown_as_answer(self) -> None:
        decision = parse_model_decision("```\n[[SKIP]]\n```")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "skip")

    @staticmethod
    def answer_is_done(manager: SessionManager) -> bool:
        questions = manager.metrics()["questions"]
        return bool(questions and "tAnswerDoneMs" in questions[-1])

    def wait_until(self, predicate) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("condition was not met")


if __name__ == "__main__":
    unittest.main()
