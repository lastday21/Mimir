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
        self.assertIn("Верни [[SKIP]]", manager.calls[0][0].content)
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

    def test_refinement_replaces_final_before_single_model_call(self) -> None:
        manager = DecisionSessionManager(["[[ANSWER]]", "Назовите сроки и ответственного."])
        manager.start()

        manager.ingest_transcript("remote", "когда будет готова вчерне и в цвету")
        manager.ingest_transcript(
            "remote",
            "Когда будет готово в черновом и чистовом виде?",
            is_refinement=True,
        )

        self.wait_until(lambda: self.answer_is_done(manager))
        snapshot = manager.snapshot()

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(
            snapshot["memory"]["exchanges"][-1]["question"],
            "Когда будет готово в черновом и чистовом виде?",
        )

    def test_unclear_marker_does_not_create_invented_answer(self) -> None:
        manager = DecisionSessionManager(["[[UNC", "LEAR]]"])
        manager.start()

        manager.ingest_transcript("remote", "Тимур, а что там по вчернецвету?")
        self.wait_until(lambda: manager.metrics().get("unclearUtterances") == 1)
        snapshot = manager.snapshot()

        self.assertIsNone(snapshot["currentQuestion"])
        self.assertEqual(snapshot["metrics"]["questions"], [])
        uncertain = [event for event in manager._events if event.event == "transcript_uncertain"]
        self.assertEqual(len(uncertain), 1)
        self.assertIn("Лучше переспросить", uncertain[0].payload["message"])

    def test_known_garbled_question_is_not_sent_to_model(self) -> None:
        for utterance in (
            "Что такое когда восстающий вентиляционный",
            "Что такое тогда восстающий вентиляционный",
        ):
            with self.subTest(utterance=utterance):
                manager = DecisionSessionManager(["[[ANSWER]]Придуманный ответ"])
                manager.start()

                manager.ingest_transcript("remote", utterance)
                self.wait_until(lambda: manager.metrics().get("unclearUtterances") == 1)

                self.assertEqual(manager.calls, [])
                self.assertIsNone(manager.snapshot()["currentQuestion"])

    def test_garbled_question_closes_previous_answer_binding(self) -> None:
        manager = DecisionSessionManager(["[[ANSWER]]", "Назовите шахтный ствол."])
        manager.start()
        manager.ingest_transcript(
            "remote",
            "Какие вертикальные выработки вы можете привести в пример?",
        )
        self.wait_until(lambda: self.answer_is_done(manager))

        manager.ingest_transcript(
            "remote",
            "Что такое тогда восстающий вентиляционный",
        )
        self.wait_until(lambda: manager.metrics().get("unclearUtterances") == 1)
        manager.ingest_transcript("mic", "Он направлен к поверхности")

        snapshot = manager.snapshot()
        self.assertIsNone(snapshot["currentQuestion"])
        self.assertEqual(snapshot["memory"]["exchanges"][-1]["userAnswer"], "")
        self.assertTrue(snapshot["memory"]["turns"][-2]["uncertain"])
        self.assertNotIn("восстающий вентиляционный", manager._memory.summary_source()[0])

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

    def test_unclear_marker_is_parsed_before_stream_finishes(self) -> None:
        decision = parse_model_decision("[[UNCLEAR]]")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "unclear")

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
