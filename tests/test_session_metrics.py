import json
import os
import tempfile
import time
import unittest

from mimir.live_trace import current_trace_path, reset_trace_for_tests
from mimir.models import ChatMessage
from mimir.session import SessionManager


class FastSessionManager(SessionManager):
    def _stream_answer(self, _messages: list[ChatMessage]):
        yield "Короткая подсказка."


class SessionMetricsTests(unittest.TestCase):
    def test_records_question_metrics_and_trace_events(self) -> None:
        original_dir = os.environ.get("MIMIR_LIVE_TRACE_DIR")
        original_enabled = os.environ.get("MIMIR_LIVE_TRACE")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["MIMIR_LIVE_TRACE_DIR"] = directory
                os.environ["MIMIR_LIVE_TRACE"] = "1"
                reset_trace_for_tests()

                manager = FastSessionManager()
                manager.start()
                manager.record_audio_speech_started("remote")
                manager.record_audio_chunk("remote", 3200)
                manager.record_stt_result("remote", False)
                manager.record_stt_result("remote", True)
                manager.ingest_transcript("remote", "Как бы вы ускорили очередь задач?")
                self.wait_for_answer(manager)

                metrics = manager.metrics()
                question = metrics["questions"][-1]

                self.assertEqual(question["source"], "remote")
                self.assertIn("tAudioChunkMs", question)
                self.assertIn("tSttInterimMs", question)
                self.assertIn("tSttFinalMs", question)
                self.assertIn("tDetectMs", question)
                self.assertIn("tContextBuildMs", question)
                self.assertIn("tLlmTtfbMs", question)
                self.assertIn("tFirstHintMs", question)
                self.assertIn("tAnswerDoneMs", question)
                self.assertEqual(metrics["currentQuestionId"], question["questionId"])

                events = [
                    json.loads(line)["event"]
                    for line in current_trace_path().read_text(encoding="utf-8").splitlines()
                ]
                self.assertIn("metric.stage", events)
                self.assertIn("metric.question", events)
        finally:
            if original_dir is None:
                os.environ.pop("MIMIR_LIVE_TRACE_DIR", None)
            else:
                os.environ["MIMIR_LIVE_TRACE_DIR"] = original_dir
            if original_enabled is None:
                os.environ.pop("MIMIR_LIVE_TRACE", None)
            else:
                os.environ["MIMIR_LIVE_TRACE"] = original_enabled
            reset_trace_for_tests()

    def test_metrics_reset_on_new_session_start_after_pause(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_chunk("remote", 3200)
        manager.record_stt_result("remote", True)
        manager.ingest_transcript("remote", "Как бы вы ускорили очередь задач?")
        self.wait_for_answer(manager)

        manager.pause()
        clean = manager.start()

        self.assertEqual(clean["metrics"]["questions"], [])
        self.assertEqual(clean["metrics"]["sources"], {})

    def wait_for_answer(self, manager: SessionManager) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            questions = manager.metrics()["questions"]
            if questions and "tAnswerDoneMs" in questions[-1]:
                return
            time.sleep(0.01)
        self.fail("answer metrics were not completed")


if __name__ == "__main__":
    unittest.main()
