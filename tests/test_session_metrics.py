import json
import os
import tempfile
import threading
import time
import unittest

from mimir.live_trace import current_trace_path, reset_trace_for_tests
from mimir.models import ChatMessage
from mimir.session import SessionManager


class FastSessionManager(SessionManager):
    def _stream_answer(self, _messages: list[ChatMessage]):
        yield "Короткая подсказка."


class ControlledSessionManager(SessionManager):
    def __init__(self) -> None:
        super().__init__()
        self.provider_started = threading.Event()
        self.release_provider = threading.Event()

    def _stream_answer(self, _messages: list[ChatMessage]):
        self.provider_started.set()
        self.release_provider.wait(2)
        yield "[[ANSWER]]Короткая подсказка."


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
                time.sleep(0.02)
                manager.record_audio_speech_ended("remote", trailing_silence_ms=2_000)
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
                self.assertIn("tQuestionToFirstHintMs", question)
                self.assertIn("tQuestionToAnswerDoneMs", question)
                self.assertEqual(question["latencyBaseline"], "speech_end")
                self.assertEqual(question["latencySchemaVersion"], 2)
                self.assertEqual(question["vadTailMs"], 2_000)
                self.assertLess(question["trailingSilenceMs"], 100)
                source = metrics["sources"]["remote"]
                self.assertGreaterEqual(source["speechEndedAtMs"], source["speechStartedAtMs"])
                self.assertGreaterEqual(question["tFirstHintMs"], question["tQuestionToFirstHintMs"])
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

    def test_warns_when_detected_speech_produces_no_final_transcript(self) -> None:
        captured: list[tuple[str, dict[str, object]]] = []
        manager = FastSessionManager(lambda event, payload: captured.append((event, payload)))
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")

        manager._check_missing_stt_utterance(  # noqa: SLF001 - deterministic timer check
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            1,
        )

        self.assertEqual(manager.metrics()["missingSttUtterances"], 1)
        self.assertTrue(any(event == "stt_warning" for event, _payload in captured))

    def test_delayed_first_final_does_not_hide_missing_second_utterance(self) -> None:
        manager = FastSessionManager()
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager.record_stt_result("mic", True)

        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            1,
        )
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            2,
        )

        self.assertEqual(manager.metrics()["missingSttUtterances"], 1)

    def test_late_final_clears_missing_transcript_warning(self) -> None:
        captured: list[tuple[str, dict[str, object]]] = []
        manager = FastSessionManager(lambda event, payload: captured.append((event, payload)))
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            1,
        )

        manager.record_stt_result("mic", True)

        metrics = manager.metrics()
        self.assertEqual(metrics["missingSttUtterances"], 0)
        self.assertEqual(metrics["lateSttRecoveries"], 1)
        self.assertTrue(any(event == "stt_recovered" for event, _payload in captured))

    def test_new_final_does_not_recover_an_older_missing_utterance(self) -> None:
        captured: list[tuple[str, dict[str, object]]] = []
        manager = FastSessionManager(lambda event, payload: captured.append((event, payload)))
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            1,
        )

        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager.record_stt_result("mic", True)
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            2,
        )

        metrics = manager.metrics()
        self.assertEqual(metrics["missingSttUtterances"], 1)
        self.assertEqual(metrics.get("lateSttRecoveries", 0), 0)
        self.assertNotIn(1, manager._finalized_speech_sequences["mic"])  # noqa: SLF001
        self.assertIn(2, manager._finalized_speech_sequences["mic"])  # noqa: SLF001
        self.assertFalse(any(event == "stt_recovered" for event, _payload in captured))

    def test_new_partial_links_final_to_new_utterance_before_warning(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_stt_result("mic", False)
        manager.record_audio_speech_ended("mic")

        manager.record_audio_speech_started("mic")
        manager.record_stt_result("mic", False)
        manager.record_audio_speech_ended("mic")
        manager.record_stt_result("mic", True)

        finalized = manager._finalized_speech_sequences["mic"]  # noqa: SLF001
        self.assertNotIn(1, finalized)
        self.assertIn(2, finalized)

    def test_warned_partial_does_not_capture_next_final_without_partial(self) -> None:
        manager = FastSessionManager()
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_stt_result("mic", False)
        manager.record_audio_speech_ended("mic")
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            1,
        )

        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager.record_stt_result("mic", True)

        metrics = manager.metrics()
        finalized = manager._finalized_speech_sequences["mic"]  # noqa: SLF001
        self.assertNotIn(1, finalized)
        self.assertIn(2, finalized)
        self.assertEqual(metrics["missingSttUtterances"], 1)
        self.assertEqual(metrics.get("lateSttRecoveries", 0), 0)

    def test_late_refinement_does_not_finalize_next_utterance(self) -> None:
        manager = FastSessionManager()
        started = manager.start()
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")
        manager.record_stt_result("mic", True)
        manager.record_audio_speech_started("mic")
        manager.record_audio_speech_ended("mic")

        manager.record_stt_result("mic", True, is_refinement=True)
        manager._check_missing_stt_utterance(  # noqa: SLF001
            manager._generation,  # noqa: SLF001
            str(started["sessionId"]),
            "mic",
            2,
        )

        self.assertEqual(manager.metrics()["missingSttUtterances"], 1)

    def test_question_keeps_finalized_utterance_timing_after_next_speech_starts(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_chunk("remote", 3200)
        manager.record_audio_speech_ended("remote", trailing_silence_ms=2_000)
        manager.record_stt_result("remote", True)
        first_end_ms = manager.metrics()["sources"]["remote"]["speechEndedAtMs"]

        manager.ingest_transcript("remote", "Как бы вы ускорили очередь задач?")
        manager.record_audio_speech_started("remote")
        self.wait_for_answer(manager)

        question = manager.metrics()["questions"][-1]
        self.assertEqual(question["utteranceEndedAtMs"], first_end_ms)
        self.assertEqual(question["latencyBaseline"], "speech_end")

    def test_late_vad_end_replaces_stt_final_latency_baseline(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_chunk("remote", 3200)
        manager.record_stt_result("remote", True)
        manager.ingest_transcript("remote", "Как бы вы ускорили очередь задач?")
        self.wait_for_answer(manager)

        before = manager.metrics()["questions"][-1]
        self.assertEqual(before["latencyBaseline"], "stt_final")

        manager.record_audio_speech_ended("remote", trailing_silence_ms=2_000)

        after = manager.metrics()["questions"][-1]
        self.assertEqual(after["latencyBaseline"], "speech_end")
        self.assertEqual(after["vadTailMs"], 2_000)
        self.assertGreaterEqual(after["tFirstHintMs"], 0)
        self.assertGreaterEqual(after["tAnswerDoneMs"], after["tFirstHintMs"])

    def test_delayed_final_uses_first_utterance_metrics_after_second_starts(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_chunk("remote", 3200)
        first_chunk_ms = manager.metrics()["sources"]["remote"]["tAudioChunkMs"]
        manager.record_audio_speech_ended("remote", trailing_silence_ms=2_000)
        first_end_ms = manager.metrics()["sources"]["remote"]["speechEndedAtMs"]

        manager.record_audio_speech_started("remote")
        manager.record_audio_chunk("remote", 6400)
        manager.record_stt_result("remote", True)
        manager.ingest_transcript("remote", "Как бы вы ускорили очередь задач?")
        self.wait_for_answer(manager)

        question = manager.metrics()["questions"][-1]
        self.assertEqual(question["utteranceSequence"], 1)
        self.assertEqual(question["utteranceEndedAtMs"], first_end_ms)
        self.assertEqual(question["tAudioChunkMs"], first_chunk_ms)

    def test_first_question_keeps_its_sequence_when_second_final_arrives(self) -> None:
        manager = ControlledSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_speech_ended("remote")
        manager.record_stt_result("remote", True)
        manager.ingest_transcript("remote", "Как устроена первая очередь?")
        self.assertTrue(manager.provider_started.wait(1))

        manager.record_audio_speech_started("remote")
        manager.record_audio_speech_ended("remote")
        manager.record_stt_result("remote", True)
        manager.ingest_transcript("remote", "Как устроена вторая очередь?")
        manager.release_provider.set()
        self.wait_for_answer(manager)

        first_question = manager.metrics()["questions"][0]
        self.assertEqual(first_question["utteranceSequence"], 1)

    def test_external_question_uses_explicit_utterance_sequence(self) -> None:
        manager = FastSessionManager()
        manager.start()
        manager.record_audio_speech_started("remote")
        manager.record_audio_speech_ended("remote")
        first_end_ms = manager.metrics()["sources"]["remote"]["speechEndedAtMs"]
        manager.record_stt_result("remote", True)

        manager.record_audio_speech_started("remote")
        manager.record_audio_speech_ended("remote")
        manager.record_stt_result("remote", True)
        manager.record_external_question(
            "external_first",
            "Первый вопрос",
            utterance_sequence=1,
        )

        question = manager.metrics()["questions"][-1]
        self.assertEqual(question["utteranceSequence"], 1)
        self.assertEqual(question["utteranceEndedAtMs"], first_end_ms)

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
