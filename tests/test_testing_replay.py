import tempfile
import threading
import unittest
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

from mimir.audio.live import LiveAudioConfig, LiveAudioController
from mimir.audio.recordings import CallRecordingStore
from mimir.models import SpeechRecognitionResult
from mimir.testing_replay import ReplayEventLog, TestingReplayController as ReplayController


def pcm(value: int, frames: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * frames


class FakeRecognizer:
    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str,
        sample_rate_hertz: int,
    ) -> Iterator[SpeechRecognitionResult]:
        samples = array("h")
        for chunk in chunks:
            samples.frombytes(chunk)
        if not samples:
            return
        peak = max(abs(sample) for sample in samples)
        text = "Как работает проверка?" if peak < 1_500 else "Я отвечаю на вопрос"
        yield SpeechRecognitionResult(text=text, is_final=True, end_of_utterance=True)


class FakePcmSource:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def chunks(self, stop_event: threading.Event):
        for chunk in self._chunks:
            if stop_event.is_set():
                return
            yield chunk


class CancelableBlockingRecognizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def stream_lpcm(self, chunks: Iterable[bytes], **_payload):
        for _chunk in chunks:
            self.started.set()
            self.cancelled.wait(10)
            return
        if False:
            yield SpeechRecognitionResult(text="", is_final=False)

    def cancel(self) -> None:
        self.cancelled.set()


class BlockingRecognizer:
    def __init__(self, release: threading.Event) -> None:
        self.started = threading.Event()
        self.release = release

    def stream_lpcm(self, chunks: Iterable[bytes], **_payload):
        for _chunk in chunks:
            self.started.set()
            self.release.wait(10)
            return
        if False:
            yield SpeechRecognitionResult(text="", is_final=False)


class FakeConfig:
    audio_mode = "speechkit"


class FakeSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sinks = []
        self._exchanges: list[dict[str, str]] = []
        self.override = None

    def start(self):
        self._publish("session_state", {"state": "listening"})
        return self.snapshot()

    def stop(self):
        self._publish("session_state", {"state": "stopped"})
        return self.snapshot()

    def ingest_transcript(self, source: str, text: str, **_payload):
        self._publish("transcript", {"source": source, "text": text, "isFinal": True})
        if source == "remote":
            self._exchanges = [
                {
                    "question": text,
                    "hint": "Короткая подсказка",
                    "userAnswer": "",
                }
            ]
            self._publish("question", {"question": text, "questionId": "question-1"})
            self._publish("answer_done", {"questionId": "question-1"})
        elif self._exchanges:
            self._exchanges[-1]["userAnswer"] = text
        return self.snapshot()

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        self._publish(event, payload)

    def add_event_sink(self, sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_event_sink(self, sink) -> None:
        with self._lock:
            self._sinks = [item for item in self._sinks if item is not sink]

    def set_answer_provider_override(self, value) -> None:
        self.override = value

    def is_processing(self) -> bool:
        return False

    def snapshot(self):
        with self._lock:
            return {"memory": {"exchanges": [dict(item) for item in self._exchanges]}}

    def _publish(self, event: str, payload: dict[str, object]) -> None:
        with self._lock:
            sinks = tuple(self._sinks)
        for sink in sinks:
            sink(event, payload)


class TestingReplayTests(unittest.TestCase):
    def test_enabled_live_session_records_two_tracks_for_later_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            session = FakeSession()
            controller = LiveAudioController(
                session,
                lambda _key: FakeRecognizer(),
                lambda source, _config: FakePcmSource(
                    [pcm(1_000 if source == "remote" else 2_000, 3_200)]
                ),
                recording_store=store,
            )

            controller.start(
                LiveAudioConfig(
                    sources=("remote", "mic"),
                    chunk_duration_ms=200,
                    vad_enabled=False,
                    record_testing=True,
                ),
                "test-key",
            )
            deadline = threading.Event()
            for _ in range(100):
                if not controller.snapshot()["running"]:
                    break
                deadline.wait(0.01)

            recordings = store.list()
            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0]["status"], "complete")
            self.assertTrue(store.verify(recordings[0]["id"])["ok"])

    def test_replays_both_tracks_through_normal_audio_path_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            started_ns = 1_000_000_000
            store.start("call-1", started_monotonic_ns=started_ns)
            store.write("remote", pcm(1_000, 4_800), captured_at_ns=started_ns + 300_000_000)
            store.write("mic", pcm(2_000, 4_800), captured_at_ns=started_ns + 300_000_000)
            store.finish()
            session = FakeSession()
            replay = ReplayController(
                store,
                session,
                lambda: FakeConfig(),
                lambda _name: "test-key",
                lambda _key: FakeRecognizer(),
                lambda _key: FakeRecognizer(),
            )

            started = replay.start("call-1")
            finished = replay.wait(timeout=5)

            self.assertEqual(started["state"], "running")
            self.assertEqual(finished["state"], "completed")
            self.assertEqual(
                finished["report"],
                {
                    "remoteTurns": 1,
                    "micTurns": 1,
                    "questions": 1,
                    "answers": 1,
                    "duplicates": 0,
                    "errors": 0,
                },
            )
            self.assertIsNone(session.override)
            reports = list((Path(temporary) / "call-1" / "runs").glob("*/report.json"))
            events = list((Path(temporary) / "call-1" / "runs").glob("*/events.jsonl"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(len(events), 1)

    def test_stop_finishes_quickly_and_cleans_session_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            started_ns = 1_000_000_000
            store.start("long-call", started_monotonic_ns=started_ns)
            store.write("remote", pcm(1_000, 160_000), captured_at_ns=started_ns + 10_000_000_000)
            store.write("mic", pcm(2_000, 160_000), captured_at_ns=started_ns + 10_000_000_000)
            store.finish()
            session = FakeSession()
            replay = ReplayController(
                store,
                session,
                lambda: FakeConfig(),
                lambda _name: "test-key",
                lambda _key: FakeRecognizer(),
                lambda _key: FakeRecognizer(),
            )

            replay.start("long-call")
            threading.Event().wait(0.1)
            replay.stop()
            finished = replay.wait(timeout=5)

            self.assertEqual(finished["state"], "stopped")
            self.assertEqual(session._sinks, [])
            self.assertIsNone(session.override)

    def test_stop_cancels_blocked_recognizers_and_leaves_no_audio_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            started_ns = 1_000_000_000
            store.start("blocked-call", started_monotonic_ns=started_ns)
            store.write("remote", pcm(1_000, 32_000), captured_at_ns=started_ns + 2_000_000_000)
            store.write("mic", pcm(2_000, 32_000), captured_at_ns=started_ns + 2_000_000_000)
            store.finish()
            session = FakeSession()
            recognizers: list[CancelableBlockingRecognizer] = []

            def recognizer_factory(_key: str) -> CancelableBlockingRecognizer:
                recognizer = CancelableBlockingRecognizer()
                recognizers.append(recognizer)
                return recognizer

            replay = ReplayController(
                store,
                session,
                lambda: FakeConfig(),
                lambda _name: "test-key",
                recognizer_factory,
                recognizer_factory,
            )

            replay.start("blocked-call")
            deadline = threading.Event()
            for _ in range(100):
                if len(recognizers) == 2 and all(item.started.is_set() for item in recognizers):
                    break
                deadline.wait(0.02)
            replay.stop()
            finished = replay.wait(timeout=5)

            self.assertEqual(finished["state"], "stopped")
            self.assertFalse(replay.has_live_audio())
            self.assertTrue(all(item.cancelled.is_set() for item in recognizers))

    def test_uncancellable_threads_fail_run_and_block_next_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            started_ns = 1_000_000_000
            store.start("uncancellable", started_monotonic_ns=started_ns)
            store.write("remote", pcm(1_000, 32_000), captured_at_ns=started_ns + 2_000_000_000)
            store.write("mic", pcm(2_000, 32_000), captured_at_ns=started_ns + 2_000_000_000)
            store.finish()
            session = FakeSession()
            release = threading.Event()
            recognizers: list[BlockingRecognizer] = []

            def recognizer_factory(_key: str) -> BlockingRecognizer:
                recognizer = BlockingRecognizer(release)
                recognizers.append(recognizer)
                return recognizer

            replay = ReplayController(
                store,
                session,
                lambda: FakeConfig(),
                lambda _name: "test-key",
                recognizer_factory,
                recognizer_factory,
            )

            replay.start("uncancellable")
            waiter = threading.Event()
            for _ in range(100):
                if len(recognizers) == 2 and all(item.started.is_set() for item in recognizers):
                    break
                waiter.wait(0.02)
            replay.stop()
            finished = replay.wait(timeout=5)

            self.assertEqual(finished["state"], "failed")
            self.assertTrue(replay.has_live_audio())
            with self.assertRaisesRegex(ValueError, "звуковой поток"):
                replay.start("uncancellable")

            release.set()
            for _ in range(100):
                if not replay.has_live_audio():
                    break
                waiter.wait(0.02)
            self.assertFalse(replay.has_live_audio())

    def test_final_refinement_keeps_one_turn_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_log = ReplayEventLog(Path(temporary) / "events.jsonl")
            event_log(
                "transcript",
                {
                    "source": "remote",
                    "turnId": "turn-1",
                    "text": "Сколько будет четыре плюс пять",
                    "isFinal": True,
                    "operation": "append",
                },
            )
            event_log(
                "transcript",
                {
                    "source": "remote",
                    "turnId": "turn-1",
                    "text": "Сколько будет 4 + 5",
                    "isFinal": True,
                    "operation": "replace",
                },
            )
            event_log("question", {"question": "Сколько будет 4 + 5"})
            event_log.close()

            summary = event_log.summary()
            self.assertEqual(summary["remoteTurns"], 1)
            self.assertEqual(summary["questionsBeforeFinal"], 0)

    def test_cleanup_survives_session_snapshot_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=200)
            started_ns = 1_000_000_000
            store.start("snapshot-error", started_monotonic_ns=started_ns)
            store.write("remote", pcm(1_000, 4_800), captured_at_ns=started_ns + 300_000_000)
            store.write("mic", pcm(2_000, 4_800), captured_at_ns=started_ns + 300_000_000)
            store.finish()
            session = FakeSession()
            replay = ReplayController(
                store,
                session,
                lambda: FakeConfig(),
                lambda _name: "test-key",
                lambda _key: FakeRecognizer(),
                lambda _key: FakeRecognizer(),
            )

            replay.start("snapshot-error")
            session.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("snapshot failed"))
            finished = replay.wait(timeout=5)

            self.assertEqual(finished["state"], "failed")
            self.assertIn("snapshot failed", finished["error"])
            self.assertEqual(session._sinks, [])
            self.assertIsNone(session.override)


if __name__ == "__main__":
    unittest.main()
