import json
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

from mimir.audio.recordings import (
    CallRecordingError,
    CallRecordingStore,
    RecordedPcmSource,
    ReplayClock,
)


def pcm(value: int, frames: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * frames


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CallRecordingStoreTests(unittest.TestCase):
    def test_writes_two_synchronized_tracks_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary, chunk_duration_ms=100)
            started_ns = 1_000_000_000
            started = store.start(
                "call-1",
                title="Проверочный созвон",
                application={"executable": "browser.exe", "apiKey": "must-not-leak"},
                started_monotonic_ns=started_ns,
                created_at_ms=1234,
            )

            self.assertEqual(started["id"], "call-1")
            self.assertEqual(started["status"], "recording")
            self.assertEqual(store.active_recording_id(), "call-1")
            self.assertTrue(
                store.write("remote", pcm(1000, 1600), captured_at_ns=started_ns + 100_000_000)
            )
            self.assertTrue(
                store.write("mic", pcm(2000, 800), captured_at_ns=started_ns + 150_000_000)
            )

            completed = store.finish()

            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["durationMs"], 150)
            self.assertEqual(completed["sources"], ["remote", "mic"])
            self.assertEqual(store.active_recording_id(), None)
            self.assertTrue(Path(completed["paths"]["remote"]).is_absolute())
            self.assertGreater(completed["sizeBytes"], 0)
            for source in ("remote", "mic"):
                self.assertEqual(completed["tracks"][source]["frames"], 2400)
                self.assertTrue(completed["tracks"][source]["receivedAudio"])
                self.assertEqual(len(completed["tracks"][source]["sha256"]), 64)

            with wave.open(completed["paths"]["remote"], "rb") as remote:
                self.assertEqual((remote.getnchannels(), remote.getsampwidth(), remote.getframerate()), (1, 2, 16000))
                self.assertEqual(remote.getnframes(), 2400)
                remote_pcm = remote.readframes(2400)
            with wave.open(completed["paths"]["mic"], "rb") as mic:
                self.assertEqual(mic.getnframes(), 2400)
                mic_pcm = mic.readframes(2400)

            self.assertEqual(remote_pcm[: 1600 * 2], pcm(1000, 1600))
            self.assertEqual(remote_pcm[1600 * 2 :], pcm(0, 800))
            self.assertEqual(mic_pcm[: 1600 * 2], pcm(0, 1600))
            self.assertEqual(mic_pcm[1600 * 2 :], pcm(2000, 800))
            self.assertEqual(store.get("call-1")["id"], "call-1")
            self.assertEqual([item["id"] for item in store.list()], ["call-1"])

            events = [
                json.loads(line)
                for line in Path(completed["eventsPath"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["event"], "recording.started")
            self.assertEqual(events[-1]["event"], "recording.finished")
            manifest = json.loads(Path(completed["manifestPath"]).read_text(encoding="utf-8"))
            self.assertNotIn("must-not-leak", json.dumps(manifest))
            self.assertEqual(manifest["application"]["apiKey"], "<redacted>")

    def test_write_never_raises_and_failed_recording_can_be_finished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary)
            store.start("write-failure", started_monotonic_ns=0)

            self.assertFalse(store.write("unknown", b"12", captured_at_ns=0))
            self.assertFalse(store.write("remote", b"1", captured_at_ns=0))
            completed = store.finish()

            self.assertEqual(completed["status"], "failed")
            self.assertEqual(completed["durationMs"], 0)
            self.assertFalse(store.write("remote", b"12"))

    def test_safe_ids_reports_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary)
            with self.assertRaises(ValueError):
                store.start("../outside")

            store.start("safe-id", started_monotonic_ns=0)
            store.finish()
            run_directory = Path(temporary) / "safe-id" / "runs" / "run-1"
            run_directory.mkdir(parents=True)
            (run_directory / "events.jsonl").write_text("", encoding="utf-8")
            report = store.write_report(
                "safe-id",
                {"status": "passed", "token": "must-not-leak"},
                run_id="run-1",
            )

            self.assertTrue(Path(report["path"]).is_file())
            self.assertEqual(report["report"]["token"], "<redacted>")
            self.assertIsNone(store.get("../outside"))
            self.assertFalse(store.delete("../outside"))
            self.assertTrue(store.track_path("safe-id", "remote").is_file())
            self.assertTrue(store.delete("safe-id"))
            self.assertIsNone(store.get("safe-id"))

    def test_verify_detects_changed_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CallRecordingStore(temporary)
            store.start("verified", started_monotonic_ns=0)
            store.write("remote", pcm(100, 1600), captured_at_ns=100_000_000)
            store.write("mic", pcm(200, 1600), captured_at_ns=100_000_000)
            store.finish()

            self.assertTrue(store.verify("verified")["ok"])
            remote = store.track_path("verified", "remote")
            raw = bytearray(remote.read_bytes())
            raw[-1] ^= 1
            remote.write_bytes(raw)

            verification = store.verify("verified")
            self.assertFalse(verification["ok"])
            self.assertIn("контрольная сумма", " ".join(verification["errors"]))


class RecordedPcmSourceTests(unittest.TestCase):
    def test_replays_pcm_at_recorded_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "remote.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(pcm(123, 3200))

            fake_time = FakeTime()
            clock = ReplayClock(monotonic=fake_time.monotonic, sleeper=fake_time.sleep)
            source = RecordedPcmSource(path, clock, chunk_duration_ms=100)

            chunks = list(source.chunks(threading.Event()))

            self.assertEqual(chunks, [pcm(123, 1600), pcm(123, 1600)])
            self.assertAlmostEqual(fake_time.now, 0.2)
            self.assertAlmostEqual(sum(fake_time.sleeps), 0.2)

    def test_rejects_incompatible_wave(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(b"\0" * 400)

            source = RecordedPcmSource(path, ReplayClock())
            with self.assertRaises(CallRecordingError):
                list(source.chunks(threading.Event()))

    def test_two_tracks_wait_for_each_other_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "remote.wav", Path(temporary) / "mic.wav"]
            for path in paths:
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16_000)
                    writer.writeframes(pcm(123, 1600))
            clock = ReplayClock(participants=2)
            chunks: list[list[bytes]] = [[], []]

            first = threading.Thread(
                target=lambda: chunks[0].extend(
                    RecordedPcmSource(paths[0], clock, chunk_duration_ms=100).chunks(threading.Event())
                )
            )
            first.start()
            time.sleep(0.05)
            self.assertEqual(chunks[0], [])
            second = threading.Thread(
                target=lambda: chunks[1].extend(
                    RecordedPcmSource(paths[1], clock, chunk_duration_ms=100).chunks(threading.Event())
                )
            )
            second.start()
            first.join(timeout=2)
            second.join(timeout=2)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(chunks, [[pcm(123, 1600)], [pcm(123, 1600)]])


if __name__ == "__main__":
    unittest.main()
