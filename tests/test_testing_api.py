import json
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mimir.server as server


class FakeRecordingStore:
    def __init__(self, recordings=None, active_id=None) -> None:
        self.recordings = list(recordings or [])
        self.active_id = active_id
        self.deleted: list[str] = []

    def active_recording_id(self):
        return self.active_id

    def list(self):
        return list(self.recordings)

    def delete(self, recording_id: str) -> bool:
        for index, recording in enumerate(self.recordings):
            if recording.get("id") == recording_id:
                del self.recordings[index]
                self.deleted.append(recording_id)
                return True
        return False


class FakeReplayController:
    def __init__(self, state="idle") -> None:
        self.state = state
        self.recording_id = None
        self.start_calls: list[str] = []
        self.stop_calls = 0
        self.wait_calls: list[float | None] = []

    def snapshot(self):
        return {
            "state": self.state,
            "recordingId": self.recording_id,
            "elapsedMs": 0,
            "durationMs": 0,
            "error": "",
        }

    def is_running(self) -> bool:
        return self.state == "running"

    def start(self, recording_id: str):
        self.start_calls.append(recording_id)
        self.recording_id = recording_id
        self.state = "running"
        return self.snapshot()

    def stop(self):
        self.stop_calls += 1
        self.state = "stopped"
        return self.snapshot()

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.snapshot()


class FakeAudioController:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return {"running": False, "sources": []}

    def snapshot(self):
        return {"running": False, "sources": []}


class TestingApiTests(unittest.TestCase):
    def test_testing_snapshot_converts_recording_statuses(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote.wav"
            mic = root / "mic.wav"
            remote.write_bytes(b"remote")
            mic.write_bytes(b"mic")
            created_at_ms = int(time.time() * 1_000) - 2_000
            store = FakeRecordingStore(
                [
                    recording_payload("ready", "complete", remote, mic),
                    recording_payload("incomplete", "complete", remote, root / "missing.wav"),
                    {
                        **recording_payload("failed", "failed", remote, mic),
                        "errors": ["Ошибка дорожки", "Недостаточно места"],
                    },
                    {
                        **recording_payload("active", "recording", remote, mic),
                        "createdAtMs": created_at_ms,
                        "durationMs": 100,
                    },
                ],
                active_id="active",
            )

            with (
                patch.object(server, "CALL_RECORDINGS", store),
                patch.object(server, "TESTING_REPLAY", FakeReplayController()),
            ):
                snapshot = server.testing_snapshot()

        by_id = {item["id"]: item for item in snapshot["recordings"]}
        self.assertEqual(snapshot["activeRecordingId"], "active")
        self.assertEqual(by_id["ready"]["status"], "ready")
        self.assertEqual(by_id["ready"]["tracks"], {"remote": True, "mic": True})
        self.assertEqual(by_id["incomplete"]["status"], "incomplete")
        self.assertEqual(by_id["failed"]["status"], "failed")
        self.assertEqual(by_id["failed"]["error"], "Ошибка дорожки; Недостаточно места")
        self.assertEqual(by_id["active"]["status"], "recording")
        self.assertGreaterEqual(by_id["active"]["durationMs"], 1_900)

    def test_starts_and_stops_replay_without_real_providers(self) -> None:
        store = FakeRecordingStore([{"id": "call-1"}])
        replay = FakeReplayController()
        audio = [FakeAudioController() for _ in range(3)]

        with patched_testing_globals(store, replay, audio):
            started = server.start_testing_replay("call-1")
            stopped = server.stop_testing_replay()

        self.assertEqual(started["state"], "running")
        self.assertEqual(replay.start_calls, ["call-1"])
        self.assertTrue(all(item.stop_calls == 1 for item in audio))
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(replay.stop_calls, 1)
        self.assertEqual(replay.wait_calls, [5])

    def test_deletes_finished_recording_and_returns_new_snapshot(self) -> None:
        store = FakeRecordingStore([{"id": "call-1", "status": "complete"}])
        replay = FakeReplayController()

        with (
            patch.object(server, "CALL_RECORDINGS", store),
            patch.object(server, "TESTING_REPLAY", replay),
        ):
            snapshot = server.delete_testing_recording("call-1")

        self.assertEqual(store.deleted, ["call-1"])
        self.assertEqual(snapshot["recordings"], [])

    def test_testing_http_routes_use_server_commands(self) -> None:
        store = FakeRecordingStore([{"id": "call-1", "status": "complete"}])
        replay = FakeReplayController()
        audio = [FakeAudioController() for _ in range(3)]

        with patched_testing_globals(store, replay, audio):
            httpd = server.create_server(port=0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = httpd.server_address
                status, snapshot = get_json(host, port, "/api/testing")
                start_status, started = post_json(
                    host,
                    port,
                    "/api/testing/replay/start",
                    {"recordingId": "call-1"},
                )
                stop_status, stopped = post_json(host, port, "/api/testing/replay/stop", {})
                delete_status, deleted = post_json(
                    host,
                    port,
                    "/api/testing/recordings/delete",
                    {"recordingId": "call-1"},
                )
                invalid_status, invalid = post_json(
                    host,
                    port,
                    "/api/testing/replay/start",
                    {},
                )
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

        self.assertEqual(status, 200)
        self.assertEqual(snapshot["recordings"][0]["id"], "call-1")
        self.assertEqual(start_status, 200)
        self.assertEqual(started["state"], "running")
        self.assertEqual(stop_status, 200)
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(delete_status, 200)
        self.assertEqual(deleted["recordings"], [])
        self.assertEqual(invalid_status, 400)
        self.assertEqual(invalid["error"], "Не указана запись для проверки")


def recording_payload(recording_id: str, status: str, remote: Path, mic: Path):
    return {
        "id": recording_id,
        "status": status,
        "createdAtMs": 1_700_000_000_000,
        "durationMs": 1_500,
        "sizeBytes": 42,
        "tracks": {
            "remote": {"receivedAudio": True, "frames": 16_000},
            "mic": {"receivedAudio": True, "frames": 16_000},
        },
        "paths": {"remote": str(remote), "mic": str(mic)},
        "errors": [],
    }


def patched_testing_globals(store, replay, audio):
    return patch.multiple(
        server,
        CALL_RECORDINGS=store,
        TESTING_REPLAY=replay,
        LIVE_AUDIO=audio[0],
        LOCAL_AUDIO=audio[1],
        REALTIME_AUDIO=audio[2],
    )


def get_json(host: str, port: int, path: str):
    connection = HTTPConnection(host, port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def post_json(host: str, port: int, path: str, payload: dict[str, object]):
    body = json.dumps(payload).encode("utf-8")
    connection = HTTPConnection(host, port, timeout=5)
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, data


if __name__ == "__main__":
    unittest.main()
