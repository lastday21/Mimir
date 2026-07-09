import json
import threading
import unittest
from http.client import HTTPConnection

import mimir.server as server
from mimir.session import SessionManager


class SessionControlTests(unittest.TestCase):
    def test_pause_closes_context_and_next_start_is_clean(self) -> None:
        manager = SessionManager()
        first = manager.start()
        manager.ingest_transcript("remote", "Мы обсуждаем очереди задач", detect_question=False)

        paused = manager.pause()
        second = manager.start()

        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["memory"]["turns"], [])
        self.assertEqual(paused["memory"]["questions"], [])
        self.assertEqual(second["state"], "listening")
        self.assertNotEqual(second["sessionId"], first["sessionId"])
        self.assertEqual(second["memory"]["turns"], [])
        self.assertEqual(second["memory"]["questions"], [])

    def test_paused_session_ignores_late_transcripts(self) -> None:
        manager = SessionManager()
        manager.start()
        manager.pause()

        payload = manager.ingest_transcript("remote", "Поздний кусок старого разговора", detect_question=False)

        self.assertTrue(payload["skipped"])
        self.assertEqual(payload["reason"], "paused")
        self.assertEqual(manager.snapshot()["memory"]["turns"], [])


class ServerSessionControlTests(unittest.TestCase):
    def test_pause_route_stops_audio_and_closes_context(self) -> None:
        original_session = server.SESSION_MANAGER
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        manager = SessionManager()
        live_audio = FakeAudioControls()
        realtime_audio = FakeAudioControls()
        server.SESSION_MANAGER = manager
        server.LIVE_AUDIO = live_audio
        server.REALTIME_AUDIO = realtime_audio
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            first_status, first = post_json(host, port, "/api/session/start", {})
            manager.ingest_transcript("remote", "Старый рабочий контекст", detect_question=False)
            pause_status, paused = post_json(host, port, "/api/session/pause", {})
            second_status, second = post_json(host, port, "/api/session/start", {})

            self.assertEqual(first_status, 200)
            self.assertEqual(pause_status, 200)
            self.assertEqual(second_status, 200)
            self.assertEqual(paused["state"], "paused")
            self.assertEqual(paused["memory"]["turns"], [])
            self.assertNotEqual(second["sessionId"], first["sessionId"])
            self.assertEqual(second["memory"]["turns"], [])
            self.assertEqual(live_audio.stop_calls, 1)
            self.assertEqual(realtime_audio.stop_calls, 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.SESSION_MANAGER = original_session
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio

    def test_resume_and_manual_question_are_not_api_contract(self) -> None:
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            resume_status = post_status(host, port, "/api/session/resume", {})
            manual_status = post_status(host, port, "/api/manual/question", {"question": "test"})

            self.assertEqual(resume_status, 404)
            self.assertEqual(manual_status, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


class FakeAudioControls:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> dict[str, object]:
        self.stop_calls += 1
        return {"running": False, "sources": []}

    def snapshot(self) -> dict[str, object]:
        return {"running": False, "sources": []}


def post_json(host: str, port: int, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    status, data = post_raw(host, port, path, payload)
    return status, json.loads(data.decode("utf-8"))


def post_status(host: str, port: int, path: str, payload: dict[str, object]) -> int:
    status, _data = post_raw(host, port, path, payload)
    return status


def post_raw(host: str, port: int, path: str, payload: dict[str, object]) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    connection = HTTPConnection(host, port, timeout=5)
    connection.request(
        "POST",
        path,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


if __name__ == "__main__":
    unittest.main()
