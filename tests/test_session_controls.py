import json
import threading
import time
import unittest
from http.client import HTTPConnection

import mimir.server as server
from mimir.session import SessionManager


class SessionControlTests(unittest.TestCase):
    def test_cross_source_duplicate_prefers_remote_turn(self) -> None:
        manager = SessionManager()
        manager.start()

        manager.ingest_transcript("mic", "Добрый день", detect_question=False)
        manager.ingest_transcript("remote", "Добрый день", detect_question=False)

        turns = manager.snapshot()["memory"]["turns"]
        self.assertEqual([(turn["source"], turn["text"]) for turn in turns], [("remote", "Добрый день")])

    def test_remote_partial_prevents_echo_from_becoming_user_answer(self) -> None:
        manager = SessionManager()
        manager.start()

        manager.ingest_transcript(
            "remote",
            "Сколько будет пять плюс пять",
            is_final=False,
            detect_question=False,
        )
        result = manager.ingest_transcript(
            "mic",
            "Сколько будет 5 + 5",
            detect_question=False,
        )
        manager.ingest_transcript(
            "remote",
            "Сколько будет 5 + 5",
            detect_question=False,
        )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "cross_source_duplicate")
        turns = manager.snapshot()["memory"]["turns"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["source"], "remote")

    def test_cross_source_filter_keeps_different_numbers(self) -> None:
        manager = SessionManager()
        manager.start()

        manager.ingest_transcript("remote", "Итоговая реплика 1", detect_question=False)
        manager.ingest_transcript("mic", "Итоговая реплика 2", detect_question=False)

        self.assertEqual(len(manager.snapshot()["memory"]["turns"]), 2)

    def test_event_sink_receives_session_events_and_can_be_removed(self) -> None:
        captured: list[tuple[str, dict[str, object]]] = []

        def sink(event: str, payload: dict[str, object]) -> None:
            captured.append((event, payload))

        manager = SessionManager(sink)
        manager.start()
        manager.ingest_transcript("remote", "Проверка записи", detect_question=False)
        manager.remove_event_sink(sink)
        manager.ingest_transcript("mic", "После отключения", detect_question=False)

        names = [event for event, _payload in captured]
        self.assertIn("session_state", names)
        self.assertEqual(names.count("transcript"), 1)

    def test_final_transcript_replaces_interim_in_event_and_snapshot(self) -> None:
        manager = SessionManager()
        manager.start()

        interim = manager.ingest_transcript("remote", "Как вы", is_final=False, detect_question=False)
        final = manager.ingest_transcript("remote", "Как вы строили сервис?", detect_question=False)
        snapshot = manager.snapshot()

        self.assertEqual(interim["operation"], "append")
        self.assertEqual(final["operation"], "replace")
        self.assertEqual(final["turnId"], interim["turnId"])
        self.assertEqual(len(snapshot["memory"]["turns"]), 1)
        self.assertEqual(snapshot["memory"]["turns"][0]["text"], "Как вы строили сервис?")
        self.assertTrue(snapshot["memory"]["turns"][0]["isFinal"])

    def test_session_links_question_hint_and_final_user_answer(self) -> None:
        manager = SessionManager()
        manager.start()
        manager.record_external_question("question_link", "Как вы обеспечивали надежность?")
        manager.record_answer_delta("question_link", "Упомяните резервирование и наблюдаемость.")
        manager.ingest_transcript("mic", "Я обеспечивал", is_final=False, detect_question=False)
        manager.ingest_transcript(
            "mic",
            "Я обеспечивал резервирование и добавил метрики",
            detect_question=False,
        )

        exchange = manager.snapshot()["memory"]["exchanges"][-1]

        self.assertEqual(exchange["questionId"], "question_link")
        self.assertEqual(exchange["hint"], "Упомяните резервирование и наблюдаемость.")
        self.assertEqual(exchange["userAnswer"], "Я обеспечивал резервирование и добавил метрики")

    def test_late_answer_delta_does_not_change_current_question(self) -> None:
        manager = SessionManager()
        manager.start()
        manager.record_external_question("question_old", "Старый вопрос")
        manager.record_answer_delta("question_old", "Старый ответ")
        manager.record_external_question("question_new", "Новый вопрос")

        manager.record_answer_delta("question_old", " запоздал")
        manager.record_answer_delta("question_new", "Новый ответ")
        snapshot = manager.snapshot()

        self.assertEqual(snapshot["currentQuestion"]["questionId"], "question_new")
        self.assertEqual(snapshot["currentAnswer"]["questionId"], "question_new")
        self.assertEqual(snapshot["currentAnswer"]["text"], "Новый ответ")

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

    def test_summary_is_built_in_background_after_six_final_turns(self) -> None:
        class SummarySessionManager(SessionManager):
            def _stream_answer(self, _messages):
                yield "Команда обсуждает выпуск. Пользователь должен проверить сборку сегодня."

        manager = SummarySessionManager()
        manager.start()
        for index in range(6):
            source = "remote" if index % 2 == 0 else "mic"
            manager.ingest_transcript(source, f"Итоговая реплика {index}", detect_question=False)

        deadline = time.monotonic() + 2
        summary = ""
        while time.monotonic() < deadline:
            summary = str(manager.snapshot()["memory"]["summary"])
            if summary:
                break
            time.sleep(0.01)

        self.assertIn("Пользователь должен проверить сборку", summary)
        manager.ingest_transcript("remote", "Какой текущий срок?", detect_question=False)
        context = manager.realtime_context(max_turns=12, max_chars=800)
        self.assertIn("Сжатая сводка разговора", context)
        self.assertIn("Собеседник: Какой текущий срок?", context)


class ServerSessionControlTests(unittest.TestCase):
    def test_saving_config_refreshes_active_realtime_settings(self) -> None:
        original_realtime_audio = server.REALTIME_AUDIO
        original_save_config = server.save_config

        class RefreshingAudio:
            def __init__(self) -> None:
                self.refresh_calls = 0

            def refresh_settings(self) -> None:
                self.refresh_calls += 1

        realtime_audio = RefreshingAudio()
        server.REALTIME_AUDIO = realtime_audio
        server.save_config = lambda _config: None
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            status, _payload = post_json(
                host,
                port,
                "/api/config",
                {
                    "conversation": {"mode": "meeting", "goal": "Понять новую задачу"},
                    "hotkeys": {"overlayToggle": "Ctrl+M", "audioToggle": "Ctrl+Space"},
                },
            )

            self.assertEqual(status, 200)
            self.assertEqual(realtime_audio.refresh_calls, 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.REALTIME_AUDIO = original_realtime_audio
            server.save_config = original_save_config

    def test_new_event_stream_starts_with_current_snapshot(self) -> None:
        original_session = server.SESSION_MANAGER
        manager = SessionManager()
        manager.start()
        manager.ingest_transcript("remote", "Старое событие", detect_question=False)
        server.SESSION_MANAGER = manager
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            connection = HTTPConnection(host, port, timeout=5)
            connection.request("GET", "/api/session/events")
            response = connection.getresponse()
            lines = [response.fp.readline().decode("utf-8").strip() for _ in range(4)]
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(lines[1], "event: session_snapshot")
            snapshot = json.loads(lines[2].removeprefix("data: "))
            self.assertEqual(snapshot["memory"]["turns"][-1]["text"], "Старое событие")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.SESSION_MANAGER = original_session

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
