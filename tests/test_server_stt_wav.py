import io
import json
import threading
import unittest
import wave
from http.client import HTTPConnection

import mimir.server as server


def sample_wav() -> bytes:
    raw = io.BytesIO()
    with wave.open(raw, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0" * 3_200)
    return raw.getvalue()


class ServerSpeechKitWavTests(unittest.TestCase):
    def test_local_preflight_rejects_missing_model(self) -> None:
        original_import_check = server.add_import_check
        original_local_vosk_status = server.local_vosk_status
        server.add_import_check = lambda checks, name, _module, _error: server.add_preflight_check(
            checks, name, True, "available"
        )
        server.local_vosk_status = lambda: {
            "model": "vosk-model-small-ru-0.22",
            "path": "F:\\missing-model",
            "installed": False,
        }
        checks: list[dict[str, object]] = []
        try:
            server.add_local_stt_checks(checks)
        finally:
            server.add_import_check = original_import_check
            server.local_vosk_status = original_local_vosk_status

        model_check = next(check for check in checks if check["name"] == "local_stt_model")
        self.assertFalse(model_check["ok"])

    def test_hotkey_toggle_uses_configured_audio_mode(self) -> None:
        original_audio_is_running = server.audio_is_running
        original_build_preflight = server.build_live_audio_preflight
        original_load_config = server.load_config
        original_start_locked = server.start_live_audio_locked
        requests: list[dict[str, object]] = []

        class FakeConfig:
            audio_mode = "speechkit"

        server.audio_is_running = lambda: False
        server.build_live_audio_preflight = lambda payload: {
            "ok": True,
            "errors": [],
            "mode": payload["mode"],
        }
        server.load_config = lambda: FakeConfig()
        server.start_live_audio_locked = lambda payload: requests.append(payload) or {
            "running": True,
            "mode": payload["mode"],
            "sources": payload["sources"],
        }
        try:
            payload = server.toggle_live_audio()
        finally:
            server.audio_is_running = original_audio_is_running
            server.build_live_audio_preflight = original_build_preflight
            server.load_config = original_load_config
            server.start_live_audio_locked = original_start_locked

        self.assertEqual(payload["mode"], "speechkit")
        self.assertEqual(requests[0]["mode"], "speechkit")

    def test_starting_audio_mode_stops_every_other_controller(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_local_audio = server.LOCAL_AUDIO
        original_read_secret = server.read_secret
        events: list[str] = []

        class FakeAudio:
            def __init__(self, name: str, running: bool = False) -> None:
                self.name = name
                self.running = running

            def start(self, *_args) -> dict[str, object]:
                events.append(f"start:{self.name}")
                self.running = True
                return {"running": True, "mode": self.name, "sources": ["remote"]}

            def stop(self) -> dict[str, object]:
                events.append(f"stop:{self.name}")
                self.running = False
                return {"running": False, "mode": self.name, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": self.running, "mode": self.name, "sources": ["remote"] if self.running else []}

        server.REALTIME_AUDIO = FakeAudio("yandex_realtime", running=True)
        server.LIVE_AUDIO = FakeAudio("speechkit")
        server.LOCAL_AUDIO = FakeAudio("local_vosk")
        server.read_secret = lambda _name: "test-key"
        try:
            payload = server.start_live_audio({"mode": "speechkit", "sources": ["remote"]})

            self.assertTrue(payload["running"])
            self.assertEqual(
                events,
                ["stop:yandex_realtime", "stop:speechkit", "stop:local_vosk", "start:speechkit"],
            )
            self.assertFalse(server.REALTIME_AUDIO.snapshot()["running"])
            self.assertTrue(server.LIVE_AUDIO.snapshot()["running"])
            self.assertFalse(server.LOCAL_AUDIO.snapshot()["running"])
        finally:
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.LOCAL_AUDIO = original_local_audio
            server.read_secret = original_read_secret

    def test_audio_device_error_returns_json_response(self) -> None:
        original_list_audio_devices = server.list_audio_devices
        server.list_audio_devices = lambda: (_ for _ in ()).throw(RuntimeError("device failure"))
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            conn = HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/api/audio/devices")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 500)
            self.assertEqual(payload["error"], "device failure")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.list_audio_devices = original_list_audio_devices

    def test_starts_wav_stt_job(self) -> None:
        original_job = server.run_wav_stt_job
        original_read_secret = server.read_secret
        called = threading.Event()
        calls: list[tuple[str, str, str, int, int, str]] = []

        def fake_job(
            job_id: str,
            source: str,
            language: str,
            chunk_duration_ms: int,
            data: bytes,
            key: str,
        ) -> None:
            calls.append((job_id, source, language, chunk_duration_ms, len(data), key))
            called.set()

        server.run_wav_stt_job = fake_job
        server.read_secret = lambda name: "test-key" if name == "yandex_speechkit" else ""
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = sample_wav()
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/stt/wav?source=remote&language=ru-RU&chunkMs=100",
                body=body,
                headers={
                    "Content-Type": "audio/wav",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(payload["started"])
            self.assertTrue(payload["jobId"].startswith("stt_wav_"))
            self.assertTrue(called.wait(2))
            self.assertEqual(calls[0][1:], ("remote", "ru-RU", 100, len(body), "test-key"))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.run_wav_stt_job = original_job
            server.read_secret = original_read_secret

    def test_starts_live_audio_session(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_read_secret = server.read_secret
        calls: list[tuple[tuple[str, ...], str, bool]] = []

        class FakeLiveAudio:
            def start(self, config, key: str) -> dict[str, object]:
                calls.append((config.sources, key, config.vad_enabled))
                return {
                    "running": True,
                    "sources": list(config.sources),
                    "language": config.language,
                    "sampleRateHertz": config.sample_rate_hertz,
                    "chunkDurationMs": config.chunk_duration_ms,
                    "vadEnabled": config.vad_enabled,
                }

            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeRealtimeAudio:
            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        server.LIVE_AUDIO = FakeLiveAudio()
        server.REALTIME_AUDIO = FakeRealtimeAudio()
        server.read_secret = lambda name: "test-key" if name == "yandex_speechkit" else ""
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps({"sources": ["remote"], "mode": "speechkit", "vadEnabled": True}).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/start",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(payload["running"])
            self.assertEqual(payload["sources"], ["remote"])
            self.assertEqual(calls, [(("remote",), "test-key", True)])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.read_secret = original_read_secret

    def test_starts_realtime_audio_session(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_read_secret = server.read_secret
        original_load_config = server.load_config
        calls: list[tuple[tuple[str, ...], str, str, bool]] = []

        class FakeLiveAudio:
            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeRealtimeAudio:
            def start(self, config, key: str, folder_id: str) -> dict[str, object]:
                calls.append((config.sources, key, folder_id, config.vad_enabled))
                return {
                    "running": True,
                    "mode": "yandex_realtime",
                    "sources": list(config.sources),
                    "language": config.language,
                    "sampleRateHertz": config.sample_rate_hertz,
                    "chunkDurationMs": config.chunk_duration_ms,
                    "vadEnabled": config.vad_enabled,
                }

            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeConfig:
            yandex_folder_id = "folder-id"

        server.LIVE_AUDIO = FakeLiveAudio()
        server.REALTIME_AUDIO = FakeRealtimeAudio()
        server.read_secret = lambda name: "test-key" if name == "yandex_ai_studio" else ""
        server.load_config = lambda: FakeConfig()
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps({"sources": ["remote", "mic"], "mode": "yandex_realtime", "vadEnabled": True}).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/start",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(payload["running"])
            self.assertEqual(payload["mode"], "yandex_realtime")
            self.assertEqual(payload["sources"], ["remote", "mic"])
            self.assertEqual(calls, [(("remote", "mic"), "test-key", "folder-id", True)])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.read_secret = original_read_secret
            server.load_config = original_load_config

    def test_starts_local_vosk_audio_session(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_local_audio = server.LOCAL_AUDIO
        calls: list[tuple[tuple[str, ...], str, bool]] = []

        class FakeAudio:
            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeLocalAudio:
            def start(self, config, key: str) -> dict[str, object]:
                calls.append((config.sources, key, config.vad_enabled))
                return {
                    "running": True,
                    "mode": "local_vosk",
                    "sources": list(config.sources),
                    "language": config.language,
                    "sampleRateHertz": config.sample_rate_hertz,
                    "chunkDurationMs": config.chunk_duration_ms,
                    "vadEnabled": config.vad_enabled,
                }

            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        server.LIVE_AUDIO = FakeAudio()
        server.REALTIME_AUDIO = FakeAudio()
        server.LOCAL_AUDIO = FakeLocalAudio()
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps({"sources": ["remote", "mic"], "mode": "local_vosk", "vadEnabled": True}).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/start",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(payload["running"])
            self.assertEqual(payload["mode"], "local_vosk")
            self.assertEqual(calls, [(("remote", "mic"), "", True)])
            self.assertEqual(server.SESSION_MANAGER.metrics()["answerProviderOverride"], "ollama")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.SESSION_MANAGER.set_answer_provider_override(None)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.LOCAL_AUDIO = original_local_audio

    def test_realtime_start_falls_back_to_local_vosk_on_provider_error(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_local_audio = server.LOCAL_AUDIO
        original_read_secret = server.read_secret
        original_load_config = server.load_config
        local_calls: list[tuple[tuple[str, ...], str]] = []

        class FakeLiveAudio:
            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FailingRealtimeAudio:
            def start(self, _config, _key: str, _folder_id: str) -> dict[str, object]:
                raise server.ProviderError("quota exceeded")

            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeLocalAudio:
            def start(self, config, key: str) -> dict[str, object]:
                local_calls.append((config.sources, key))
                return {"running": True, "mode": "local_vosk", "sources": list(config.sources)}

            def stop(self) -> dict[str, object]:
                return {"running": False, "sources": []}

            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeConfig:
            yandex_folder_id = "folder-id"

        server.LIVE_AUDIO = FakeLiveAudio()
        server.REALTIME_AUDIO = FailingRealtimeAudio()
        server.LOCAL_AUDIO = FakeLocalAudio()
        server.read_secret = lambda name: "test-key" if name == "yandex_ai_studio" else ""
        server.load_config = lambda: FakeConfig()
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps({"sources": ["remote"], "mode": "yandex_realtime"}).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/start",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(payload["mode"], "local_vosk")
            self.assertEqual(local_calls, [(("remote",), "")])
            self.assertEqual(server.SESSION_MANAGER.metrics()["answerProviderOverride"], "ollama")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.SESSION_MANAGER.set_answer_provider_override(None)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.LOCAL_AUDIO = original_local_audio
            server.read_secret = original_read_secret
            server.load_config = original_load_config

    def test_realtime_audio_preflight_rejects_missing_folder(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_read_secret = server.read_secret
        original_load_config = server.load_config
        original_list_audio_devices = server.list_audio_devices

        class FakeAudio:
            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeConfig:
            yandex_folder_id = ""

        server.LIVE_AUDIO = FakeAudio()
        server.REALTIME_AUDIO = FakeAudio()
        server.read_secret = lambda name: "test-key" if name == "yandex_ai_studio" else ""
        server.load_config = lambda: FakeConfig()
        server.list_audio_devices = lambda: [
            {"id": "remote1", "source": "remote"},
            {"id": "mic1", "source": "mic"},
        ]
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps({"sources": ["remote", "mic"], "mode": "yandex_realtime"}).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/preflight",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertFalse(payload["ok"])
            self.assertIn("Yandex folder ID is missing", payload["errors"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.read_secret = original_read_secret
            server.load_config = original_load_config
            server.list_audio_devices = original_list_audio_devices

    def test_realtime_audio_preflight_passes_ready_setup(self) -> None:
        original_live_audio = server.LIVE_AUDIO
        original_realtime_audio = server.REALTIME_AUDIO
        original_read_secret = server.read_secret
        original_load_config = server.load_config
        original_list_audio_devices = server.list_audio_devices

        class FakeAudio:
            def snapshot(self) -> dict[str, object]:
                return {"running": False, "sources": []}

        class FakeConfig:
            yandex_folder_id = "folder-id"

        server.LIVE_AUDIO = FakeAudio()
        server.REALTIME_AUDIO = FakeAudio()
        server.read_secret = lambda name: "test-key" if name in {"yandex_ai_studio", "yandex_speechkit"} else ""
        server.load_config = lambda: FakeConfig()
        server.list_audio_devices = lambda: [
            {"id": "remote1", "source": "remote"},
            {"id": "mic1", "source": "mic"},
        ]
        httpd = server.create_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = httpd.server_address
            body = json.dumps(
                {
                    "sources": ["remote", "mic"],
                    "mode": "yandex_realtime",
                    "deviceIds": {"remote": "remote1", "mic": "mic1"},
                }
            ).encode("utf-8")
            conn = HTTPConnection(host, port, timeout=5)
            conn.request(
                "POST",
                "/api/session/audio/preflight",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["errors"], [])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            server.LIVE_AUDIO = original_live_audio
            server.REALTIME_AUDIO = original_realtime_audio
            server.read_secret = original_read_secret
            server.load_config = original_load_config
            server.list_audio_devices = original_list_audio_devices


if __name__ == "__main__":
    unittest.main()
