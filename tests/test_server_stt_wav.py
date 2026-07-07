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


if __name__ == "__main__":
    unittest.main()
