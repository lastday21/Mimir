from __future__ import annotations

import unittest

from mimir.audio.capture import AudioCaptureError
from mimir.audio.preflight import PreflightDependencies, build_live_audio_preflight
from mimir.preflight_probes import probe_audio_source
from mimir.providers.base import ProviderError


class FakeTesting:
    enabled = False


class FakeApplication:
    process_id = 42


class FakeConfig:
    audio_mode = "speechkit"
    yandex_folder_id = "folder-id"
    llm_model = "yandexgpt/latest"
    testing = FakeTesting()
    audio_application = FakeApplication()
    ollama_base_url = "http://127.0.0.1:11434"


class FakeOllama:
    def __init__(self, _base_url: str) -> None:
        pass

    def list_models(self) -> list[object]:
        return []


def dependencies(**overrides: object) -> PreflightDependencies:
    values: dict[str, object] = {
        "load_config": lambda: FakeConfig(),
        "read_secret": lambda _name: "test-key",
        "list_audio_devices": lambda: [{"id": "mic-1", "source": "mic"}],
        "local_vosk_status": lambda: {"installed": True, "model": "vosk", "path": "model"},
        "ollama_client": FakeOllama,
        "select_preferred_model": lambda models: models[0] if models else None,
        "import_module": lambda _name: object(),
        "audio_is_running": lambda: False,
        "list_audio_applications": lambda: [{"processId": 42, "title": "Созвон"}],
        "speechkit_probe": lambda _key: "SpeechKit доступен",
        "yandex_model_probe": lambda *_args: "Модель доступна",
        "audio_source_probe": lambda source, *_args: f"Захват {source} доступен",
    }
    values.update(overrides)
    return PreflightDependencies(**values)  # type: ignore[arg-type]


class PreflightTests(unittest.TestCase):
    def test_ready_cloud_setup_runs_every_runtime_probe(self) -> None:
        calls: list[str] = []
        result = build_live_audio_preflight(
            {
                "mode": "speechkit",
                "sources": ["remote", "mic"],
                "deviceIds": {"mic": "mic-1"},
                "applicationProcessId": 42,
            },
            dependencies(
                speechkit_probe=lambda _key: calls.append("speechkit") or "SpeechKit доступен",
                yandex_model_probe=lambda *_args: calls.append("model") or "Модель доступна",
                audio_source_probe=lambda source, *_args: calls.append(source) or f"Захват {source} доступен",
            ),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(calls, ["model", "speechkit", "remote", "mic"])
        self.assertEqual(result["errors"], [])

    def test_returns_every_runtime_failure(self) -> None:
        def fail_model(*_args: object) -> str:
            raise ProviderError("Модель недоступна")

        def fail_speechkit(_key: str) -> str:
            raise ProviderError("SpeechKit недоступен")

        def fail_audio(source: str, *_args: object) -> str:
            raise AudioCaptureError(f"Не открыт источник: {source}")

        result = build_live_audio_preflight(
            {
                "mode": "speechkit",
                "sources": ["remote", "mic"],
                "deviceIds": {"mic": "mic-1"},
                "applicationProcessId": 42,
            },
            dependencies(
                speechkit_probe=fail_speechkit,
                yandex_model_probe=fail_model,
                audio_source_probe=fail_audio,
            ),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["errors"],
            [
                "Модель недоступна",
                "SpeechKit недоступен",
                "Не открыт источник: remote",
                "Не открыт источник: mic",
            ],
        )

    def test_does_not_probe_sources_while_audio_is_running(self) -> None:
        calls: list[str] = []
        result = build_live_audio_preflight(
            {"mode": "speechkit", "sources": ["mic"], "deviceIds": {"mic": "mic-1"}},
            dependencies(
                audio_is_running=lambda: True,
                audio_source_probe=lambda source, *_args: calls.append(source) or "готово",
            ),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(calls, [])

    def test_audio_probe_closes_source_after_first_chunk(self) -> None:
        stopped: list[bool] = []

        class FakeSource:
            def chunks(self, stop_event):
                try:
                    yield b"\0\0"
                finally:
                    stopped.append(stop_event.is_set())

        detail = probe_audio_source(
            "mic",
            "mic-1",
            0,
            source_factory=lambda *_args: FakeSource(),
        )

        self.assertEqual(detail, "Захват звука открывается")
        self.assertEqual(stopped, [True])


if __name__ == "__main__":
    unittest.main()
