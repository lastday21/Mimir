from __future__ import annotations

import json
import mimetypes
import threading
from importlib import import_module
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .audio import (
    AudioCaptureError,
    LiveAudioConfig,
    LiveAudioController,
    RealtimeAudioConfig,
    RealtimeAudioController,
    list_audio_devices,
)
from .config import AppConfig, load_config, save_config
from .credentials import read_secret, write_secret
from .models import ModelInfo
from .ollama_fallback import select_preferred_model, sort_models
from .providers import OllamaClient, YandexAIStudioClient, YandexSpeechKitClient
from .providers.base import ProviderError
from .session import SessionManager, sse_payload
from .stt import AudioStreamConfig, SpeechKitStreamRunner, pcm_chunks_from_wav


HOST = "127.0.0.1"
PORT = 8765
STATIC_ROOT = Path(__file__).resolve().parents[1] / "dist"
ALLOWED_CORS_HEADERS = "Content-Type"
SESSION_MANAGER = SessionManager()
LIVE_AUDIO = LiveAudioController(SESSION_MANAGER, YandexSpeechKitClient)
REALTIME_AUDIO = RealtimeAudioController(SESSION_MANAGER, YandexSpeechKitClient)
MAX_DEV_WAV_BYTES = 25_000_000


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "MimirPython/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        if parsed.path == "/api/config":
            self.send_json(config_payload(load_config()))
            return
        if parsed.path == "/api/models":
            self.handle_models()
            return
        if parsed.path == "/api/audio/devices":
            self.handle_audio_devices()
            return
        if parsed.path == "/api/metrics/current":
            self.send_json(SESSION_MANAGER.metrics())
            return
        if parsed.path == "/api/session/events":
            self.handle_session_events(parsed.query)
            return
        self.handle_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                self.handle_save_config()
            elif parsed.path == "/api/credentials/yandex":
                self.handle_yandex_key()
            elif parsed.path == "/api/session/start":
                self.send_json(SESSION_MANAGER.start())
            elif parsed.path == "/api/session/stop":
                self.handle_session_stop()
            elif parsed.path == "/api/session/pause":
                self.handle_session_pause()
            elif parsed.path == "/api/session/audio/preflight":
                self.handle_live_audio_preflight()
            elif parsed.path == "/api/session/audio/start":
                self.handle_live_audio_start()
            elif parsed.path == "/api/session/audio/stop":
                self.handle_live_audio_stop()
            elif parsed.path == "/api/session/transcript":
                self.handle_session_transcript()
            elif parsed.path == "/api/session/stt/wav":
                self.handle_session_stt_wav(parsed.query)
            else:
                self.send_error(404)
        except ProviderError as error:
            self.send_json({"error": str(error)}, status=502)
        except AudioCaptureError as error:
            self.send_json({"error": str(error)}, status=502)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

    def handle_save_config(self) -> None:
        payload = self.read_json()
        config = AppConfig.from_dict(payload)
        save_config(config)
        self.send_json(config_payload(config))

    def handle_yandex_key(self) -> None:
        payload = self.read_json()
        key = str(payload.get("apiKey") or "").strip()
        if not key:
            raise ValueError("apiKey is required")
        write_secret("yandex_ai_studio", key)
        write_secret("yandex_speechkit", key)
        self.send_json({"stored": True})

    def handle_models(self) -> None:
        config = load_config()
        if config.llm_provider == "ollama":
            models = sort_models(OllamaClient(config.ollama_base_url).list_models())
        else:
            key = read_secret("yandex_ai_studio") or ""
            models = YandexAIStudioClient(key, config.yandex_folder_id).list_models()
        preferred = select_preferred_model(models) if config.llm_provider == "ollama" else None
        self.send_json(
            {
                "models": [model_payload(model) for model in models],
                "preferredModel": preferred.id if preferred else None,
            }
        )

    def handle_audio_devices(self) -> None:
        try:
            devices = list_audio_devices()
            self.send_json({"available": True, "devices": devices})
        except AudioCaptureError as error:
            self.send_json({"available": False, "devices": [], "error": str(error)})

    def handle_session_stop(self) -> None:
        LIVE_AUDIO.stop()
        REALTIME_AUDIO.stop()
        self.send_json(SESSION_MANAGER.stop())

    def handle_session_pause(self) -> None:
        LIVE_AUDIO.stop()
        REALTIME_AUDIO.stop()
        self.send_json(SESSION_MANAGER.pause())

    def handle_live_audio_preflight(self) -> None:
        self.send_json(build_live_audio_preflight(self.read_json()))

    def handle_live_audio_start(self) -> None:
        payload = self.read_json()
        mode, common_config = parse_live_audio_request(payload)
        if mode == "speechkit":
            REALTIME_AUDIO.stop()
            key = read_secret("yandex_speechkit") or read_secret("yandex_ai_studio") or ""
            self.send_json(LIVE_AUDIO.start(LiveAudioConfig(**common_config), key))
            return
        if mode != "yandex_realtime":
            raise ValueError("audio mode must be yandex_realtime or speechkit")

        LIVE_AUDIO.stop()
        config = load_config()
        key = read_secret("yandex_ai_studio") or read_secret("yandex_speechkit") or ""
        self.send_json(REALTIME_AUDIO.start(RealtimeAudioConfig(**common_config), key, config.yandex_folder_id))

    def handle_live_audio_stop(self) -> None:
        realtime_was_running = bool(REALTIME_AUDIO.snapshot().get("running"))
        speechkit_was_running = bool(LIVE_AUDIO.snapshot().get("running"))
        if realtime_was_running:
            self.send_json(REALTIME_AUDIO.stop())
            return
        if speechkit_was_running:
            self.send_json(LIVE_AUDIO.stop())
            return
        REALTIME_AUDIO.stop()
        LIVE_AUDIO.stop()
        self.send_json(
            {
                "running": False,
                "mode": "idle",
                "sources": [],
                "language": "ru-RU",
                "sampleRateHertz": 16_000,
                "chunkDurationMs": 200,
                "vadEnabled": True,
                "deviceIds": {},
            }
        )

    def handle_session_transcript(self) -> None:
        payload = self.read_json()
        source = str(payload.get("source") or "")
        text = str(payload.get("text") or "")
        is_final = bool(payload.get("isFinal", True))
        self.send_json(SESSION_MANAGER.ingest_transcript(source, text, is_final=is_final))

    def handle_session_stt_wav(self, query: str) -> None:
        params = parse_qs(query)
        source = str(params.get("source", ["remote"])[0])
        language = str(params.get("language", ["ru-RU"])[0])
        chunk_duration_ms = int(str(params.get("chunkMs", ["400"])[0]))
        data = self.read_body(MAX_DEV_WAV_BYTES)
        key = read_secret("yandex_speechkit") or read_secret("yandex_ai_studio") or ""
        job_id = f"stt_wav_{id(data):x}"

        thread = threading.Thread(
            target=run_wav_stt_job,
            args=(job_id, source, language, chunk_duration_ms, data, key),
            name=job_id,
            daemon=True,
        )
        thread.start()
        self.send_json({"started": True, "jobId": job_id})

    def handle_session_events(self, query: str) -> None:
        params = parse_qs(query)
        after = 0
        if params.get("after"):
            try:
                after = int(params["after"][0])
            except ValueError:
                after = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", ALLOWED_CORS_HEADERS)
        self.end_headers()
        try:
            for event in SESSION_MANAGER.listen(after=after):
                self.wfile.write(sse_payload(event))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        return data

    def read_body(self, max_bytes: int) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request body is empty")
        if length > max_bytes:
            raise ValueError(f"Request body is too large: {length} bytes")
        return self.rfile.read(length)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", ALLOWED_CORS_HEADERS)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def handle_static(self, request_path: str) -> None:
        if not STATIC_ROOT.exists():
            self.send_error(404, "Frontend build not found. Run npm run build.")
            return

        relative = request_path.lstrip("/") or "index.html"
        target = (STATIC_ROOT / relative).resolve()
        static_root = STATIC_ROOT.resolve()
        if static_root not in target.parents and target != static_root:
            self.send_error(403)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            target = static_root / "index.html"

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", ALLOWED_CORS_HEADERS)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "yandexFolderId": config.yandex_folder_id,
        "llmProvider": config.llm_provider,
        "llmModel": config.llm_model,
        "ollamaBaseUrl": config.ollama_base_url,
        "hasYandexKey": bool(read_secret("yandex_ai_studio")),
    }


def model_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "contextWindow": model.context_window,
    }


def parse_live_audio_request(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sources = payload.get("sources") or ["remote", "mic"]
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    device_ids = payload.get("deviceIds") or {}
    if not isinstance(device_ids, dict):
        raise ValueError("deviceIds must be an object")
    mode = str(payload.get("mode") or "yandex_realtime").strip().lower()
    if mode not in {"yandex_realtime", "speechkit"}:
        raise ValueError("audio mode must be yandex_realtime or speechkit")
    return (
        mode,
        {
            "sources": tuple(normalize_live_audio_source(str(source)) for source in sources),
            "language": str(payload.get("language") or "ru-RU"),
            "sample_rate_hertz": int(payload.get("sampleRateHertz") or 16_000),
            "chunk_duration_ms": int(payload.get("chunkDurationMs") or 200),
            "vad_enabled": bool(payload.get("vadEnabled", True)),
            "device_ids": {str(key): str(value) for key, value in device_ids.items()},
        },
    )


def build_live_audio_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    mode, common_config = parse_live_audio_request(payload)
    sources = list(common_config["sources"])
    device_ids = dict(common_config["device_ids"])
    checks: list[dict[str, Any]] = []

    add_preflight_check(checks, "audio_idle", not audio_is_running(), "Live audio is already running")
    if mode == "yandex_realtime":
        config = load_config()
        add_preflight_check(checks, "yandex_folder_id", bool(config.yandex_folder_id.strip()), "Yandex folder ID is missing")
        add_preflight_check(
            checks,
            "yandex_ai_studio_key",
            bool(read_secret("yandex_ai_studio") or read_secret("yandex_speechkit")),
            "Yandex AI Studio API key is missing",
        )
        add_import_check(checks, "aiohttp", "aiohttp", "Realtime websocket dependency is missing")
        if "mic" in sources:
            add_speechkit_checks(checks)
    elif mode == "speechkit":
        add_speechkit_checks(checks)

    add_device_checks(checks, sources, device_ids)
    errors = [str(check["detail"]) for check in checks if not check["ok"]]
    return {
        "ok": not errors,
        "mode": mode,
        "sources": sources,
        "deviceIds": device_ids,
        "checks": checks,
        "errors": errors,
    }


def add_speechkit_checks(checks: list[dict[str, Any]]) -> None:
    add_preflight_check(
        checks,
        "yandex_speechkit_key",
        bool(read_secret("yandex_speechkit") or read_secret("yandex_ai_studio")),
        "Yandex SpeechKit API key is missing",
    )
    add_import_check(checks, "grpc", "grpc", "SpeechKit gRPC dependency is missing")
    add_import_check(checks, "yandexcloud_stt", "yandex.cloud.ai.stt.v3.stt_service_pb2_grpc", "SpeechKit stubs are missing")


def normalize_live_audio_source(source: str) -> str:
    value = source.strip().lower()
    if value in {"remote", "them", "system", "loopback"}:
        return "remote"
    if value in {"mic", "me", "user"}:
        return "mic"
    raise ValueError("audio source must be remote or mic")


def add_device_checks(checks: list[dict[str, Any]], sources: list[str], device_ids: dict[str, str]) -> None:
    try:
        devices = list_audio_devices()
    except AudioCaptureError as error:
        add_preflight_check(checks, "audio_devices", False, str(error))
        return

    add_preflight_check(checks, "audio_devices", True, f"{len(devices)} capture devices available")
    for source in sources:
        source_devices = [device for device in devices if device.get("source") == source]
        device_id = device_ids.get(source)
        if device_id:
            ok = any(str(device.get("id")) == device_id for device in source_devices)
            add_preflight_check(checks, f"{source}_device", ok, f"{source} device is not available: {device_id}")
        else:
            add_preflight_check(checks, f"{source}_device", bool(source_devices), f"No {source} capture device is available")


def add_import_check(checks: list[dict[str, Any]], name: str, module: str, error: str) -> None:
    try:
        import_module(module)
    except ImportError:
        add_preflight_check(checks, name, False, error)
        return
    add_preflight_check(checks, name, True, "available")


def add_preflight_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail})


def audio_is_running() -> bool:
    return bool(REALTIME_AUDIO.snapshot().get("running") or LIVE_AUDIO.snapshot().get("running"))


def run_wav_stt_job(job_id: str, source: str, language: str, chunk_duration_ms: int, data: bytes, key: str) -> None:
    try:
        SESSION_MANAGER.start()
        sample_rate, chunks = pcm_chunks_from_wav(data, chunk_duration_ms=chunk_duration_ms)
        config = AudioStreamConfig(
            language=language,
            sample_rate_hertz=sample_rate,
            chunk_duration_ms=chunk_duration_ms,
        )
        SESSION_MANAGER.publish_status(
            "stt_status",
            {
                "jobId": job_id,
                "source": source,
                "status": "streaming",
                "sampleRateHertz": sample_rate,
            },
        )
        runner = SpeechKitStreamRunner(YandexSpeechKitClient(key), config)
        for event in runner.run(source, chunks):
            SESSION_MANAGER.ingest_transcript(event.source, event.text, is_final=event.is_final)
        SESSION_MANAGER.publish_status(
            "stt_status",
            {"jobId": job_id, "source": source, "status": "done"},
        )
    except Exception as error:
        SESSION_MANAGER.publish_status(
            "stt_error",
            {"jobId": job_id, "source": source, "error": str(error)},
        )


def create_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ApiHandler)


def serve(host: str = HOST, port: int = PORT) -> None:
    server = create_server(host, port)
    actual_host, actual_port = server.server_address
    print(f"Mimir Python API listening on http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
