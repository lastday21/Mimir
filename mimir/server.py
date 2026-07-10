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
from .audio.preflight import (
    PreflightDependencies,
    add_device_checks as add_audio_device_checks,
    add_import_check as add_audio_import_check,
    add_ollama_checks as add_audio_ollama_checks,
    add_preflight_check as append_preflight_check,
    add_speechkit_checks as add_audio_speechkit_checks,
    build_live_audio_preflight as prepare_live_audio_preflight,
    normalize_live_audio_source as normalize_audio_source,
    parse_live_audio_request as parse_audio_request,
)
from .audio.runtime import (
    AudioRuntimeDependencies,
    active_audio_snapshot as current_audio_snapshot,
    audio_is_running as runtime_audio_is_running,
    copy_live_audio_config as clone_live_audio_config,
    idle_audio_snapshot as empty_audio_snapshot,
    pause_live_session as pause_audio_session,
    start_cloud_audio_fallback_locked as start_cloud_fallback,
    start_live_audio_locked as start_audio,
    start_local_audio_fallback_locked as start_local_fallback,
    stop_all_audio_locked as stop_audio_controllers,
    stop_live_audio as stop_audio,
    stop_live_session as stop_audio_session,
)
from .config import AppConfig, load_config, save_config
from .credentials import read_secret, write_secret
from .hotkeys import normalize_hotkey_text
from .models import ModelInfo
from .ollama_fallback import select_preferred_model, sort_models
from .providers import OllamaClient, YandexAIStudioClient, YandexSpeechKitClient
from .providers.base import ProviderError
from .session import SessionEvent, SessionManager, sse_payload
from .stt import AudioStreamConfig, SpeechKitStreamRunner, pcm_chunks_from_wav
from .stt.local_vosk import LocalVoskRecognizer, local_vosk_status


HOST = "127.0.0.1"
PORT = 8765
STATIC_ROOT = Path(__file__).resolve().parents[1] / "dist"
ALLOWED_CORS_HEADERS = "Content-Type"
SESSION_MANAGER = SessionManager()
LIVE_AUDIO = LiveAudioController(
    SESSION_MANAGER,
    YandexSpeechKitClient,
    fallback_starter=lambda config, reason: start_local_audio_fallback(config, reason),
)
LOCAL_AUDIO = LiveAudioController(
    SESSION_MANAGER,
    lambda _key: LocalVoskRecognizer(),
    mode="local_vosk",
    requires_api_key=False,
)
REALTIME_AUDIO = RealtimeAudioController(
    SESSION_MANAGER,
    YandexSpeechKitClient,
    fallback_starter=lambda config, reason: start_cloud_audio_fallback(config, reason),
)
AUDIO_CONTROL_LOCK = threading.RLock()
MAX_DEV_WAV_BYTES = 25_000_000


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "MimirPython/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
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
        except ProviderError as error:
            self.send_json({"error": str(error)}, status=502)
        except AudioCaptureError as error:
            self.send_json({"error": str(error)}, status=502)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

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
        config.overlay_hotkey = normalize_hotkey_text(config.overlay_hotkey)
        config.audio_hotkey = normalize_hotkey_text(config.audio_hotkey)
        if config.overlay_hotkey == config.audio_hotkey:
            raise ValueError("Hotkeys must be different")
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
        self.send_json(stop_live_session())

    def handle_session_pause(self) -> None:
        self.send_json(pause_live_session())

    def handle_live_audio_preflight(self) -> None:
        self.send_json(build_live_audio_preflight(self.read_json()))

    def handle_live_audio_start(self) -> None:
        self.send_json(start_live_audio(self.read_json()))

    def handle_live_audio_stop(self) -> None:
        self.send_json(stop_live_audio())

    def handle_session_transcript(self) -> None:
        payload = self.read_json()
        source = str(payload.get("source") or "")
        text = str(payload.get("text") or "")
        is_final = bool(payload.get("isFinal", True))
        is_refinement = bool(payload.get("isRefinement", False))
        self.send_json(
            SESSION_MANAGER.ingest_transcript(
                source,
                text,
                is_final=is_final,
                is_refinement=is_refinement,
            )
        )

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
        has_cursor = False
        if params.get("after"):
            try:
                after = int(params["after"][0])
                has_cursor = True
            except ValueError:
                after = 0
        last_event_id = self.headers.get("Last-Event-ID")
        if last_event_id:
            try:
                after = max(after, int(last_event_id))
                has_cursor = True
            except ValueError:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", ALLOWED_CORS_HEADERS)
        self.end_headers()
        try:
            if not has_cursor:
                snapshot = SESSION_MANAGER.snapshot()
                after = int(snapshot["eventSequence"])
                self.wfile.write(sse_payload(SessionEvent(after, "session_snapshot", snapshot)))
                self.wfile.flush()
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


def start_live_audio(payload: dict[str, Any]) -> dict[str, object]:
    with AUDIO_CONTROL_LOCK:
        return start_live_audio_locked(payload)


def start_live_audio_locked(payload: dict[str, Any]) -> dict[str, object]:
    return start_audio(payload, audio_runtime_dependencies(), parse_live_audio_request)


def audio_runtime_dependencies() -> AudioRuntimeDependencies:
    return AudioRuntimeDependencies(
        session_manager=SESSION_MANAGER,
        live_audio=LIVE_AUDIO,
        local_audio=LOCAL_AUDIO,
        realtime_audio=REALTIME_AUDIO,
        load_config=load_config,
        read_secret=read_secret,
    )


def config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "yandexFolderId": config.yandex_folder_id,
        "llmProvider": config.llm_provider,
        "llmModel": config.llm_model,
        "audioMode": config.audio_mode,
        "ollamaBaseUrl": config.ollama_base_url,
        "hasYandexKey": bool(read_secret("yandex_ai_studio")),
        "hotkeys": {
            "overlayToggle": config.overlay_hotkey,
            "audioToggle": config.audio_hotkey,
        },
    }


def model_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "contextWindow": model.context_window,
    }


def parse_live_audio_request(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return parse_audio_request(payload, load_config)


def build_live_audio_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    return prepare_live_audio_preflight(payload, preflight_dependencies())


def preflight_dependencies() -> PreflightDependencies:
    return PreflightDependencies(
        load_config=load_config,
        read_secret=read_secret,
        list_audio_devices=list_audio_devices,
        local_vosk_status=local_vosk_status,
        ollama_client=OllamaClient,
        select_preferred_model=select_preferred_model,
        import_module=import_module,
        audio_is_running=audio_is_running,
    )


def add_speechkit_checks(checks: list[dict[str, Any]]) -> None:
    add_audio_speechkit_checks(checks, preflight_dependencies())


def add_local_stt_checks(checks: list[dict[str, Any]]) -> None:
    add_import_check(checks, "vosk", "vosk", "Local Vosk STT dependency is missing")
    status = local_vosk_status()
    detail = f"{status['model']} at {status['path']}"
    if not status["installed"]:
        detail = f"{detail}; install the model before starting local mode"
    add_preflight_check(checks, "local_stt_model", bool(status["installed"]), detail)


def add_ollama_checks(checks: list[dict[str, Any]]) -> None:
    add_audio_ollama_checks(checks, preflight_dependencies())


def normalize_live_audio_source(source: str) -> str:
    return normalize_audio_source(source)


def add_device_checks(checks: list[dict[str, Any]], sources: list[str], device_ids: dict[str, str]) -> None:
    add_audio_device_checks(checks, sources, device_ids, list_audio_devices)


def add_import_check(checks: list[dict[str, Any]], name: str, module: str, error: str) -> None:
    add_audio_import_check(checks, name, module, error, import_module)


def add_preflight_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    append_preflight_check(checks, name, ok, detail)


def audio_is_running() -> bool:
    return runtime_audio_is_running(audio_runtime_dependencies())


def stop_all_audio() -> None:
    with AUDIO_CONTROL_LOCK:
        stop_all_audio_locked()


def stop_all_audio_locked() -> None:
    stop_audio_controllers(audio_runtime_dependencies())


def stop_live_audio() -> dict[str, object]:
    with AUDIO_CONTROL_LOCK:
        return stop_audio(audio_runtime_dependencies())


def pause_live_session() -> dict[str, Any]:
    with AUDIO_CONTROL_LOCK:
        return pause_audio_session(audio_runtime_dependencies())


def stop_live_session() -> dict[str, Any]:
    with AUDIO_CONTROL_LOCK:
        return stop_audio_session(audio_runtime_dependencies())


def toggle_live_audio() -> dict[str, object]:
    with AUDIO_CONTROL_LOCK:
        mode = "idle"
        try:
            if audio_is_running():
                pause_live_session()
                return idle_audio_snapshot()

            config = load_config()
            mode = config.audio_mode
            request = {
                "mode": mode,
                "sources": ["remote", "mic"],
                "language": "ru-RU",
                "vadEnabled": True,
            }
            preflight = build_live_audio_preflight(request)
            if not preflight["ok"]:
                errors = preflight.get("errors") or ["Live audio preflight failed"]
                raise ProviderError(str(errors[0]))
            return start_live_audio_locked(request)
        except Exception as error:
            SESSION_MANAGER.publish_status(
                "audio_error",
                {
                    "source": "remote",
                    "mode": mode,
                    "phase": "control",
                    "error": str(error),
                    "running": False,
                },
            )
            raise


def active_audio_snapshot() -> dict[str, object]:
    return current_audio_snapshot(audio_runtime_dependencies())


def idle_audio_snapshot() -> dict[str, object]:
    return empty_audio_snapshot()


def start_cloud_audio_fallback(config: RealtimeAudioConfig, reason: str) -> dict[str, object]:
    with AUDIO_CONTROL_LOCK:
        return start_cloud_audio_fallback_locked(config, reason)


def start_cloud_audio_fallback_locked(config: RealtimeAudioConfig, reason: str) -> dict[str, object]:
    return start_cloud_fallback(config, reason, audio_runtime_dependencies())


def start_local_audio_fallback(config: RealtimeAudioConfig | LiveAudioConfig, reason: str) -> dict[str, object]:
    with AUDIO_CONTROL_LOCK:
        return start_local_audio_fallback_locked(config, reason)


def start_local_audio_fallback_locked(
    config: RealtimeAudioConfig | LiveAudioConfig,
    reason: str,
) -> dict[str, object]:
    return start_local_fallback(config, reason, audio_runtime_dependencies())


def copy_live_audio_config(config: RealtimeAudioConfig | LiveAudioConfig) -> LiveAudioConfig:
    return clone_live_audio_config(config)


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
        SESSION_MANAGER.record_audio_speech_started(source)
        SESSION_MANAGER.record_audio_chunk(source, len(data))
        for event in runner.run(source, chunks):
            SESSION_MANAGER.record_stt_result(event.source, event.is_final)
            SESSION_MANAGER.ingest_transcript(
                event.source,
                event.text,
                is_final=event.is_final,
                is_refinement=event.is_refinement,
            )
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
