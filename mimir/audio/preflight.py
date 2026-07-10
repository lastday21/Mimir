from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..providers.base import ProviderError
from .capture import AudioCaptureError


@dataclass(frozen=True)
class PreflightDependencies:
    load_config: Callable[[], Any]
    read_secret: Callable[[str], str | None]
    list_audio_devices: Callable[[], list[dict[str, Any]]]
    local_vosk_status: Callable[[], dict[str, Any]]
    ollama_client: type[Any]
    select_preferred_model: Callable[[list[Any]], Any]
    import_module: Callable[[str], Any]
    audio_is_running: Callable[[], bool]


def parse_live_audio_request(
    payload: dict[str, Any],
    load_config: Callable[[], Any],
) -> tuple[str, dict[str, Any]]:
    sources = payload.get("sources") or ["remote", "mic"]
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    device_ids = payload.get("deviceIds") or {}
    if not isinstance(device_ids, dict):
        raise ValueError("deviceIds must be an object")
    mode = str(payload.get("mode") or load_config().audio_mode).strip().lower()
    if mode not in {"yandex_realtime", "speechkit", "local_vosk"}:
        raise ValueError("audio mode must be yandex_realtime, speechkit, or local_vosk")
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


def build_live_audio_preflight(
    payload: dict[str, Any],
    dependencies: PreflightDependencies,
) -> dict[str, Any]:
    mode, common_config = parse_live_audio_request(payload, dependencies.load_config)
    sources = list(common_config["sources"])
    device_ids = dict(common_config["device_ids"])
    checks: list[dict[str, Any]] = []

    audio_running = dependencies.audio_is_running()
    add_preflight_check(
        checks,
        "audio_idle",
        not audio_running,
        "Live audio is already running" if audio_running else "Live audio is idle",
    )
    if mode == "yandex_realtime":
        config = dependencies.load_config()
        add_preflight_check(
            checks,
            "yandex_folder_id",
            bool(config.yandex_folder_id.strip()),
            "Yandex folder ID is missing",
        )
        add_preflight_check(
            checks,
            "yandex_ai_studio_key",
            bool(dependencies.read_secret("yandex_ai_studio") or dependencies.read_secret("yandex_speechkit")),
            "Yandex AI Studio API key is missing",
        )
        add_import_check(
            checks,
            "aiohttp",
            "aiohttp",
            "Realtime websocket dependency is missing",
            dependencies.import_module,
        )
        if "mic" in sources:
            add_speechkit_checks(checks, dependencies)
    elif mode == "speechkit":
        add_speechkit_checks(checks, dependencies)
    elif mode == "local_vosk":
        add_local_stt_checks(checks, dependencies)
        add_ollama_checks(checks, dependencies)

    add_device_checks(checks, sources, device_ids, dependencies.list_audio_devices)
    errors = [str(check["detail"]) for check in checks if not check["ok"]]
    return {
        "ok": not errors,
        "mode": mode,
        "sources": sources,
        "deviceIds": device_ids,
        "checks": checks,
        "errors": errors,
    }


def add_speechkit_checks(checks: list[dict[str, Any]], dependencies: PreflightDependencies) -> None:
    add_preflight_check(
        checks,
        "yandex_speechkit_key",
        bool(dependencies.read_secret("yandex_speechkit") or dependencies.read_secret("yandex_ai_studio")),
        "Yandex SpeechKit API key is missing",
    )
    add_import_check(
        checks,
        "grpc",
        "grpc",
        "SpeechKit gRPC dependency is missing",
        dependencies.import_module,
    )
    add_import_check(
        checks,
        "yandexcloud_stt",
        "yandex.cloud.ai.stt.v3.stt_service_pb2_grpc",
        "SpeechKit stubs are missing",
        dependencies.import_module,
    )


def add_local_stt_checks(checks: list[dict[str, Any]], dependencies: PreflightDependencies) -> None:
    add_import_check(
        checks,
        "vosk",
        "vosk",
        "Local Vosk STT dependency is missing",
        dependencies.import_module,
    )
    status = dependencies.local_vosk_status()
    detail = f"{status['model']} at {status['path']}"
    if not status["installed"]:
        detail = f"{detail}; install the model before starting local mode"
    add_preflight_check(checks, "local_stt_model", bool(status["installed"]), detail)


def add_ollama_checks(checks: list[dict[str, Any]], dependencies: PreflightDependencies) -> None:
    config = dependencies.load_config()
    try:
        models = dependencies.ollama_client(config.ollama_base_url).list_models()
    except ProviderError as error:
        add_preflight_check(checks, "ollama", False, str(error))
        return
    preferred = dependencies.select_preferred_model(models)
    add_preflight_check(
        checks,
        "ollama_model",
        preferred is not None,
        f"preferred local model: {preferred.id}" if preferred else "No local Ollama models are installed",
    )


def normalize_live_audio_source(source: str) -> str:
    value = source.strip().lower()
    if value in {"remote", "them", "system", "loopback"}:
        return "remote"
    if value in {"mic", "me", "user"}:
        return "mic"
    raise ValueError("audio source must be remote or mic")


def add_device_checks(
    checks: list[dict[str, Any]],
    sources: list[str],
    device_ids: dict[str, str],
    list_audio_devices: Callable[[], list[dict[str, Any]]],
) -> None:
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
            detail = f"{source} device selected: {device_id}" if ok else f"{source} device is not available: {device_id}"
            add_preflight_check(checks, f"{source}_device", ok, detail)
        else:
            detail = (
                f"{len(source_devices)} {source} capture devices available"
                if source_devices
                else f"No {source} capture device is available"
            )
            add_preflight_check(checks, f"{source}_device", bool(source_devices), detail)


def add_import_check(
    checks: list[dict[str, Any]],
    name: str,
    module: str,
    error: str,
    import_module: Callable[[str], Any],
) -> None:
    try:
        import_module(module)
    except ImportError:
        add_preflight_check(checks, name, False, error)
        return
    add_preflight_check(checks, name, True, "available")


def add_preflight_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail})
