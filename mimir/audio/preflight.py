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
    list_audio_applications: Callable[[], list[dict[str, Any]]] = lambda: []
    speechkit_probe: Callable[[str], str] | None = None
    yandex_model_probe: Callable[[str, str, str], str] | None = None
    audio_source_probe: Callable[[str, str | None, int], str] | None = None


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
    config = load_config()
    mode = str(payload.get("mode") or config.audio_mode).strip().lower()
    if mode not in {"yandex_realtime", "speechkit", "local_vosk"}:
        raise ValueError("audio mode must be yandex_realtime, speechkit, or local_vosk")
    try:
        application_process_id = int(
            payload.get("applicationProcessId")
            or getattr(getattr(config, "audio_application", None), "process_id", 0)
            or 0
        )
    except (TypeError, ValueError):
        application_process_id = 0
    saved_testing = getattr(getattr(config, "testing", None), "enabled", False)
    requested_testing = payload.get("recordTesting")
    record_testing = bool(saved_testing) if requested_testing is None else requested_testing is True
    return (
        mode,
        {
            "sources": tuple(normalize_live_audio_source(str(source)) for source in sources),
            "language": str(payload.get("language") or "ru-RU"),
            "sample_rate_hertz": int(payload.get("sampleRateHertz") or 16_000),
            "chunk_duration_ms": int(payload.get("chunkDurationMs") or 200),
            "vad_enabled": bool(payload.get("vadEnabled", True)),
            "device_ids": {str(key): str(value) for key, value in device_ids.items()},
            "application_process_id": max(0, application_process_id),
            "record_testing": record_testing,
        },
    )


def build_live_audio_preflight(
    payload: dict[str, Any],
    dependencies: PreflightDependencies,
) -> dict[str, Any]:
    mode, common_config = parse_live_audio_request(payload, dependencies.load_config)
    sources = list(common_config["sources"])
    device_ids = dict(common_config["device_ids"])
    application_process_id = int(common_config["application_process_id"])
    record_testing = bool(common_config["record_testing"])
    checks: list[dict[str, Any]] = []

    audio_running = dependencies.audio_is_running()
    add_preflight_check(
        checks,
        "audio_idle",
        not audio_running,
        "Live audio is already running" if audio_running else "Live audio is idle",
    )
    if mode == "yandex_realtime":
        add_yandex_answer_checks(checks, dependencies)
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
        add_yandex_answer_checks(checks, dependencies)
        add_speechkit_checks(checks, dependencies)
    elif mode == "local_vosk":
        add_local_stt_checks(checks, dependencies)
        add_ollama_checks(checks, dependencies)

    add_device_checks(
        checks,
        sources,
        device_ids,
        dependencies.list_audio_devices,
        application_process_id=application_process_id,
        list_audio_applications=dependencies.list_audio_applications,
    )
    if not audio_running:
        add_audio_source_probe_checks(
            checks,
            sources,
            device_ids,
            application_process_id,
            dependencies.audio_source_probe,
        )
    errors = [str(check["detail"]) for check in checks if not check["ok"]]
    return {
        "ok": not errors,
        "mode": mode,
        "sources": sources,
        "deviceIds": device_ids,
        "applicationProcessId": application_process_id,
        "recordTesting": record_testing,
        "checks": checks,
        "errors": errors,
    }


def add_speechkit_checks(checks: list[dict[str, Any]], dependencies: PreflightDependencies) -> None:
    key = dependencies.read_secret("yandex_speechkit") or dependencies.read_secret("yandex_ai_studio") or ""
    key_ready = bool(key)
    add_preflight_check(
        checks,
        "yandex_speechkit_key",
        key_ready,
        "Yandex SpeechKit key is configured" if key_ready else "Yandex SpeechKit API key is missing",
    )
    grpc_ready = add_import_check(
        checks,
        "grpc",
        "grpc",
        "SpeechKit gRPC dependency is missing",
        dependencies.import_module,
    )
    stubs_ready = add_import_check(
        checks,
        "yandexcloud_stt",
        "yandex.cloud.ai.stt.v3.stt_service_pb2_grpc",
        "SpeechKit stubs are missing",
        dependencies.import_module,
    )
    probe = dependencies.speechkit_probe
    if key_ready and grpc_ready and stubs_ready and probe is not None:
        add_runtime_probe_check(
            checks,
            "speechkit_connection",
            lambda: probe(key),
            "Соединение со SpeechKit установлено",
        )


def add_yandex_answer_checks(checks: list[dict[str, Any]], dependencies: PreflightDependencies) -> None:
    config = dependencies.load_config()
    folder_id = str(config.yandex_folder_id).strip()
    folder_ready = bool(folder_id)
    add_preflight_check(
        checks,
        "yandex_folder_id",
        folder_ready,
        "Yandex folder is configured" if folder_ready else "Yandex folder ID is missing",
    )
    key = dependencies.read_secret("yandex_ai_studio") or dependencies.read_secret("yandex_speechkit") or ""
    key_ready = bool(key)
    add_preflight_check(
        checks,
        "yandex_ai_studio_key",
        key_ready,
        "Yandex AI Studio key is configured" if key_ready else "Yandex AI Studio API key is missing",
    )
    probe = dependencies.yandex_model_probe
    if folder_ready and key_ready and probe is not None:
        model = str(getattr(config, "llm_model", "")).strip()
        add_runtime_probe_check(
            checks,
            "yandex_model_connection",
            lambda: probe(key, folder_id, model),
            "Соединение с моделью установлено",
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
    *,
    application_process_id: int = 0,
    list_audio_applications: Callable[[], list[dict[str, Any]]] = lambda: [],
) -> None:
    if "remote" in sources:
        try:
            applications = list_audio_applications()
        except AudioCaptureError as error:
            add_preflight_check(checks, "remote_application", False, str(error))
        else:
            selected = next(
                (
                    application
                    for application in applications
                    if int(application.get("processId") or 0) == application_process_id
                ),
                None,
            )
            if selected is None:
                detail = (
                    "Выбранное приложение созвона не запущено"
                    if application_process_id
                    else "Выберите приложение созвона в настройках"
                )
                add_preflight_check(checks, "remote_application", False, detail)
            else:
                name = str(selected.get("title") or selected.get("executable") or application_process_id)
                add_preflight_check(checks, "remote_application", True, f"Приложение созвона: {name}")

    if "mic" not in sources:
        return

    try:
        devices = list_audio_devices()
    except AudioCaptureError as error:
        add_preflight_check(checks, "audio_devices", False, str(error))
        return

    add_preflight_check(checks, "audio_devices", True, f"{len(devices)} capture devices available")
    for source in sources:
        if source == "remote":
            continue
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


def add_audio_source_probe_checks(
    checks: list[dict[str, Any]],
    sources: list[str],
    device_ids: dict[str, str],
    application_process_id: int,
    probe: Callable[[str, str | None, int], str] | None,
) -> None:
    if probe is None:
        return
    for source in sources:
        prerequisite = "remote_application" if source == "remote" else f"{source}_device"
        if not check_passed(checks, prerequisite):
            continue
        label = "программы созвона" if source == "remote" else "микрофона"
        add_runtime_probe_check(
            checks,
            f"{source}_capture",
            lambda source=source: probe(
                source,
                device_ids.get(source) or None,
                application_process_id,
            ),
            f"Захват {label} открывается",
        )


def check_passed(checks: list[dict[str, Any]], name: str) -> bool:
    return any(check.get("name") == name and check.get("ok") is True for check in checks)


def add_runtime_probe_check(
    checks: list[dict[str, Any]],
    name: str,
    probe: Callable[[], str],
    success_detail: str,
) -> None:
    try:
        detail = probe().strip() or success_detail
    except Exception as error:
        detail = str(error).strip() or error.__class__.__name__
        add_preflight_check(checks, name, False, detail)
        return
    add_preflight_check(checks, name, True, detail)


def add_import_check(
    checks: list[dict[str, Any]],
    name: str,
    module: str,
    error: str,
    import_module: Callable[[str], Any],
) -> bool:
    try:
        import_module(module)
    except ImportError:
        add_preflight_check(checks, name, False, error)
        return False
    add_preflight_check(checks, name, True, "available")
    return True


def add_preflight_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail})
