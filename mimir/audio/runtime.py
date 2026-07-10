from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..providers.base import ProviderError
from .live import LiveAudioConfig
from .realtime import RealtimeAudioConfig


@dataclass(frozen=True)
class AudioRuntimeDependencies:
    session_manager: Any
    live_audio: Any
    local_audio: Any
    realtime_audio: Any
    load_config: Callable[[], Any]
    read_secret: Callable[[str], str | None]


def start_live_audio_locked(
    payload: dict[str, Any],
    dependencies: AudioRuntimeDependencies,
    parse_request: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
) -> dict[str, object]:
    mode, common_config = parse_request(payload)
    stop_all_audio_locked(dependencies)
    if mode == "speechkit":
        dependencies.session_manager.set_answer_provider_override(None)
        key = dependencies.read_secret("yandex_speechkit") or dependencies.read_secret("yandex_ai_studio") or ""
        return dependencies.live_audio.start(LiveAudioConfig(**common_config), key)
    if mode == "local_vosk":
        dependencies.session_manager.set_answer_provider_override("ollama")
        return dependencies.local_audio.start(LiveAudioConfig(**common_config), "")
    if mode != "yandex_realtime":
        raise ValueError("audio mode must be yandex_realtime, speechkit, or local_vosk")

    dependencies.session_manager.set_answer_provider_override(None)
    config = dependencies.load_config()
    key = dependencies.read_secret("yandex_ai_studio") or dependencies.read_secret("yandex_speechkit") or ""
    realtime_config = RealtimeAudioConfig(**common_config)
    try:
        return dependencies.realtime_audio.start(realtime_config, key, config.yandex_folder_id)
    except ProviderError as error:
        return start_cloud_audio_fallback_locked(realtime_config, str(error), dependencies)


def audio_is_running(dependencies: AudioRuntimeDependencies) -> bool:
    return bool(
        dependencies.realtime_audio.snapshot().get("running")
        or dependencies.live_audio.snapshot().get("running")
        or dependencies.local_audio.snapshot().get("running")
    )


def stop_all_audio_locked(dependencies: AudioRuntimeDependencies) -> None:
    dependencies.realtime_audio.stop()
    dependencies.live_audio.stop()
    dependencies.local_audio.stop()


def stop_live_audio(dependencies: AudioRuntimeDependencies) -> dict[str, object]:
    active = active_audio_snapshot(dependencies)
    stop_all_audio_locked(dependencies)
    dependencies.session_manager.set_answer_provider_override(None)
    return {
        **idle_audio_snapshot(),
        "mode": str(active.get("mode") or "idle"),
    }


def pause_live_session(dependencies: AudioRuntimeDependencies) -> dict[str, Any]:
    stop_all_audio_locked(dependencies)
    dependencies.session_manager.set_answer_provider_override(None)
    return dependencies.session_manager.pause()


def stop_live_session(dependencies: AudioRuntimeDependencies) -> dict[str, Any]:
    stop_all_audio_locked(dependencies)
    dependencies.session_manager.set_answer_provider_override(None)
    return dependencies.session_manager.stop()


def active_audio_snapshot(dependencies: AudioRuntimeDependencies) -> dict[str, object]:
    for controller in (
        dependencies.realtime_audio,
        dependencies.live_audio,
        dependencies.local_audio,
    ):
        snapshot = controller.snapshot()
        if snapshot.get("running"):
            return snapshot
    return idle_audio_snapshot()


def idle_audio_snapshot() -> dict[str, object]:
    return {
        "running": False,
        "mode": "idle",
        "sources": [],
        "language": "ru-RU",
        "sampleRateHertz": 16_000,
        "chunkDurationMs": 200,
        "vadEnabled": True,
        "deviceIds": {},
    }


def start_cloud_audio_fallback_locked(
    config: RealtimeAudioConfig,
    reason: str,
    dependencies: AudioRuntimeDependencies,
) -> dict[str, object]:
    stop_all_audio_locked(dependencies)
    dependencies.session_manager.set_answer_provider_override(None)
    dependencies.session_manager.publish_status(
        "audio_status",
        {
            "status": "fallback",
            "mode": "speechkit",
            "source": "remote",
            "reason": reason,
            "running": True,
        },
    )
    key = dependencies.read_secret("yandex_speechkit") or dependencies.read_secret("yandex_ai_studio") or ""
    try:
        return dependencies.live_audio.start(copy_live_audio_config(config), key)
    except Exception as error:
        combined_reason = f"Realtime: {reason}. SpeechKit: {error}"
        return start_local_audio_fallback_locked(config, combined_reason, dependencies)


def start_local_audio_fallback_locked(
    config: RealtimeAudioConfig | LiveAudioConfig,
    reason: str,
    dependencies: AudioRuntimeDependencies,
) -> dict[str, object]:
    stop_all_audio_locked(dependencies)
    dependencies.session_manager.set_answer_provider_override("ollama")
    dependencies.session_manager.publish_status(
        "audio_status",
        {
            "status": "fallback",
            "mode": "local_vosk",
            "source": "remote",
            "reason": reason,
            "running": True,
        },
    )
    return dependencies.local_audio.start(copy_live_audio_config(config), "")


def copy_live_audio_config(config: RealtimeAudioConfig | LiveAudioConfig) -> LiveAudioConfig:
    return LiveAudioConfig(
        sources=config.sources,
        language=config.language,
        sample_rate_hertz=config.sample_rate_hertz,
        chunk_duration_ms=config.chunk_duration_ms,
        vad_enabled=config.vad_enabled,
        vad=config.vad,
        device_ids=dict(config.device_ids),
    )
