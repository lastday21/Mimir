from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


APP_DIR_NAME = "io.github.lastday21.mimir"
CONFIG_FILE = "config.json"
AUDIO_MODES = {"yandex_realtime", "local_vosk"}


@dataclass
class AppConfig:
    yandex_folder_id: str = ""
    llm_provider: str = "yandex_ai_studio"
    llm_model: str = "yandexgpt/latest"
    audio_mode: str = "yandex_realtime"
    ollama_base_url: str = "http://localhost:11434"
    overlay_hotkey: str = "Ctrl+M"
    audio_hotkey: str = "Ctrl+Space"

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppConfig":
        hotkeys = data.get("hotkeys")
        hotkey_data = hotkeys if isinstance(hotkeys, dict) else {}
        provider = str(data.get("llmProvider") or data.get("llm_provider") or "yandex_ai_studio")
        default_audio_mode = "local_vosk" if provider == "ollama" else "yandex_realtime"
        audio_mode = str(data.get("audioMode") or data.get("audio_mode") or default_audio_mode)
        if audio_mode not in AUDIO_MODES:
            audio_mode = default_audio_mode
        if provider == "ollama":
            audio_mode = "local_vosk"
        elif audio_mode == "local_vosk":
            audio_mode = "yandex_realtime"
        return cls(
            yandex_folder_id=str(
                data.get("yandexFolderId")
                or data.get("yandex_folder_id")
                or ""
            ),
            llm_provider=provider,
            llm_model=str(data.get("llmModel") or data.get("llm_model") or "yandexgpt/latest"),
            audio_mode=audio_mode,
            ollama_base_url=str(data.get("ollamaBaseUrl") or data.get("ollama_base_url") or "http://localhost:11434"),
            overlay_hotkey=str(
                hotkey_data.get("overlayToggle")
                or data.get("overlayHotkey")
                or data.get("overlay_hotkey")
                or "Ctrl+M"
            ),
            audio_hotkey=str(
                hotkey_data.get("audioToggle")
                or data.get("audioHotkey")
                or data.get("audio_hotkey")
                or "Ctrl+Space"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        audio_mode = str(data["audio_mode"])
        if audio_mode not in AUDIO_MODES:
            audio_mode = "local_vosk" if data["llm_provider"] == "ollama" else "yandex_realtime"
        return {
            "yandexFolderId": data["yandex_folder_id"],
            "llmProvider": data["llm_provider"],
            "llmModel": data["llm_model"],
            "audioMode": audio_mode,
            "ollamaBaseUrl": data["ollama_base_url"],
            "hotkeys": {
                "overlayToggle": data["overlay_hotkey"],
                "audioToggle": data["audio_hotkey"],
            },
        }


def app_data_dir() -> Path:
    root = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(root) / APP_DIR_NAME


def config_path() -> Path:
    return app_data_dir() / CONFIG_FILE


def load_config() -> AppConfig:
    path = config_path()
    if not path.exists():
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    if not isinstance(data, dict):
        return AppConfig()
    return AppConfig.from_dict(data)


def save_config(config: AppConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
