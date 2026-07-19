from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


APP_DIR_NAME = "io.github.lastday21.mimir"
CONFIG_FILE = "config.json"
AUDIO_MODES = {"speechkit", "local_vosk"}
CONVERSATION_MODES = {"interview", "meeting", "technical", "custom"}


@dataclass
class UserProfile:
    name: str = ""
    role: str = ""
    background: str = ""
    projects: str = ""
    stories: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "UserProfile":
        return cls(
            name=str(data.get("name") or ""),
            role=str(data.get("role") or ""),
            background=str(data.get("background") or ""),
            projects=str(data.get("projects") or ""),
            stories=str(data.get("stories") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ConversationSettings:
    mode: str = "interview"
    goal: str = ""
    context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationSettings":
        mode = str(data.get("mode") or "interview").strip().lower()
        if mode not in CONVERSATION_MODES:
            mode = "interview"
        return cls(
            mode=mode,
            goal=str(data.get("goal") or ""),
            context=str(data.get("context") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class AudioApplicationSettings:
    process_id: int = 0
    executable: str = ""
    title: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AudioApplicationSettings":
        try:
            process_id = int(data.get("processId") or data.get("process_id") or 0)
        except (TypeError, ValueError):
            process_id = 0
        return cls(
            process_id=max(0, process_id),
            executable=str(data.get("executable") or ""),
            title=str(data.get("title") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "processId": self.process_id,
            "executable": self.executable,
            "title": self.title,
        }


@dataclass
class TestingSettings:
    enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TestingSettings":
        return cls(enabled=data.get("enabled") is True)

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class AppConfig:
    yandex_folder_id: str = ""
    llm_provider: str = "yandex_ai_studio"
    llm_model: str = "yandexgpt/latest"
    audio_mode: str = "speechkit"
    ollama_base_url: str = "http://localhost:11434"
    overlay_hotkey: str = "Ctrl+M"
    audio_hotkey: str = "Ctrl+Space"
    audio_application: AudioApplicationSettings = field(default_factory=AudioApplicationSettings)
    profile: UserProfile = field(default_factory=UserProfile)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    testing: TestingSettings = field(default_factory=TestingSettings)
    setup_completed: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppConfig":
        hotkeys = data.get("hotkeys")
        hotkey_data = hotkeys if isinstance(hotkeys, dict) else {}
        profile = data.get("profile")
        profile_data = profile if isinstance(profile, dict) else {}
        conversation = data.get("conversation")
        conversation_data = conversation if isinstance(conversation, dict) else {}
        testing = data.get("testing")
        testing_data = testing if isinstance(testing, dict) else {}
        if not testing_data and data.get("testingEnabled") is True:
            testing_data = {"enabled": True}
        audio_application = data.get("audioApplication") or data.get("audio_application")
        audio_application_data = audio_application if isinstance(audio_application, dict) else {}
        provider = str(data.get("llmProvider") or data.get("llm_provider") or "yandex_ai_studio")
        default_audio_mode = "local_vosk" if provider == "ollama" else "speechkit"
        audio_mode = str(data.get("audioMode") or data.get("audio_mode") or default_audio_mode)
        if audio_mode == "yandex_realtime":
            audio_mode = "speechkit"
        if audio_mode not in AUDIO_MODES:
            audio_mode = default_audio_mode
        if provider == "ollama":
            audio_mode = "local_vosk"
        elif audio_mode == "local_vosk":
            audio_mode = "speechkit"
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
            audio_application=AudioApplicationSettings.from_dict(audio_application_data),
            profile=UserProfile.from_dict(profile_data),
            conversation=ConversationSettings.from_dict(conversation_data),
            testing=TestingSettings.from_dict(testing_data),
            setup_completed=data.get("setupCompleted") is True or data.get("setup_completed") is True,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        audio_mode = str(data["audio_mode"])
        if audio_mode == "yandex_realtime":
            audio_mode = "speechkit"
        if audio_mode not in AUDIO_MODES:
            audio_mode = "local_vosk" if data["llm_provider"] == "ollama" else "speechkit"
        return {
            "yandexFolderId": data["yandex_folder_id"],
            "llmProvider": data["llm_provider"],
            "llmModel": data["llm_model"],
            "audioMode": audio_mode,
            "audioApplication": self.audio_application.to_dict(),
            "ollamaBaseUrl": data["ollama_base_url"],
            "profile": self.profile.to_dict(),
            "conversation": self.conversation.to_dict(),
            "testing": self.testing.to_dict(),
            "setupCompleted": self.setup_completed,
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
