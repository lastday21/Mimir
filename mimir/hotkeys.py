from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass


MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
VK_M = 0x4D
VK_SPACE = 0x20
DEFAULT_OVERLAY_HOTKEY = "Ctrl+M"
DEFAULT_AUDIO_HOTKEY = "Ctrl+Space"

KEY_ALIASES = {
    "SPACE": VK_SPACE,
    "ПРОБЕЛ": VK_SPACE,
    "ENTER": 0x0D,
    "RETURN": 0x0D,
    "TAB": 0x09,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "BACKSPACE": 0x08,
    "DELETE": 0x2E,
    "DEL": 0x2E,
    "INSERT": 0x2D,
    "INS": 0x2D,
    "HOME": 0x24,
    "END": 0x23,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
}

KEY_NAMES = {
    **{ord(letter): letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{ord(digit): digit for digit in "0123456789"},
    VK_SPACE: "Space",
    0x0D: "Enter",
    0x09: "Tab",
    0x1B: "Esc",
    0x08: "Backspace",
    0x2E: "Delete",
    0x2D: "Insert",
    0x24: "Home",
    0x23: "End",
    0x21: "PageUp",
    0x22: "PageDown",
    0x26: "Up",
    0x28: "Down",
    0x25: "Left",
    0x27: "Right",
}

for number in range(1, 25):
    KEY_ALIASES[f"F{number}"] = 0x70 + number - 1
    KEY_NAMES[0x70 + number - 1] = f"F{number}"


@dataclass(frozen=True)
class HotkeySpec:
    identifier: int
    modifiers: int
    virtual_key: int
    callback: Callable[[], None]


class WindowsHotkeyController:
    def __init__(self, specs: list[HotkeySpec]) -> None:
        self.specs = specs
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()

    def start(self) -> None:
        if sys.platform != "win32" or not self.specs:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="mimir-hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2)

    def stop(self) -> None:
        if sys.platform != "win32" or not self._thread:
            return
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._thread.join(timeout=2)
        self._thread = None
        self._thread_id = 0

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        registered: list[int] = []
        try:
            for spec in self.specs:
                if user32.RegisterHotKey(None, spec.identifier, spec.modifiers, spec.virtual_key):
                    registered.append(spec.identifier)
            self._ready.set()

            msg = ctypes.wintypes.MSG()
            callbacks = {spec.identifier: spec.callback for spec in self.specs}
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY:
                    callback = callbacks.get(int(msg.wParam))
                    if callback is not None:
                        callback()
        finally:
            for identifier in registered:
                user32.UnregisterHotKey(None, identifier)
            self._ready.set()


def parse_hotkey_text(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.replace("-", "+").split("+") if part.strip()]
    if len(parts) < 2:
        raise ValueError("Hotkey must contain at least one modifier and one key")

    modifiers = 0
    virtual_key: int | None = None
    for part in parts:
        normalized = part.upper()
        if normalized in {"CTRL", "CONTROL"}:
            modifiers |= MOD_CONTROL
        elif normalized == "ALT":
            modifiers |= MOD_ALT
        elif normalized == "SHIFT":
            modifiers |= MOD_SHIFT
        elif normalized in {"WIN", "WINDOWS"}:
            modifiers |= MOD_WIN
        else:
            if virtual_key is not None:
                raise ValueError("Hotkey can contain only one non-modifier key")
            virtual_key = virtual_key_from_text(normalized)

    if modifiers == 0 or virtual_key is None:
        raise ValueError("Hotkey must contain at least one modifier and one key")
    return modifiers, virtual_key


def normalize_hotkey_text(text: str) -> str:
    modifiers, virtual_key = parse_hotkey_text(text)
    modifier_names: list[str] = []
    if modifiers & MOD_CONTROL:
        modifier_names.append("Ctrl")
    if modifiers & MOD_ALT:
        modifier_names.append("Alt")
    if modifiers & MOD_SHIFT:
        modifier_names.append("Shift")
    if modifiers & MOD_WIN:
        modifier_names.append("Win")
    key_name = KEY_NAMES.get(virtual_key)
    if key_name is None:
        raise ValueError("Unsupported hotkey key")
    return "+".join([*modifier_names, key_name])


def virtual_key_from_text(text: str) -> int:
    if len(text) == 1 and ("A" <= text <= "Z" or "0" <= text <= "9"):
        return ord(text)
    key = KEY_ALIASES.get(text)
    if key is None:
        raise ValueError(f"Unsupported hotkey key: {text}")
    return key


def overlay_hotkey(callback: Callable[[], None], text: str = DEFAULT_OVERLAY_HOTKEY) -> HotkeySpec:
    modifiers, virtual_key = parse_hotkey_text(text)
    return HotkeySpec(1, modifiers, virtual_key, callback)


def audio_hotkey(callback: Callable[[], None], text: str = DEFAULT_AUDIO_HOTKEY) -> HotkeySpec:
    modifiers, virtual_key = parse_hotkey_text(text)
    return HotkeySpec(2, modifiers, virtual_key, callback)
