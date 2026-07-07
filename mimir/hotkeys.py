from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
VK_M = 0x4D
VK_P = 0x50


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


def overlay_hotkey(callback: Callable[[], None]) -> HotkeySpec:
    return HotkeySpec(1, MOD_CONTROL | MOD_ALT, VK_M, callback)


def audio_hotkey(callback: Callable[[], None]) -> HotkeySpec:
    return HotkeySpec(2, MOD_CONTROL | MOD_ALT, VK_P, callback)
