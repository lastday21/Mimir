from __future__ import annotations

import argparse
import ctypes
import http.client
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .config import app_data_dir, load_config
from .hotkeys import HotkeySpec, WindowsHotkeyController, audio_hotkey, overlay_hotkey
from .server import HOST, STATIC_ROOT, create_server, stop_live_session, toggle_live_audio


APP_TITLE = "Mimir"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_INSTANCE_MUTEX = "Local\\io.github.lastday21.mimir"
ERROR_ALREADY_EXISTS = 183


class SingleInstanceGuard:
    def __init__(self, name: str = SINGLE_INSTANCE_MUTEX) -> None:
        self.name = name
        self._kernel32: object | None = None
        self._handle: object | None = None

    def acquire(self) -> bool:
        if os.name != "nt":
            return True
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._kernel32 = kernel32
        self._handle = handle
        return True

    def release(self) -> None:
        kernel32 = self._kernel32
        handle = self._handle
        self._kernel32 = None
        self._handle = None
        if kernel32 is not None and handle is not None:
            kernel32.CloseHandle(handle)

    def __enter__(self) -> "SingleInstanceGuard":
        if not self.acquire():
            raise RuntimeError("Mimir уже запущен")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def show_already_running_message() -> None:
    message = "Mimir уже запущен. Закройте открытое окно перед повторным запуском."
    if os.name != "nt":
        print(message, file=sys.stderr)
        return
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
        user32.MessageBoxW.restype = ctypes.c_int
        user32.MessageBoxW(None, message, APP_TITLE, 0x40)
    except OSError:
        print(message, file=sys.stderr)


@dataclass
class DesktopServer:
    host: str = HOST
    port: int = 0

    def __post_init__(self) -> None:
        self._server = create_server(self.host, self.port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mimir-http",
            daemon=True,
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def shutdown_desktop_runtime(server: DesktopServer) -> None:
    try:
        stop_live_session()
    except Exception as error:
        print(f"Не удалось полностью остановить звук Mimir: {error}", file=sys.stderr)
    finally:
        server.stop()


def ensure_frontend_build(auto_build: bool) -> None:
    index_file = STATIC_ROOT / "index.html"
    if index_file.exists():
        return
    if not auto_build:
        raise RuntimeError("Frontend build not found. Run npm run build.")
    package_json = PROJECT_ROOT / "package.json"
    if not package_json.exists():
        raise RuntimeError("Frontend sources not found. Run npm run build from the project root.")
    subprocess.run(["npm", "run", "build"], cwd=PROJECT_ROOT, check=True)
    if not index_file.exists():
        raise RuntimeError("Frontend build finished, but dist/index.html was not created.")


def smoke_server(url: str) -> None:
    host, port_text = url.removeprefix("http://").split(":", 1)
    port = int(port_text)

    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("GET", "/api/health")
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200 or b'"ok": true' not in body:
        raise RuntimeError(f"Health check failed: HTTP {response.status}")

    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("GET", "/")
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200 or b'<div id="root"></div>' not in body:
        raise RuntimeError(f"Frontend check failed: HTTP {response.status}")


class DesktopWindowController:
    def __init__(self, overlay_window: object, overlay_visible: bool = False) -> None:
        self.overlay_window = overlay_window
        self.overlay_visible = overlay_visible
        self._lock = threading.Lock()
        config = load_config()
        self.hotkeys = WindowsHotkeyController(
            [
                self._hotkey_or_default(overlay_hotkey, self.toggle_overlay, config.overlay_hotkey),
                self._hotkey_or_default(audio_hotkey, self.toggle_audio, config.audio_hotkey),
            ]
        )

    def start(self) -> None:
        self.hotkeys.start()

    def stop(self) -> None:
        self.hotkeys.stop()

    def _hotkey_or_default(
        self,
        factory: Callable[[Callable[[], None], str], HotkeySpec],
        callback: Callable[[], None],
        text: str,
    ) -> HotkeySpec:
        try:
            return factory(callback, text)
        except ValueError:
            return factory(callback)

    def toggle_overlay(self) -> None:
        with self._lock:
            if self.overlay_visible:
                self.overlay_window.hide()
                self.overlay_visible = False
                return
            self.overlay_window.show()
            self.overlay_window.on_top = True
            self.overlay_visible = True

    def toggle_audio(self) -> None:
        try:
            toggle_live_audio()
        except Exception as error:
            print(f"Audio hotkey failed: {error}", file=sys.stderr)


def open_window(url: str, debug: bool) -> None:
    try:
        import webview
    except ImportError as error:
        raise RuntimeError(
            "pywebview is not installed. Install the project with `python -m pip install -e .`."
        ) from error

    storage_path = app_data_dir() / "webview"
    storage_path.mkdir(parents=True, exist_ok=True)

    main_window = webview.create_window(
        APP_TITLE,
        url,
        width=1240,
        height=820,
        min_size=(900, 620),
        text_select=True,
        background_color="#0e1116",
    )
    overlay_window = webview.create_window(
        "Mimir Overlay",
        f"{url}/#overlay",
        width=460,
        height=360,
        min_size=(360, 240),
        hidden=True,
        frameless=True,
        easy_drag=True,
        on_top=True,
        text_select=True,
        background_color="#0e1116",
    )
    if main_window is None or overlay_window is None:
        raise RuntimeError("Failed to create desktop windows.")

    controller = DesktopWindowController(overlay_window)
    try:
        webview.start(
            func=controller.start,
            debug=debug,
            private_mode=False,
            storage_path=str(storage_path),
        )
    finally:
        controller.stop()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Mimir as a desktop app.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-build", action="store_true", help="Require an existing dist/ frontend build.")
    parser.add_argument("--debug", action="store_true", help="Enable pywebview debug mode.")
    parser.add_argument("--check", action="store_true", help="Start the local server and run smoke checks without opening a window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    instance = SingleInstanceGuard()
    try:
        if not args.check and not instance.acquire():
            show_already_running_message()
            return 2
        ensure_frontend_build(auto_build=not args.no_build)
        server = DesktopServer(args.host, args.port)
        server.start()
        try:
            smoke_server(server.url)
            if args.check:
                print(f"Mimir desktop check ok at {server.url}")
                return 0
            open_window(server.url, args.debug)
        finally:
            shutdown_desktop_runtime(server)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Failed to start Mimir desktop: {error}", file=sys.stderr)
        return 1
    finally:
        instance.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
