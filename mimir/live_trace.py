from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / ".work" / "live-traces"
MAX_TEXT_LENGTH = 4_000

_lock = threading.Lock()
_trace_path: Path | None = None


def trace_live_event(event: str, **payload: Any) -> None:
    if not trace_enabled():
        return
    try:
        path = current_trace_path()
        item = {
            "ts": int(time.time() * 1000),
            "event": event,
            **sanitize(payload),
        }
        raw = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(raw)
                handle.write("\n")
    except Exception:
        return


def current_trace_path() -> Path:
    global _trace_path
    with _lock:
        if _trace_path is None:
            directory = Path(os.environ.get("MIMIR_LIVE_TRACE_DIR") or DEFAULT_TRACE_DIR)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            _trace_path = directory / f"live-{stamp}-{os.getpid()}.jsonl"
        return _trace_path


def trace_enabled() -> bool:
    value = os.environ.get("MIMIR_LIVE_TRACE", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def trace_path_payload() -> str:
    return str(current_trace_path())


def reset_trace_for_tests() -> None:
    global _trace_path
    with _lock:
        _trace_path = None


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in {"apikey", "api_key", "authorization", "key", "token"}:
                clean[name] = "<redacted>"
            else:
                clean[name] = sanitize(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) > MAX_TEXT_LENGTH:
            return f"{value[:MAX_TEXT_LENGTH]}...<truncated {len(value) - MAX_TEXT_LENGTH}>"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)
