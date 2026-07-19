from __future__ import annotations

import argparse
import json
import time
from typing import Any

from mimir import server


def select_recording(recording_id: str) -> dict[str, Any]:
    recordings = [
        item
        for item in server.CALL_RECORDINGS.list()
        if item.get("status") != "recording"
    ]
    if not recordings:
        raise RuntimeError("Нет завершенных записей для проверки")
    if recording_id == "latest":
        return recordings[0]
    recording = next((item for item in recordings if item.get("id") == recording_id), None)
    if recording is None:
        raise RuntimeError(f"Запись не найдена: {recording_id}")
    return recording


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Повтор сохраненного созвона через рабочий путь Mimir",
    )
    parser.add_argument(
        "recording_id",
        nargs="?",
        default="latest",
        help="Номер записи или latest для последней завершенной записи",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Предельное время проверки в секундах",
    )
    args = parser.parse_args()

    try:
        recording = select_recording(args.recording_id)
        duration_seconds = int(recording.get("durationMs") or 0) / 1_000
        timeout = args.timeout if args.timeout > 0 else max(180.0, duration_seconds + 180.0)
        server.start_testing_replay(str(recording["id"]))
        deadline = time.monotonic() + timeout
        while server.TESTING_REPLAY.is_running() and time.monotonic() < deadline:
            time.sleep(0.5)
        if server.TESTING_REPLAY.is_running():
            server.stop_testing_replay()
            raise TimeoutError(f"Проверка не завершилась за {timeout:.0f} секунд")
        replay = server.TESTING_REPLAY.snapshot()
        print(
            json.dumps(
                {
                    "recording": {
                        "id": recording.get("id"),
                        "durationMs": recording.get("durationMs"),
                        "manifestPath": recording.get("manifestPath"),
                        "eventsPath": recording.get("eventsPath"),
                    },
                    "replay": replay,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if replay.get("state") == "completed" else 1
    except KeyboardInterrupt:
        server.stop_testing_replay()
        print("Проверка остановлена")
        return 130
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
