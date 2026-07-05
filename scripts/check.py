from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    if not compileall.compile_dir(ROOT / "mimir", quiet=1):
        return 1
    if not compileall.compile_dir(ROOT / "tests", quiet=1):
        return 1
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
