from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import app_data_dir
from ..models import SpeechRecognitionResult
from ..providers.base import ProviderError


DEFAULT_VOSK_MODEL_NAME = "vosk-model-small-ru-0.22"
DEFAULT_VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_VOSK_MODEL_NAME}.zip"
VOSK_MODEL_ENV = "MIMIR_VOSK_MODEL_PATH"

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[Path, Any] = {}


class LocalVoskRecognizer:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = ensure_vosk_model(model_path)
        self.model = load_vosk_model(self.model_path)

    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str,
        sample_rate_hertz: int,
    ) -> Iterator[SpeechRecognitionResult]:
        try:
            import vosk
        except ImportError as error:
            raise ProviderError("Local STT requires `vosk`. Reinstall with `pip install -e .`.") from error

        recognizer = vosk.KaldiRecognizer(self.model, sample_rate_hertz)
        recognizer.SetWords(False)
        last_partial = ""

        for chunk in chunks:
            if not chunk:
                continue
            if recognizer.AcceptWaveform(chunk):
                text = vosk_text(recognizer.Result(), "text")
                if text:
                    last_partial = ""
                    yield SpeechRecognitionResult(text=text, is_final=True, end_of_utterance=True)
                continue

            partial = vosk_text(recognizer.PartialResult(), "partial")
            if partial and partial != last_partial:
                last_partial = partial
                yield SpeechRecognitionResult(text=partial, is_final=False, end_of_utterance=False)

        final_text = vosk_text(recognizer.FinalResult(), "text")
        if final_text:
            yield SpeechRecognitionResult(text=final_text, is_final=True, end_of_utterance=True)


def ensure_vosk_model(model_path: str | Path | None = None, *, download: bool = True) -> Path:
    configured = model_path or os.environ.get(VOSK_MODEL_ENV)
    if configured:
        path = Path(configured).expanduser()
        if is_vosk_model_dir(path):
            return path
        if not download:
            raise ProviderError(f"Local STT model is not ready: {path}")
        install_vosk_model(path)
        if not is_vosk_model_dir(path):
            raise ProviderError(f"Local STT model install did not produce a valid model: {path}")
        return path

    target = default_vosk_model_dir()
    if is_vosk_model_dir(target):
        return target
    if not download:
        raise ProviderError(f"Local STT model is not installed: {target}")

    install_vosk_model(target)
    if not is_vosk_model_dir(target):
        raise ProviderError(f"Local STT model install did not produce a valid model: {target}")
    return target


def local_vosk_status() -> dict[str, object]:
    configured = os.environ.get(VOSK_MODEL_ENV)
    path = Path(configured).expanduser() if configured else default_vosk_model_dir()
    return {
        "provider": "vosk",
        "model": DEFAULT_VOSK_MODEL_NAME,
        "path": str(path),
        "installed": is_vosk_model_dir(path),
        "sourceUrl": DEFAULT_VOSK_MODEL_URL,
    }


def default_vosk_model_dir() -> Path:
    return app_data_dir() / "models" / "vosk" / DEFAULT_VOSK_MODEL_NAME


def install_vosk_model(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / f"{DEFAULT_VOSK_MODEL_NAME}.zip"
    partial = archive.with_suffix(".zip.tmp")

    urllib.request.urlretrieve(DEFAULT_VOSK_MODEL_URL, partial)
    partial.replace(archive)

    extract_dir = target.parent / f"{DEFAULT_VOSK_MODEL_NAME}.extracting"
    if extract_dir.exists():
        remove_tree(extract_dir)
    extract_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as package:
            safe_extract(package, extract_dir)
        unpacked = extract_dir / DEFAULT_VOSK_MODEL_NAME
        if not is_vosk_model_dir(unpacked):
            raise ProviderError(f"Downloaded Vosk package does not contain {DEFAULT_VOSK_MODEL_NAME}")
        if target.exists():
            remove_tree(target)
        unpacked.replace(target)
    finally:
        if extract_dir.exists():
            remove_tree(extract_dir)


def load_vosk_model(model_path: Path) -> object:
    try:
        import vosk
    except ImportError as error:
        raise ProviderError("Local STT requires `vosk`. Reinstall with `pip install -e .`.") from error

    path = model_path.resolve()
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(path)
        if model is None:
            vosk.SetLogLevel(-1)
            model = vosk.Model(str(path))
            _MODEL_CACHE[path] = model
        return model


def is_vosk_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "am").is_dir() and (path / "conf").is_dir()


def safe_extract(package: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in package.infolist():
        destination = (target / member.filename).resolve()
        if root != destination and root not in destination.parents:
            raise ProviderError(f"Unsafe path in Vosk package: {member.filename}")
    package.extractall(target)


def remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def vosk_text(payload: str, key: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    return str(data.get(key) or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install or inspect the local Vosk STT model for Mimir.")
    parser.add_argument("--install", action="store_true", help="Download and unpack the default local STT model.")
    args = parser.parse_args()

    if args.install:
        path = ensure_vosk_model()
        print(f"Local STT model ready: {path}")
        return
    print(json.dumps(local_vosk_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
