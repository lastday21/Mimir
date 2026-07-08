from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..dialogue import MIC_SOURCE, REMOTE_SOURCE


class AudioCaptureError(RuntimeError):
    pass


HEADSET_MIC_KEYWORDS = (
    "headset",
    "headphone",
    "headphones",
    "hands-free",
    "handsfree",
    "earbuds",
    "earphone",
    "airpods",
    "buds",
    "bluetooth",
    "wireless",
    "jabra",
    "sony",
    "bose",
    "steelseries",
    "hyperx",
    "razer",
    "logitech",
    "corsair",
    "sennheiser",
    "plantronics",
    "poly",
    "гарнитур",
    "науш",
)
BUILTIN_MIC_KEYWORDS = (
    "built-in",
    "internal",
    "integrated",
    "microphone array",
    "array",
    "realtek",
    "intel",
    "встро",
)
VIRTUAL_MIC_KEYWORDS = (
    "loopback",
    "stereo mix",
    "what u hear",
    "monitor",
    "virtual",
    "cable",
    "vb-audio",
    "voicemeeter",
)


@dataclass(frozen=True)
class AudioCaptureConfig:
    sample_rate_hertz: int = 16_000
    chunk_duration_ms: int = 200
    device_id: str | None = None


class SoundcardPcmSource:
    def __init__(self, source: str, config: AudioCaptureConfig | None = None) -> None:
        self.source = source
        self.config = config or AudioCaptureConfig()

    def chunks(self, stop_event: threading.Event) -> Iterator[bytes]:
        sc = import_soundcard()
        device = select_capture_device(sc, self.source, self.config.device_id)
        frames_per_chunk = max(
            1,
            round(self.config.sample_rate_hertz * self.config.chunk_duration_ms / 1000),
        )

        with device.recorder(
            samplerate=self.config.sample_rate_hertz,
            channels=1,
            blocksize=frames_per_chunk,
        ) as recorder:
            while not stop_event.is_set():
                frames = recorder.record(numframes=frames_per_chunk)
                pcm = float_frames_to_pcm16(frames)
                if pcm:
                    yield pcm


def list_audio_devices() -> list[dict[str, Any]]:
    sc = import_soundcard()
    devices: list[dict[str, Any]] = []
    default_mic = safe_call(sc.default_microphone)
    default_speaker = safe_call(sc.default_speaker)
    recommended_mic = safe_select_capture_device(sc, MIC_SOURCE)
    recommended_loopback = safe_select_capture_device(sc, REMOTE_SOURCE)

    for device in sc.all_microphones(include_loopback=True):
        loopback = is_loopback_device(device)
        recommended = same_device(device, recommended_loopback) if loopback else same_device(device, recommended_mic)
        devices.append(
            {
                "id": device_id(device),
                "name": device_name(device),
                "source": REMOTE_SOURCE if loopback else MIC_SOURCE,
                "loopback": loopback,
                "recommended": recommended,
                "default": same_device(device, default_mic)
                or (loopback and devices_look_related(device, default_speaker)),
            }
        )
    return devices


def select_capture_device(sc: Any, source: str, preferred_id: str | None = None) -> Any:
    normalized = source.strip().lower()
    if normalized == MIC_SOURCE:
        return select_microphone(sc, preferred_id)
    if normalized == REMOTE_SOURCE:
        return select_loopback(sc, preferred_id)
    raise AudioCaptureError("audio source must be remote or mic")


def select_microphone(sc: Any, preferred_id: str | None = None) -> Any:
    microphones = list(sc.all_microphones(include_loopback=False))
    if preferred_id:
        match = find_device(microphones, preferred_id)
        if match is not None:
            return match
        raise AudioCaptureError(f"Microphone device was not found: {preferred_id}")
    default = safe_call(sc.default_microphone)
    if microphones:
        return max(microphones, key=lambda device: microphone_score(device, default))
    if default is not None:
        return default
    raise AudioCaptureError("No microphone capture devices were found")


def select_loopback(sc: Any, preferred_id: str | None = None) -> Any:
    microphones = list(sc.all_microphones(include_loopback=True))
    loopbacks = [device for device in microphones if is_loopback_device(device)]
    if preferred_id:
        match = find_device(loopbacks or microphones, preferred_id)
        if match is not None:
            return match
        raise AudioCaptureError(f"Loopback device was not found: {preferred_id}")

    speaker = safe_call(sc.default_speaker)
    if speaker is not None:
        for device in loopbacks:
            if devices_look_related(device, speaker):
                return device
    if loopbacks:
        return max(loopbacks, key=lambda device: loopback_score(device, speaker))
    raise AudioCaptureError("No loopback capture device was found for system audio")


def import_soundcard() -> Any:
    try:
        import soundcard as sc
    except ImportError as error:
        raise AudioCaptureError(
            "Live audio capture requires the `soundcard` package. Reinstall with `pip install -e .`."
        ) from error
    return sc


def safe_call(function: Any) -> Any | None:
    try:
        return function()
    except Exception:
        return None


def safe_select_capture_device(sc: Any, source: str) -> Any | None:
    try:
        return select_capture_device(sc, source)
    except AudioCaptureError:
        return None


def find_device(devices: list[Any], key: str) -> Any | None:
    wanted = normalize_device_key(key)
    for device in devices:
        if normalize_device_key(device_id(device)) == wanted:
            return device
        if normalize_device_key(device_name(device)) == wanted:
            return device
    return None


def is_loopback_device(device: Any) -> bool:
    if bool(getattr(device, "isloopback", False)):
        return True
    name = device_name(device).lower()
    identifier = device_id(device).lower()
    return "loopback" in name or "loopback" in identifier


def same_device(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return normalize_device_key(device_id(left)) == normalize_device_key(device_id(right)) or normalize_device_key(
        device_name(left)
    ) == normalize_device_key(device_name(right))


def devices_look_related(left: Any, right: Any) -> bool:
    if right is None:
        return False
    left_name = normalize_device_key(device_name(left))
    right_name = normalize_device_key(device_name(right))
    left_id = normalize_device_key(device_id(left))
    right_id = normalize_device_key(device_id(right))
    return right_name in left_name or left_name in right_name or right_id in left_id or left_id in right_id


def microphone_score(device: Any, default: Any | None = None) -> int:
    text = normalized_device_text(device)
    score = 10
    if same_device(device, default):
        score += 20
    if has_keyword(text, HEADSET_MIC_KEYWORDS):
        score += 120
    if has_keyword(text, BUILTIN_MIC_KEYWORDS):
        score += 45
    if has_keyword(text, VIRTUAL_MIC_KEYWORDS):
        score -= 80
    return score


def loopback_score(device: Any, speaker: Any | None = None) -> int:
    score = 10
    if devices_look_related(device, speaker):
        score += 100
    return score


def normalized_device_text(device: Any) -> str:
    return f"{normalize_device_key(device_name(device))} {normalize_device_key(device_id(device))}"


def has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def device_id(device: Any) -> str:
    return str(getattr(device, "id", None) or getattr(device, "_id", None) or device_name(device))


def device_name(device: Any) -> str:
    return str(getattr(device, "name", None) or device)


def normalize_device_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def float_frames_to_pcm16(frames: Any) -> bytes:
    try:
        import numpy as np
    except ImportError:
        return float_sequence_to_pcm16(frames)

    data = np.asarray(frames, dtype=np.float32)
    if data.size == 0:
        return b""
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.clip(data, -1.0, 1.0)
    return np.rint(data * 32767).astype("<i2").tobytes()


def float_sequence_to_pcm16(frames: Any) -> bytes:
    samples = bytearray()
    for frame in frames:
        sample = frame[0] if isinstance(frame, (list, tuple)) else frame
        value = max(-1.0, min(1.0, float(sample)))
        samples.extend(round(value * 32767).to_bytes(2, "little", signed=True))
    return bytes(samples)
