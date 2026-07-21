from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
import wave
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


REMOTE_SOURCE = "remote"
MIC_SOURCE = "mic"
SOURCES = (REMOTE_SOURCE, MIC_SOURCE)
SAMPLE_RATE_HERTZ = 16_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
DEFAULT_CHUNK_DURATION_MS = 200
MAX_TEXT_LENGTH = 4_000
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
_SECRET_KEYS = ("apikey", "api_key", "authorization", "password", "secret", "token", "credential")


class CallRecordingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallRecordingDescriptor:
    recording_id: str
    directory: Path

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.directory / "events.jsonl"

    def track_path(self, source: str) -> Path:
        return self.directory / "audio" / f"{normalize_source(source)}.wav"


@dataclass
class _ActiveRecording:
    descriptor: CallRecordingDescriptor
    manifest: dict[str, Any]
    started_monotonic_ns: int
    writers: dict[str, wave.Wave_write]
    events_handle: TextIO
    frames: dict[str, int]
    received_audio: dict[str, bool]
    non_silent_frames: dict[str, int]
    last_non_silent_frame: dict[str, int]
    inserted_gap_frames: dict[str, int]
    stream_dropped_frames: dict[str, int]
    capture_dropped_frames: dict[str, int]
    write_failed: bool = False


class CallRecordingStore:
    def __init__(
        self,
        root: str | Path,
        *,
        sample_rate_hertz: int = SAMPLE_RATE_HERTZ,
        chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
    ) -> None:
        if sample_rate_hertz != SAMPLE_RATE_HERTZ:
            raise ValueError("call recordings require 16000 Hz audio")
        if chunk_duration_ms <= 0:
            raise ValueError("chunk duration must be positive")
        self.root = Path(root).expanduser().resolve()
        self.sample_rate_hertz = sample_rate_hertz
        self.chunk_duration_ms = int(chunk_duration_ms)
        self._lock = threading.RLock()
        self._active: _ActiveRecording | None = None

    def start(
        self,
        recording_id: str | None = None,
        *,
        title: str = "",
        application: Mapping[str, Any] | None = None,
        audio_mode: str = "speechkit",
        app_revision: str = "",
        started_monotonic_ns: int | None = None,
        created_at_ms: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise CallRecordingError("a call recording is already active")

            clean_id = validate_safe_id(recording_id or make_recording_id())
            descriptor = CallRecordingDescriptor(clean_id, self.root / clean_id)
            if descriptor.directory.exists():
                raise FileExistsError(f"recording already exists: {clean_id}")

            descriptor.directory.mkdir(parents=True)
            (descriptor.directory / "audio").mkdir()
            writers: dict[str, wave.Wave_write] = {}
            events_handle: TextIO | None = None
            try:
                for source in SOURCES:
                    writer = wave.open(str(descriptor.track_path(source)), "wb")
                    writer.setnchannels(CHANNELS)
                    writer.setsampwidth(SAMPLE_WIDTH_BYTES)
                    writer.setframerate(self.sample_rate_hertz)
                    writers[source] = writer
                events_handle = descriptor.events_path.open("x", encoding="utf-8", buffering=64 * 1024)
            except Exception:
                for writer in writers.values():
                    try:
                        writer.close()
                    except Exception:
                        pass
                if events_handle is not None:
                    try:
                        events_handle.close()
                    except Exception:
                        pass
                shutil.rmtree(descriptor.directory, ignore_errors=True)
                raise
            assert events_handle is not None

            now_ms = int(time.time() * 1000) if created_at_ms is None else int(created_at_ms)
            manifest: dict[str, Any] = {
                "schemaVersion": 1,
                "id": clean_id,
                "status": "recording",
                "createdAtMs": now_ms,
                "sampleRateHertz": self.sample_rate_hertz,
                "sampleWidthBytes": SAMPLE_WIDTH_BYTES,
                "channels": CHANNELS,
                "chunkDurationMs": self.chunk_duration_ms,
                "durationMs": 0,
                "sources": list(SOURCES),
                "title": clean_text(title, 200),
                "application": sanitize(dict(application or {})),
                "audioMode": clean_text(audio_mode, 100),
                "appRevision": clean_text(app_revision, 200),
                "tracks": {
                    source: {
                        "path": f"audio/{source}.wav",
                        "frames": 0,
                        "receivedAudio": False,
                        "nonSilentFrames": 0,
                        "lastNonSilentAtMs": None,
                        "silentTailMs": 0,
                        "insertedGapFrames": 0,
                        "insertedGapMs": 0,
                        "streamDroppedFrames": 0,
                        "streamDroppedAudioMs": 0,
                        "captureDroppedFrames": 0,
                        "captureDroppedAudioMs": 0,
                        "tailPaddingFrames": 0,
                        "tailPaddingMs": 0,
                        "sizeBytes": 0,
                        "sha256": "",
                    }
                    for source in SOURCES
                },
                "sizeBytes": 0,
            }
            self._active = _ActiveRecording(
                descriptor=descriptor,
                manifest=manifest,
                started_monotonic_ns=(
                    time.monotonic_ns() if started_monotonic_ns is None else int(started_monotonic_ns)
                ),
                writers=writers,
                events_handle=events_handle,
                frames={source: 0 for source in SOURCES},
                received_audio={source: False for source in SOURCES},
                non_silent_frames={source: 0 for source in SOURCES},
                last_non_silent_frame={source: 0 for source in SOURCES},
                inserted_gap_frames={source: 0 for source in SOURCES},
                stream_dropped_frames={source: 0 for source in SOURCES},
                capture_dropped_frames={source: 0 for source in SOURCES},
            )
            self._write_manifest(descriptor, manifest)
            self._append_event_locked("recording.started", recordingId=clean_id)
            return self._descriptor(manifest, descriptor)

    def write(
        self,
        source: str,
        pcm: bytes,
        *,
        captured_at_ns: int | None = None,
        recording_id: str = "",
    ) -> bool:
        try:
            with self._lock:
                active = self._active
                if active is None:
                    return False
                if recording_id and active.descriptor.recording_id != recording_id:
                    return False
                try:
                    clean_source = normalize_source(source)
                    raw = bytes(pcm)
                    if not raw or len(raw) % SAMPLE_WIDTH_BYTES:
                        raise ValueError("PCM block must contain complete 16-bit samples")

                    sample_count = len(raw) // SAMPLE_WIDTH_BYTES
                    observed_at_ns = time.monotonic_ns() if captured_at_ns is None else int(captured_at_ns)
                    elapsed_ns = max(0, observed_at_ns - active.started_monotonic_ns)
                    elapsed_frames = round(elapsed_ns * self.sample_rate_hertz / 1_000_000_000)
                    expected_start_frame = max(0, elapsed_frames - sample_count)
                    if not active.received_audio[clean_source]:
                        leading_frames = expected_start_frame
                        self._write_silence(active, clean_source, leading_frames)
                        active.received_audio[clean_source] = True
                        active.manifest["tracks"][clean_source]["receivedAudio"] = True
                    else:
                        gap_frames = expected_start_frame - active.frames[clean_source]
                        tolerance_frames = max(
                            1,
                            round(self.sample_rate_hertz * min(50, self.chunk_duration_ms / 4) / 1000),
                        )
                        if gap_frames > tolerance_frames:
                            gap_offset = active.frames[clean_source]
                            self._write_silence(active, clean_source, gap_frames)
                            active.inserted_gap_frames[clean_source] += gap_frames
                            active.manifest["tracks"][clean_source]["insertedGapFrames"] = (
                                active.inserted_gap_frames[clean_source]
                            )
                            active.manifest["tracks"][clean_source]["insertedGapMs"] = round(
                                active.inserted_gap_frames[clean_source]
                                * 1000
                                / self.sample_rate_hertz
                            )
                            self._append_event_locked(
                                "audio.gap",
                                source=clean_source,
                                offsetFrames=gap_offset,
                                frames=gap_frames,
                                gapMs=round(gap_frames * 1000 / self.sample_rate_hertz),
                                elapsedMs=round(elapsed_ns / 1_000_000),
                            )

                    offset_frames = active.frames[clean_source]
                    active.writers[clean_source].writeframesraw(raw)
                    active.frames[clean_source] += sample_count
                    non_silent_count = 0
                    last_non_silent_index = -1
                    for index in range(sample_count):
                        byte_offset = index * SAMPLE_WIDTH_BYTES
                        if raw[byte_offset] or raw[byte_offset + 1]:
                            non_silent_count += 1
                            last_non_silent_index = index
                    if non_silent_count:
                        active.non_silent_frames[clean_source] += non_silent_count
                        active.last_non_silent_frame[clean_source] = (
                            offset_frames + last_non_silent_index + 1
                        )
                        track = active.manifest["tracks"][clean_source]
                        track["nonSilentFrames"] = active.non_silent_frames[clean_source]
                        track["lastNonSilentAtMs"] = round(
                            active.last_non_silent_frame[clean_source]
                            * 1000
                            / self.sample_rate_hertz
                        )
                    self._append_event_locked(
                        "audio.chunk",
                        source=clean_source,
                        offsetFrames=offset_frames,
                        frames=sample_count,
                        bytes=len(raw),
                        elapsedMs=max(
                            0,
                            round((observed_at_ns - active.started_monotonic_ns) / 1_000_000),
                        ),
                    )
                    return True
                except Exception as error:
                    active.write_failed = True
                    self._append_event_locked(
                        "recording.write_error",
                        source=str(source),
                        error=str(error),
                    )
                    return False
        except Exception:
            return False

    def record_stream_drop(
        self,
        source: str,
        frames: int,
        *,
        captured_at_ns: int | None = None,
    ) -> bool:
        """Record audio discarded before recognition because its live queue overflowed."""
        try:
            with self._lock:
                active = self._active
                if active is None:
                    return False
                clean_source = normalize_source(source)
                dropped_frames = max(0, int(frames))
                if dropped_frames <= 0:
                    return False
                active.stream_dropped_frames[clean_source] += dropped_frames
                total_frames = active.stream_dropped_frames[clean_source]
                track = active.manifest["tracks"][clean_source]
                track["streamDroppedFrames"] = total_frames
                track["streamDroppedAudioMs"] = round(
                    total_frames * 1000 / self.sample_rate_hertz
                )
                active.manifest["hasRecognitionAudioLoss"] = True
                observed_at_ns = (
                    time.monotonic_ns() if captured_at_ns is None else int(captured_at_ns)
                )
                return self._append_event_locked(
                    "audio.stream_drop",
                    source=clean_source,
                    frames=dropped_frames,
                    droppedAudioMs=round(dropped_frames * 1000 / self.sample_rate_hertz),
                    totalDroppedFrames=total_frames,
                    elapsedMs=max(
                        0,
                        round((observed_at_ns - active.started_monotonic_ns) / 1_000_000),
                    ),
                )
        except Exception:
            return False

    def record_capture_drop(
        self,
        source: str,
        frames: int,
        *,
        captured_at_ns: int | None = None,
    ) -> bool:
        """Record raw audio lost because the recording queue overflowed."""
        try:
            with self._lock:
                active = self._active
                if active is None:
                    return False
                clean_source = normalize_source(source)
                dropped_frames = max(0, int(frames))
                if dropped_frames <= 0:
                    return False
                active.capture_dropped_frames[clean_source] += dropped_frames
                total_frames = active.capture_dropped_frames[clean_source]
                track = active.manifest["tracks"][clean_source]
                track["captureDroppedFrames"] = total_frames
                track["captureDroppedAudioMs"] = round(
                    total_frames * 1000 / self.sample_rate_hertz
                )
                active.manifest["hasCaptureAudioLoss"] = True
                observed_at_ns = (
                    time.monotonic_ns() if captured_at_ns is None else int(captured_at_ns)
                )
                return self._append_event_locked(
                    "audio.capture_drop",
                    source=clean_source,
                    frames=dropped_frames,
                    droppedAudioMs=round(dropped_frames * 1000 / self.sample_rate_hertz),
                    totalDroppedFrames=total_frames,
                    elapsedMs=max(
                        0,
                        round((observed_at_ns - active.started_monotonic_ns) / 1_000_000),
                    ),
                )
        except Exception:
            return False

    def record_event(self, event: str, **payload: Any) -> bool:
        try:
            with self._lock:
                if self._active is None:
                    return False
                return self._append_event_locked(event, **payload)
        except Exception:
            return False

    def finish(
        self,
        *,
        error: str = "",
        finished_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            active = self._active
            if active is None:
                raise CallRecordingError("no call recording is active")

            audio_frames = max(active.frames.values(), default=0)
            wall_frames = audio_frames
            if finished_monotonic_ns is not None:
                elapsed_ns = max(0, int(finished_monotonic_ns) - active.started_monotonic_ns)
                wall_frames = round(elapsed_ns * self.sample_rate_hertz / 1_000_000_000)
            target_frames = max(audio_frames, wall_frames)
            errors: list[str] = []
            for source in SOURCES:
                try:
                    tail_padding_frames = max(0, target_frames - active.frames[source])
                    active.manifest["tracks"][source]["tailPaddingFrames"] = tail_padding_frames
                    active.manifest["tracks"][source]["tailPaddingMs"] = round(
                        tail_padding_frames * 1000 / self.sample_rate_hertz
                    )
                    self._write_silence(active, source, tail_padding_frames)
                except Exception as error:
                    active.write_failed = True
                    errors.append(f"{source}: {error}")
                try:
                    active.writers[source].close()
                except Exception as error:
                    active.write_failed = True
                    errors.append(f"{source} close: {error}")

            finished_at_ms = int(time.time() * 1000)
            received_sources = sum(1 for received in active.received_audio.values() if received)
            has_timeline_gaps = any(active.inserted_gap_frames.values())
            has_stream_drops = any(active.stream_dropped_frames.values())
            has_capture_drops = any(active.capture_dropped_frames.values())
            missing_non_silent_audio = any(
                active.received_audio[source] and not active.non_silent_frames[source]
                for source in SOURCES
            )
            for source in SOURCES:
                silent_tail_frames = max(
                    0,
                    target_frames - active.last_non_silent_frame[source],
                )
                silent_tail_ms = round(silent_tail_frames * 1000 / self.sample_rate_hertz)
                active.manifest["tracks"][source]["silentTailMs"] = silent_tail_ms
            excessive_tail_padding = any(
                int(active.manifest["tracks"][source]["tailPaddingFrames"])
                > self.sample_rate_hertz
                for source in SOURCES
            )
            if active.write_failed:
                status = "failed"
            elif (
                error
                or received_sources < len(SOURCES)
                or has_timeline_gaps
                or has_stream_drops
                or has_capture_drops
                or missing_non_silent_audio
                or excessive_tail_padding
            ):
                status = "incomplete"
            else:
                status = "complete"
            active.manifest["status"] = status
            active.manifest["finishedAtMs"] = finished_at_ms
            active.manifest["durationMs"] = round(target_frames * 1000 / self.sample_rate_hertz)
            active.manifest["wallDurationMs"] = round(wall_frames * 1000 / self.sample_rate_hertz)
            if error:
                errors.append(clean_text(error))
            if errors:
                active.manifest["errors"] = errors

            total_size = 0
            for source in SOURCES:
                path = active.descriptor.track_path(source)
                size = path.stat().st_size if path.exists() else 0
                total_size += size
                active.manifest["tracks"][source].update(
                    {
                        "frames": active.frames[source],
                        "sizeBytes": size,
                        "sha256": file_sha256(path) if path.exists() else "",
                    }
                )
            active.manifest["sizeBytes"] = total_size
            self._append_event_locked(
                "recording.finished",
                status=active.manifest["status"],
                durationMs=active.manifest["durationMs"],
                sizeBytes=total_size,
            )
            try:
                active.events_handle.flush()
                active.events_handle.close()
            except Exception as events_error:
                active.write_failed = True
                active.manifest["status"] = "failed"
                active.manifest.setdefault("errors", []).append(f"events close: {events_error}")
            self._write_manifest(active.descriptor, active.manifest)
            result = self._descriptor(active.manifest, active.descriptor)
            self._active = None
            return result

    def active_recording_id(self) -> str | None:
        with self._lock:
            return self._active.descriptor.recording_id if self._active is not None else None

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        recordings: list[dict[str, Any]] = []
        for manifest_path in self.root.glob("*/manifest.json"):
            recording_id = manifest_path.parent.name
            item = self.get(recording_id)
            if item is not None:
                recordings.append(item)
        recordings.sort(key=lambda item: int(item.get("createdAtMs", 0)), reverse=True)
        return recordings

    def get(self, recording_id: str) -> dict[str, Any] | None:
        try:
            clean_id = validate_safe_id(recording_id)
        except (TypeError, ValueError):
            return None
        descriptor = CallRecordingDescriptor(clean_id, self.root / clean_id)
        try:
            manifest = json.loads(descriptor.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(manifest, dict) or manifest.get("id") != clean_id:
            return None
        return self._descriptor(manifest, descriptor)

    def delete(self, recording_id: str) -> bool:
        try:
            clean_id = validate_safe_id(recording_id)
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._active is not None and self._active.descriptor.recording_id == clean_id:
                return False
            path = self.root / clean_id
            if not path.is_dir():
                return False
            try:
                shutil.rmtree(path)
                return True
            except OSError:
                return False

    def track_path(self, recording_id: str, source: str) -> Path:
        clean_id = validate_safe_id(recording_id)
        return CallRecordingDescriptor(clean_id, self.root / clean_id).track_path(source)

    def verify(self, recording_id: str) -> dict[str, Any]:
        recording = self.get(recording_id)
        if recording is None:
            return {"ok": False, "tracks": {}, "errors": ["Запись не найдена"]}
        tracks = dict(recording.get("tracks") or {})
        checked: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        frame_counts: list[int] = []
        for source in SOURCES:
            track = tracks.get(source)
            if not isinstance(track, dict):
                errors.append(f"Нет описания дорожки {source}")
                continue
            path = self.track_path(recording_id, source)
            source_errors: list[str] = []
            actual_frames = 0
            if not path.is_file():
                source_errors.append("файл отсутствует")
            else:
                try:
                    with wave.open(str(path), "rb") as reader:
                        validate_wave(reader)
                        actual_frames = reader.getnframes()
                except Exception as error:
                    source_errors.append(str(error) or error.__class__.__name__)
            expected_frames = int(track.get("frames") or 0)
            if actual_frames and expected_frames and actual_frames != expected_frames:
                source_errors.append(
                    f"число отсчетов {actual_frames}, ожидалось {expected_frames}"
                )
            expected_hash = str(track.get("sha256") or "")
            actual_hash = ""
            if path.is_file():
                try:
                    actual_hash = file_sha256(path)
                except OSError as error:
                    source_errors.append(str(error))
            if expected_hash and actual_hash and actual_hash != expected_hash:
                source_errors.append("контрольная сумма не совпадает")
            if track.get("receivedAudio") is False:
                source_errors.append("звук не был получен")
            if actual_frames:
                frame_counts.append(actual_frames)
            checked[source] = {
                "ok": not source_errors,
                "path": str(path.resolve()),
                "frames": actual_frames,
                "sha256": actual_hash,
                "errors": source_errors,
            }
            errors.extend(f"{source}: {message}" for message in source_errors)
        if len(frame_counts) == len(SOURCES) and len(set(frame_counts)) != 1:
            errors.append("Дорожки имеют разную длительность")
        return {"ok": not errors, "tracks": checked, "errors": errors}

    def write_report(
        self,
        recording_id: str,
        report: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        recording = self.get(recording_id)
        if recording is None:
            raise FileNotFoundError(f"recording was not found: {recording_id}")
        clean_run_id = validate_safe_id(run_id or make_run_id())
        run_directory = self.root / recording_id / "runs" / clean_run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        clean_report = sanitize(dict(report))
        path = run_directory / "report.json"
        atomic_write_json(path, clean_report)
        return {
            "recordingId": recording_id,
            "runId": clean_run_id,
            "path": str(path.resolve()),
            "report": clean_report,
        }

    def _write_silence(self, active: _ActiveRecording, source: str, frames: int) -> None:
        remaining = max(0, int(frames))
        block_frames = self.sample_rate_hertz
        while remaining:
            count = min(remaining, block_frames)
            active.writers[source].writeframesraw(bytes(count * SAMPLE_WIDTH_BYTES))
            active.frames[source] += count
            remaining -= count

    def _append_event_locked(self, event: str, **payload: Any) -> bool:
        active = self._active
        if active is None:
            return False
        item = {
            "ts": int(time.time() * 1000),
            "event": clean_text(event, 200),
            **sanitize(payload),
        }
        try:
            active.events_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            active.events_handle.write("\n")
            if event != "audio.chunk":
                active.events_handle.flush()
            return True
        except Exception:
            return False

    def _write_manifest(self, descriptor: CallRecordingDescriptor, manifest: Mapping[str, Any]) -> None:
        atomic_write_json(descriptor.manifest_path, sanitize(dict(manifest)))

    def _descriptor(
        self,
        manifest: Mapping[str, Any],
        descriptor: CallRecordingDescriptor,
    ) -> dict[str, Any]:
        result = sanitize(dict(manifest))
        result["paths"] = {
            source: str(descriptor.track_path(source).resolve())
            for source in SOURCES
        }
        tracks = dict(result.get("tracks") or {})
        size_bytes = 0
        for source in SOURCES:
            track = dict(tracks.get(source) or {})
            path = descriptor.track_path(source)
            if path.exists():
                track["sizeBytes"] = path.stat().st_size
            size_bytes += int(track.get("sizeBytes") or 0)
            tracks[source] = track
        result["tracks"] = tracks
        result["sizeBytes"] = size_bytes
        result["manifestPath"] = str(descriptor.manifest_path.resolve())
        result["eventsPath"] = str(descriptor.events_path.resolve())
        return result


class ReplayClock:
    def __init__(
        self,
        speed: float = 1.0,
        *,
        participants: int = 1,
        start_timeout: float = 5.0,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if speed <= 0:
            raise ValueError("replay speed must be positive")
        if participants <= 0:
            raise ValueError("replay participants must be positive")
        if start_timeout <= 0:
            raise ValueError("replay start timeout must be positive")
        self.speed = float(speed)
        self.participants = int(participants)
        self.start_timeout = float(start_timeout)
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._condition = threading.Condition()
        self._started_at: float | None = None
        self._arrived = 0

    def start(self) -> float:
        with self._condition:
            if self._started_at is None:
                self._started_at = self._monotonic()
            return self._started_at

    def reset(self) -> None:
        with self._condition:
            self._started_at = None
            self._arrived = 0

    def wait_for_participants(self, stop_event: threading.Event | None = None) -> bool:
        deadline = time.monotonic() + self.start_timeout
        with self._condition:
            self._arrived += 1
            if self._arrived >= self.participants:
                if self._started_at is None:
                    self._started_at = self._monotonic()
                self._condition.notify_all()
                return True
            while self._started_at is None:
                if stop_event is not None and stop_event.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(remaining, 0.05))
            return True

    def wait_until(self, offset_seconds: float, stop_event: threading.Event | None = None) -> bool:
        if offset_seconds < 0:
            raise ValueError("replay offset cannot be negative")
        target = self.start() + offset_seconds / self.speed
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            remaining = target - self._monotonic()
            if remaining <= 0:
                return True
            self._sleeper(min(remaining, 0.05) if stop_event is not None else remaining)


class RecordedPcmSource:
    def __init__(
        self,
        wav_path: str | Path,
        replay_clock: ReplayClock,
        *,
        chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
    ) -> None:
        if chunk_duration_ms <= 0:
            raise ValueError("chunk duration must be positive")
        self.wav_path = Path(wav_path)
        self.replay_clock = replay_clock
        self.chunk_duration_ms = int(chunk_duration_ms)

    def chunks(self, stop_event: threading.Event) -> Iterator[bytes]:
        with wave.open(str(self.wav_path), "rb") as reader:
            validate_wave(reader)
            if not self.replay_clock.wait_for_participants(stop_event):
                raise CallRecordingError("recorded audio tracks did not start together")
            frames_per_chunk = max(1, round(SAMPLE_RATE_HERTZ * self.chunk_duration_ms / 1000))
            elapsed_frames = 0
            while not stop_event.is_set():
                pcm = reader.readframes(frames_per_chunk)
                if not pcm:
                    return
                elapsed_frames += len(pcm) // SAMPLE_WIDTH_BYTES
                if not self.replay_clock.wait_until(elapsed_frames / SAMPLE_RATE_HERTZ, stop_event):
                    return
                yield pcm


def validate_wave(reader: wave.Wave_read) -> None:
    if reader.getnchannels() != CHANNELS:
        raise CallRecordingError("recorded audio must be mono")
    if reader.getsampwidth() != SAMPLE_WIDTH_BYTES:
        raise CallRecordingError("recorded audio must use 16-bit PCM")
    if reader.getframerate() != SAMPLE_RATE_HERTZ:
        raise CallRecordingError("recorded audio must use 16000 Hz")
    if reader.getcomptype() != "NONE":
        raise CallRecordingError("recorded audio must be uncompressed PCM")


def normalize_source(source: str) -> str:
    value = str(source).strip().lower()
    if value not in SOURCES:
        raise ValueError("audio source must be remote or mic")
    return value


def validate_safe_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError("recording id contains unsupported characters")
    return value


def make_recording_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def make_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def clean_text(value: Any, limit: int = MAX_TEXT_LENGTH) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit}>"


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            clean[name] = "<redacted>" if any(secret in lowered for secret in _SECRET_KEYS) else sanitize(item)
        return clean
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return clean_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return clean_text(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
