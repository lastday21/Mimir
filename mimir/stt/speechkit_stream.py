from __future__ import annotations

import io
import time
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from ..models import SpeechRecognitionResult
from ..providers.yandex_speechkit import DEFAULT_LANGUAGE, DEFAULT_SAMPLE_RATE


@dataclass(frozen=True)
class AudioStreamConfig:
    language: str = DEFAULT_LANGUAGE
    sample_rate_hertz: int = DEFAULT_SAMPLE_RATE
    chunk_duration_ms: int = 400
    rotate_after_seconds: int = 270
    rotate_after_bytes: int = 9_000_000


@dataclass(frozen=True)
class TranscriptEvent:
    source: str
    text: str
    is_final: bool
    end_of_utterance: bool
    timestamp_ms: int
    confidence: float | None = None


class StreamingRecognizer(Protocol):
    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str,
        sample_rate_hertz: int,
    ) -> Iterator[SpeechRecognitionResult]:
        ...


class SpeechKitStreamRunner:
    def __init__(self, recognizer: StreamingRecognizer, config: AudioStreamConfig | None = None) -> None:
        self.recognizer = recognizer
        self.config = config or AudioStreamConfig()

    def run(self, source: str, chunks: Iterable[bytes]) -> Iterator[TranscriptEvent]:
        rotating = RotatingChunks(chunks, self.config)
        for session_chunks in rotating.sessions():
            for result in self.recognizer.stream_lpcm(
                session_chunks,
                language=self.config.language,
                sample_rate_hertz=self.config.sample_rate_hertz,
            ):
                yield TranscriptEvent(
                    source=source,
                    text=result.text,
                    is_final=result.is_final,
                    end_of_utterance=result.end_of_utterance,
                    timestamp_ms=int(time.time() * 1000),
                    confidence=result.confidence,
                )


class RotatingChunks:
    def __init__(self, chunks: Iterable[bytes], config: AudioStreamConfig) -> None:
        self.iterator = iter(chunks)
        self.config = config
        self.pending: bytes | None = None
        self.exhausted = False

    def sessions(self) -> Iterator[Iterator[bytes]]:
        while not self.exhausted or self.pending is not None:
            yield self.next_session()

    def next_session(self) -> Iterator[bytes]:
        sent_bytes = 0
        started = time.monotonic()

        while True:
            chunk = self.take_chunk()
            if chunk is None:
                return

            should_rotate = (
                sent_bytes > 0
                and (
                    sent_bytes + len(chunk) > self.config.rotate_after_bytes
                    or time.monotonic() - started >= self.config.rotate_after_seconds
                )
            )
            if should_rotate:
                self.pending = chunk
                return

            sent_bytes += len(chunk)
            yield chunk

    def take_chunk(self) -> bytes | None:
        if self.pending is not None:
            chunk = self.pending
            self.pending = None
            return chunk
        if self.exhausted:
            return None
        try:
            return next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return None


def chunk_pcm(
    pcm: bytes,
    *,
    sample_rate_hertz: int = DEFAULT_SAMPLE_RATE,
    chunk_duration_ms: int = 400,
) -> Iterator[bytes]:
    if chunk_duration_ms <= 0:
        raise ValueError("chunk_duration_ms must be positive")
    bytes_per_sample = 2
    chunk_size = max(bytes_per_sample, int(sample_rate_hertz * bytes_per_sample * chunk_duration_ms / 1000))
    chunk_size -= chunk_size % bytes_per_sample
    for index in range(0, len(pcm), chunk_size):
        chunk = pcm[index : index + chunk_size]
        if chunk:
            yield chunk


def pcm_chunks_from_wav(data: bytes, *, chunk_duration_ms: int = 400) -> tuple[int, Iterator[bytes]]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1:
            raise ValueError("Only mono WAV files are supported by the dev STT feeder")
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported by the dev STT feeder")
        pcm = wav.readframes(wav.getnframes())
        if not pcm:
            raise ValueError("WAV file contains no audio frames")
    return sample_rate, chunk_pcm(pcm, sample_rate_hertz=sample_rate, chunk_duration_ms=chunk_duration_ms)
