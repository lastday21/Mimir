from __future__ import annotations

import math
import sys
from array import array
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyVadConfig:
    speech_rms_threshold: int = 350
    silence_rms_threshold: int = 180
    tail_silence_ms: int = 2_000
    min_speech_ms: int = 120
    pre_roll_ms: int = 600
    adaptive_thresholds: bool = True
    adaptive_speech_floor: int = 180
    adaptive_silence_floor: int = 100
    noise_window_ms: int = 5_000
    speech_noise_multiplier: float = 3.0
    silence_noise_multiplier: float = 1.8


@dataclass(frozen=True)
class VadDecision:
    is_speech: bool
    send_to_stt: bool
    rms: int
    speech_started: bool = False
    speech_ended: bool = False
    audio_chunks: tuple[bytes, ...] = ()
    speech_threshold: int = 0
    silence_threshold: int = 0


class EnergyVadGate:
    def __init__(self, sample_rate_hertz: int, config: EnergyVadConfig | None = None) -> None:
        self.sample_rate_hertz = sample_rate_hertz
        self.config = config or EnergyVadConfig()
        self.in_speech = False
        self.silence_ms = 0
        self._candidate_ms = 0
        self._candidate_chunks: list[bytes] = []
        self._pre_roll: deque[tuple[bytes, int]] = deque()
        self._pre_roll_duration_ms = 0
        self._noise_samples: deque[tuple[int, int]] = deque()
        self._noise_duration_ms = 0

    def process(self, pcm: bytes) -> VadDecision:
        if not pcm:
            return VadDecision(is_speech=False, send_to_stt=False, rms=0)

        rms = pcm_rms(pcm)
        duration_ms = pcm_duration_ms(pcm, self.sample_rate_hertz)
        speech_threshold, silence_threshold = self._thresholds()

        if self.in_speech:
            if rms >= silence_threshold:
                self.silence_ms = 0
                return self._decision(
                    rms,
                    speech_threshold,
                    silence_threshold,
                    is_speech=True,
                    audio_chunks=(pcm,),
                )

            self.silence_ms += duration_ms
            ended = self.silence_ms >= max(0, self.config.tail_silence_ms)
            decision = self._decision(
                rms,
                speech_threshold,
                silence_threshold,
                is_speech=False,
                speech_ended=ended,
                audio_chunks=(pcm,),
            )
            if ended:
                self.in_speech = False
                self.silence_ms = 0
                self._remember_noise(rms, duration_ms)
            return decision

        candidate_continues = bool(self._candidate_chunks) and rms >= silence_threshold
        if rms >= speech_threshold or candidate_continues:
            self._candidate_chunks.append(pcm)
            self._candidate_ms += duration_ms
            if self._candidate_ms < max(1, self.config.min_speech_ms):
                return self._decision(
                    rms,
                    speech_threshold,
                    silence_threshold,
                    is_speech=True,
                )

            buffered = tuple(chunk for chunk, _duration in self._pre_roll)
            audio_chunks = (*buffered, *self._candidate_chunks)
            self._pre_roll.clear()
            self._pre_roll_duration_ms = 0
            self._candidate_chunks = []
            self._candidate_ms = 0
            self.in_speech = True
            self.silence_ms = 0
            return self._decision(
                rms,
                speech_threshold,
                silence_threshold,
                is_speech=True,
                speech_started=True,
                audio_chunks=audio_chunks,
            )

        self._return_candidate_to_pre_roll(duration_ms)
        self._remember_pre_roll(pcm, duration_ms)
        self._remember_noise(rms, duration_ms)
        return self._decision(
            rms,
            speech_threshold,
            silence_threshold,
            is_speech=False,
        )

    def _decision(
        self,
        rms: int,
        speech_threshold: int,
        silence_threshold: int,
        *,
        is_speech: bool,
        speech_started: bool = False,
        speech_ended: bool = False,
        audio_chunks: tuple[bytes, ...] = (),
    ) -> VadDecision:
        return VadDecision(
            is_speech=is_speech,
            send_to_stt=bool(audio_chunks),
            rms=rms,
            speech_started=speech_started,
            speech_ended=speech_ended,
            audio_chunks=audio_chunks,
            speech_threshold=speech_threshold,
            silence_threshold=silence_threshold,
        )

    def _thresholds(self) -> tuple[int, int]:
        if not self.config.adaptive_thresholds:
            return (
                max(1, self.config.speech_rms_threshold),
                max(1, self.config.silence_rms_threshold),
            )

        noise_floor = self._noise_floor()
        speech = max(
            1,
            self.config.adaptive_speech_floor,
            round(noise_floor * self.config.speech_noise_multiplier),
        )
        silence = max(
            1,
            self.config.adaptive_silence_floor,
            round(noise_floor * self.config.silence_noise_multiplier),
        )
        return speech, min(speech, silence)

    def _noise_floor(self) -> int:
        if not self._noise_samples:
            return max(1, round(self.config.adaptive_speech_floor / max(1.0, self.config.speech_noise_multiplier)))
        values = sorted(rms for rms, _duration in self._noise_samples)
        index = max(0, round((len(values) - 1) * 0.2))
        return max(1, values[index])

    def _remember_noise(self, rms: int, duration_ms: int) -> None:
        self._noise_samples.append((rms, duration_ms))
        self._noise_duration_ms += duration_ms
        limit = max(1, self.config.noise_window_ms)
        while self._noise_duration_ms > limit and self._noise_samples:
            _old_rms, old_duration = self._noise_samples.popleft()
            self._noise_duration_ms -= old_duration

    def _remember_pre_roll(self, pcm: bytes, duration_ms: int) -> None:
        limit = max(0, self.config.pre_roll_ms)
        if limit <= 0:
            self._pre_roll.clear()
            self._pre_roll_duration_ms = 0
            return
        self._pre_roll.append((pcm, duration_ms))
        self._pre_roll_duration_ms += duration_ms
        while self._pre_roll_duration_ms > limit and self._pre_roll:
            _old_chunk, old_duration = self._pre_roll.popleft()
            self._pre_roll_duration_ms -= old_duration

    def _return_candidate_to_pre_roll(self, fallback_duration_ms: int) -> None:
        if not self._candidate_chunks:
            return
        count = len(self._candidate_chunks)
        duration_ms = max(1, round(self._candidate_ms / count)) if self._candidate_ms else fallback_duration_ms
        for chunk in self._candidate_chunks:
            self._remember_pre_roll(chunk, duration_ms)
        self._candidate_chunks = []
        self._candidate_ms = 0


def pcm_duration_ms(pcm: bytes, sample_rate_hertz: int) -> int:
    if sample_rate_hertz <= 0:
        raise ValueError("sample_rate_hertz must be positive")
    frames = len(pcm) // 2
    return max(1, round(frames * 1000 / sample_rate_hertz))


def pcm_rms(pcm: bytes) -> int:
    even = len(pcm) - (len(pcm) % 2)
    samples = array("h")
    samples.frombytes(pcm[:even])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0
    total = sum(sample * sample for sample in samples)
    return round(math.sqrt(total / len(samples)))
