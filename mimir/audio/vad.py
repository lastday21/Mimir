from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyVadConfig:
    speech_rms_threshold: int = 350
    silence_rms_threshold: int = 180
    tail_silence_ms: int = 700
    min_speech_ms: int = 120


@dataclass(frozen=True)
class VadDecision:
    is_speech: bool
    send_to_stt: bool
    rms: int
    speech_started: bool = False
    speech_ended: bool = False


class EnergyVadGate:
    def __init__(self, sample_rate_hertz: int, config: EnergyVadConfig | None = None) -> None:
        self.sample_rate_hertz = sample_rate_hertz
        self.config = config or EnergyVadConfig()
        self.in_speech = False
        self.speech_ms = 0
        self.silence_ms = 0

    def process(self, pcm: bytes) -> VadDecision:
        if not pcm:
            return VadDecision(is_speech=False, send_to_stt=False, rms=0)

        rms = pcm_rms(pcm)
        duration_ms = pcm_duration_ms(pcm, self.sample_rate_hertz)
        is_speech = rms >= self.config.speech_rms_threshold
        is_continuation = self.in_speech and rms >= self.config.silence_rms_threshold

        if is_speech or is_continuation:
            started = not self.in_speech
            self.in_speech = True
            self.speech_ms += duration_ms
            self.silence_ms = 0
            send = self.speech_ms >= self.config.min_speech_ms
            return VadDecision(
                is_speech=True,
                send_to_stt=send,
                rms=rms,
                speech_started=started,
            )

        if self.in_speech:
            self.silence_ms += duration_ms
            ended = self.silence_ms > self.config.tail_silence_ms
            if ended:
                self.in_speech = False
                self.speech_ms = 0
                self.silence_ms = 0
                return VadDecision(
                    is_speech=False,
                    send_to_stt=False,
                    rms=rms,
                    speech_ended=True,
                )
            return VadDecision(is_speech=False, send_to_stt=True, rms=rms)

        return VadDecision(is_speech=False, send_to_stt=False, rms=rms)


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
