import time
import unittest
from collections.abc import Iterable, Iterator

from mimir.audio.capture import AudioCaptureConfig, float_frames_to_pcm16
from mimir.audio.live import LiveAudioConfig, LiveAudioController, normalize_sources
from mimir.audio.vad import EnergyVadConfig, EnergyVadGate
from mimir.models import SpeechRecognitionResult


def pcm_constant(value: int, frames: int) -> bytes:
    sample = int(value).to_bytes(2, "little", signed=True)
    return sample * frames


class FakeSession:
    def __init__(self) -> None:
        self.started = False
        self.transcripts: list[tuple[str, str, bool]] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def start(self) -> dict[str, object]:
        self.started = True
        return {"state": "listening"}

    def ingest_transcript(self, source: str, text: str, is_final: bool = True) -> dict[str, object]:
        self.transcripts.append((source, text, is_final))
        return {"source": source, "text": text, "isFinal": is_final}

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))


class FakePcmSource:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def chunks(self, _stop_event) -> Iterator[bytes]:
        yield from self._chunks


class FakeRecognizer:
    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str,
        sample_rate_hertz: int,
    ) -> Iterator[SpeechRecognitionResult]:
        consumed = list(chunks)
        if consumed:
            yield SpeechRecognitionResult(
                text=f"{language}:{sample_rate_hertz}:{len(consumed)}",
                is_final=True,
                end_of_utterance=True,
            )


class AudioPipelineTests(unittest.TestCase):
    def test_normalizes_audio_sources(self) -> None:
        self.assertEqual(normalize_sources(["system", "me", "remote"]), ("remote", "mic"))

    def test_converts_float_frames_to_mono_pcm16(self) -> None:
        pcm = float_frames_to_pcm16([[0.5, 0.5], [-1.0, -1.0], [0.0, 0.0]])

        self.assertEqual(len(pcm), 6)
        self.assertEqual(int.from_bytes(pcm[:2], "little", signed=True), 16384)
        self.assertEqual(int.from_bytes(pcm[2:4], "little", signed=True), -32767)

    def test_vad_keeps_speech_and_short_tail_silence(self) -> None:
        gate = EnergyVadGate(
            16_000,
            EnergyVadConfig(
                speech_rms_threshold=100,
                silence_rms_threshold=50,
                tail_silence_ms=100,
                min_speech_ms=1,
            ),
        )

        speech = gate.process(pcm_constant(1000, 1600))
        tail = gate.process(pcm_constant(0, 800))
        ended = gate.process(pcm_constant(0, 3200))

        self.assertTrue(speech.speech_started)
        self.assertTrue(speech.send_to_stt)
        self.assertTrue(tail.send_to_stt)
        self.assertTrue(ended.speech_ended)
        self.assertFalse(ended.send_to_stt)

    def test_live_controller_streams_vad_chunks_into_session(self) -> None:
        session = FakeSession()
        source_chunks = [pcm_constant(1000, 3200), pcm_constant(0, 1600)]
        controller = LiveAudioController(
            session,
            lambda _key: FakeRecognizer(),
            lambda _source, _config: FakePcmSource(source_chunks),
        )

        controller.start(
            LiveAudioConfig(
                sources=("remote",),
                chunk_duration_ms=200,
                vad=EnergyVadConfig(
                    speech_rms_threshold=100,
                    silence_rms_threshold=50,
                    tail_silence_ms=500,
                    min_speech_ms=1,
                ),
            ),
            "test-key",
        )

        deadline = time.monotonic() + 2
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(controller.snapshot()["running"])
        self.assertTrue(session.started)
        self.assertEqual(session.transcripts, [("remote", "ru-RU:16000:2", True)])
        self.assertTrue(any(event == "audio_status" for event, _payload in session.events))


if __name__ == "__main__":
    unittest.main()
