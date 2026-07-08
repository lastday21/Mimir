import time
import unittest
from collections.abc import Iterable, Iterator

from mimir.audio.capture import AudioCaptureConfig, float_frames_to_pcm16, select_loopback, select_microphone
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


class FakeAudioDevice:
    def __init__(self, identifier: str, name: str, loopback: bool = False) -> None:
        self.id = identifier
        self.name = name
        self.isloopback = loopback


class FakeSoundcard:
    def __init__(
        self,
        microphones: list[FakeAudioDevice],
        loopbacks: list[FakeAudioDevice] | None = None,
        default_mic: FakeAudioDevice | None = None,
        default_speaker: FakeAudioDevice | None = None,
    ) -> None:
        self.microphones = microphones
        self.loopbacks = loopbacks or []
        self._default_mic = default_mic
        self._default_speaker = default_speaker

    def all_microphones(self, include_loopback: bool = False) -> list[FakeAudioDevice]:
        if include_loopback:
            return [*self.microphones, *self.loopbacks]
        return self.microphones

    def default_microphone(self) -> FakeAudioDevice | None:
        return self._default_mic

    def default_speaker(self) -> FakeAudioDevice | None:
        return self._default_speaker


class AudioPipelineTests(unittest.TestCase):
    def test_normalizes_audio_sources(self) -> None:
        self.assertEqual(normalize_sources(["system", "me", "remote"]), ("remote", "mic"))

    def test_converts_float_frames_to_mono_pcm16(self) -> None:
        pcm = float_frames_to_pcm16([[0.5, 0.5], [-1.0, -1.0], [0.0, 0.0]])

        self.assertEqual(len(pcm), 6)
        self.assertEqual(int.from_bytes(pcm[:2], "little", signed=True), 16384)
        self.assertEqual(int.from_bytes(pcm[2:4], "little", signed=True), -32767)

    def test_prefers_headset_microphone_over_default_builtin(self) -> None:
        builtin = FakeAudioDevice("mic_builtin", "Microphone Array (Realtek Audio)")
        headset = FakeAudioDevice("mic_headset", "Headset Microphone (Jabra)")
        sc = FakeSoundcard([builtin, headset], default_mic=builtin)

        self.assertIs(select_microphone(sc), headset)

    def test_falls_back_to_builtin_microphone_without_headset(self) -> None:
        webcam = FakeAudioDevice("mic_webcam", "HD Webcam Mic")
        builtin = FakeAudioDevice("mic_builtin", "Internal Microphone Array")
        sc = FakeSoundcard([webcam, builtin], default_mic=webcam)

        self.assertIs(select_microphone(sc), builtin)

    def test_selects_loopback_for_default_speaker_output(self) -> None:
        speaker = FakeAudioDevice("speaker_sony", "Headphones (Sony)")
        speakers_loopback = FakeAudioDevice("loop_sony", "Headphones (Sony) Loopback", loopback=True)
        monitor_loopback = FakeAudioDevice("loop_monitor", "Monitor Speakers Loopback", loopback=True)
        sc = FakeSoundcard([], [monitor_loopback, speakers_loopback], default_speaker=speaker)

        self.assertIs(select_loopback(sc), speakers_loopback)

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
