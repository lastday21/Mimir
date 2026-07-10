import time
import threading
import unittest
from collections.abc import AsyncIterator
from collections.abc import Iterable, Iterator

from mimir.audio.capture import AudioCaptureConfig, float_frames_to_pcm16, select_loopback, select_microphone
from mimir.audio.live import LiveAudioConfig, LiveAudioController, normalize_sources
from mimir.audio.realtime import REALTIME_INSTRUCTIONS, RealtimeAudioConfig, RealtimeAudioController
from mimir.audio.vad import EnergyVadConfig, EnergyVadGate
from mimir.dialogue import DialogueMemory, DialogueTurn
from mimir.models import SpeechRecognitionResult


def pcm_constant(value: int, frames: int) -> bytes:
    sample = int(value).to_bytes(2, "little", signed=True)
    return sample * frames


class FakeSession:
    def __init__(self) -> None:
        self.started = False
        self.transcripts: list[tuple[str, str, bool]] = []
        self.events: list[tuple[str, dict[str, object]]] = []
        self.metric_events: list[tuple[str, str, int | bool | None]] = []
        self.memory = DialogueMemory()

    def start(self) -> dict[str, object]:
        self.started = True
        return {"state": "listening"}

    def ingest_transcript(
        self,
        source: str,
        text: str,
        is_final: bool = True,
        detect_question: bool = True,
    ) -> dict[str, object]:
        self.transcripts.append((source, text, is_final))
        self.memory.append(DialogueTurn(source, text, is_final=is_final))
        return {"source": source, "text": text, "isFinal": is_final}

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))

    def record_audio_speech_started(self, source: str) -> None:
        self.metric_events.append(("speech_started", source, None))

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        self.metric_events.append(("audio_chunk", source, byte_count))

    def record_stt_result(self, source: str, is_final: bool) -> None:
        self.metric_events.append(("stt", source, is_final))

    def realtime_context(self, max_turns: int = 12, max_chars: int = 1800) -> str:
        return self.memory.realtime_context(max_turns=max_turns, max_chars=max_chars)


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

    def test_realtime_prompt_keeps_context_contract_without_language_rule(self) -> None:
        self.assertIn("DIALOGUE_CONTEXT", REALTIME_INSTRUCTIONS)
        self.assertIn("не участник созвона", REALTIME_INSTRUCTIONS)
        self.assertIn("Не выдумывай опыт", REALTIME_INSTRUCTIONS)
        self.assertNotIn("на языке вопроса", REALTIME_INSTRUCTIONS)
        self.assertNotIn("пиши по-русски", REALTIME_INSTRUCTIONS)

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
        self.assertTrue(any(event[:2] == ("audio_chunk", "remote") for event in session.metric_events))
        self.assertIn(("stt", "remote", True), session.metric_events)
        self.assertTrue(any(event == "audio_status" for event, _payload in session.events))

    def test_realtime_controller_sends_remote_audio_and_dialogue_context(self) -> None:
        session = FakeSession()
        appended_audio: list[bytes] = []
        dialogue_contexts: list[str] = []
        setup_calls: list[tuple[str, int]] = []

        class FakeRealtimeClient:
            def __init__(self, _config) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                pass

            async def setup_session(self, instructions: str, sample_rate_hertz: int) -> None:
                setup_calls.append((instructions, sample_rate_hertz))

            async def append_audio(self, pcm: bytes) -> None:
                appended_audio.append(pcm)

            async def add_dialogue_context(self, text: str) -> None:
                dialogue_contexts.append(text)

            async def events(self) -> AsyncIterator[dict[str, object]]:
                yield {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Как вы проектировали очередь задач?",
                }
                yield {"type": "response.output_text.delta", "delta": "Начни с требований и ограничений."}
                yield {"type": "response.output_text.done"}

        def source_factory(source: str, _config: AudioCaptureConfig) -> FakePcmSource:
            chunks = [pcm_constant(1000, 3200), pcm_constant(0, 1600)]
            return FakePcmSource(chunks)

        controller = RealtimeAudioController(
            session,
            lambda _key: FakeRecognizer(),
            FakeRealtimeClient,
            source_factory,
        )

        controller.start(
            RealtimeAudioConfig(
                sources=("remote", "mic"),
                chunk_duration_ms=200,
                vad=EnergyVadConfig(
                    speech_rms_threshold=100,
                    silence_rms_threshold=50,
                    tail_silence_ms=500,
                    min_speech_ms=1,
                ),
            ),
            "test-key",
            "folder-id",
        )

        deadline = time.monotonic() + 2
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(controller.snapshot()["running"])
        self.assertTrue(session.started)
        self.assertEqual(setup_calls[0][1], 16_000)
        self.assertIn("DIALOGUE_CONTEXT", setup_calls[0][0])
        self.assertNotIn("MIC_CONTEXT", setup_calls[0][0])
        self.assertTrue(appended_audio)
        self.assertTrue(dialogue_contexts)
        self.assertIn("Собеседник: Как вы проектировали очередь задач?", "\n".join(dialogue_contexts))
        self.assertIn("Пользователь: ru-RU:16000:2", "\n".join(dialogue_contexts))
        self.assertIn(("remote", "Как вы проектировали очередь задач?", True), session.transcripts)
        self.assertIn(("mic", "ru-RU:16000:2", True), session.transcripts)
        self.assertTrue(any(event[:2] == ("audio_chunk", "remote") for event in session.metric_events))
        self.assertIn(("stt", "remote", True), session.metric_events)
        self.assertIn(("stt", "mic", True), session.metric_events)
        self.assertTrue(any(event == "answer_delta" for event, _payload in session.events))

    def test_realtime_controller_reconnects_after_receive_error(self) -> None:
        session = FakeSession()
        connections: list[int] = []

        class FlakyRealtimeClient:
            def __init__(self, _config) -> None:
                self.index = len(connections) + 1
                connections.append(self.index)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                pass

            async def setup_session(self, instructions: str, sample_rate_hertz: int) -> None:
                pass

            async def append_audio(self, pcm: bytes) -> None:
                pass

            async def add_dialogue_context(self, text: str) -> None:
                pass

            async def events(self) -> AsyncIterator[dict[str, object]]:
                if self.index == 1:
                    raise RuntimeError("socket dropped")
                yield {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Почему нужен reconnect?",
                }
                yield {"type": "response.output_text.delta", "delta": "Потому что сеть может оборваться."}
                yield {"type": "response.output_text.done"}

        controller = RealtimeAudioController(
            session,
            lambda _key: FakeRecognizer(),
            FlakyRealtimeClient,
            lambda _source, _config: FakePcmSource([pcm_constant(1000, 3200), pcm_constant(0, 1600)]),
        )

        controller.start(
            RealtimeAudioConfig(
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
            "folder-id",
        )

        deadline = time.monotonic() + 4
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(controller.snapshot()["running"])
        self.assertGreaterEqual(len(connections), 2)
        self.assertIn(("remote", "Почему нужен reconnect?", True), session.transcripts)
        self.assertTrue(any(event == "answer_delta" for event, _payload in session.events))
        self.assertTrue(
            any(
                event == "audio_error" and payload.get("phase") == "reconnect"
                for event, payload in session.events
            )
        )

    def test_realtime_controller_falls_back_after_server_error_event(self) -> None:
        session = FakeSession()
        fallback_called = threading.Event()
        fallback_calls: list[tuple[tuple[str, ...], str, bool]] = []

        class ErrorRealtimeClient:
            def __init__(self, _config) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                pass

            async def setup_session(self, instructions: str, sample_rate_hertz: int) -> None:
                pass

            async def append_audio(self, pcm: bytes) -> None:
                pass

            async def add_dialogue_context(self, text: str) -> None:
                pass

            async def events(self) -> AsyncIterator[dict[str, object]]:
                yield {"type": "error", "error": {"message": "quota exhausted"}}

        def fallback_starter(config: RealtimeAudioConfig, reason: str) -> None:
            fallback_calls.append((config.sources, reason, config.vad_enabled))
            fallback_called.set()

        controller = RealtimeAudioController(
            session,
            lambda _key: FakeRecognizer(),
            ErrorRealtimeClient,
            lambda _source, _config: FakePcmSource([pcm_constant(1000, 3200), pcm_constant(0, 1600)]),
            fallback_starter,
        )

        controller.start(
            RealtimeAudioConfig(
                sources=("remote",),
                chunk_duration_ms=200,
                vad_enabled=True,
                vad=EnergyVadConfig(
                    speech_rms_threshold=100,
                    silence_rms_threshold=50,
                    tail_silence_ms=500,
                    min_speech_ms=1,
                ),
            ),
            "test-key",
            "folder-id",
        )

        self.assertTrue(fallback_called.wait(2))
        deadline = time.monotonic() + 2
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(controller.snapshot()["running"])
        self.assertEqual(fallback_calls, [(("remote",), "quota exhausted", True)])
        self.assertTrue(
            any(
                event == "audio_error" and payload.get("phase") == "server_event"
                for event, payload in session.events
            )
        )


if __name__ == "__main__":
    unittest.main()
