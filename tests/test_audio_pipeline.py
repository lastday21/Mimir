import asyncio
import os
import struct
import time
import threading
import unittest
from collections.abc import AsyncIterator
from collections.abc import Iterable, Iterator
from unittest.mock import patch

from mimir.audio.capture import (
    AudioCaptureConfig,
    AudioCaptureError,
    float_frames_to_pcm16,
    list_audio_devices,
    select_loopback,
    select_microphone,
)
from mimir.audio.applications import ProcessLoopbackPcmSource, _FloatStereoConverter, process_exists
from mimir.audio.live import (
    LiveAudioConfig,
    LiveAudioController,
    _BufferedPcmCapture,
    apply_pcm16_gain,
    default_source_factory,
    normalize_sources,
)
from mimir.audio.preflight import add_device_checks, parse_live_audio_request
from mimir.audio.realtime import REALTIME_INSTRUCTIONS, RealtimeAudioConfig, RealtimeAudioController
from mimir.audio.runtime import (
    AudioRuntimeDependencies,
    copy_live_audio_config,
    start_cloud_audio_fallback_locked,
    start_local_audio_fallback_locked,
)
from mimir.audio.vad import EnergyVadConfig, EnergyVadGate
from mimir.dialogue import DialogueMemory, DialogueTurn
from mimir.models import SpeechRecognitionResult


def pcm_constant(value: int, frames: int) -> bytes:
    sample = int(value).to_bytes(2, "little", signed=True)
    return sample * frames


class AudioConfigCopyTests(unittest.TestCase):
    def test_live_fallback_keeps_custom_microphone_gain(self) -> None:
        copied = copy_live_audio_config(LiveAudioConfig(mic_gain=1.5))

        self.assertEqual(copied.mic_gain, 1.5)

    def test_realtime_fallback_uses_safe_default_microphone_gain(self) -> None:
        copied = copy_live_audio_config(RealtimeAudioConfig())

        self.assertEqual(copied.mic_gain, 2.0)

    def test_automatic_fallback_waits_for_old_audio_threads(self) -> None:
        class FakeController:
            def __init__(self, *, lingering: bool = False) -> None:
                self.lingering = lingering
                self.start_calls = 0

            def stop(self) -> None:
                pass

            def has_live_threads(self) -> bool:
                return self.lingering

            def start(self, *_args) -> dict[str, object]:
                self.start_calls += 1
                return {"running": True}

        class FakeSessionManager:
            def set_answer_provider_override(self, _provider) -> None:
                pass

            def publish_status(self, _event, _payload) -> None:
                pass

        for fallback in ("cloud", "local"):
            with self.subTest(fallback=fallback):
                realtime = FakeController(lingering=True)
                live = FakeController()
                local = FakeController()
                dependencies = AudioRuntimeDependencies(
                    session_manager=FakeSessionManager(),
                    live_audio=live,
                    local_audio=local,
                    realtime_audio=realtime,
                    load_config=lambda: None,
                    read_secret=lambda _name: "test-key",
                )

                with self.assertRaisesRegex(ValueError, "еще завершается"):
                    if fallback == "cloud":
                        start_cloud_audio_fallback_locked(
                            RealtimeAudioConfig(),
                            "ошибка основного режима",
                            dependencies,
                        )
                    else:
                        start_local_audio_fallback_locked(
                            LiveAudioConfig(),
                            "ошибка облачного режима",
                            dependencies,
                        )

                self.assertEqual(live.start_calls, 0)
                self.assertEqual(local.start_calls, 0)


class FakeSession:
    def __init__(self) -> None:
        self.started = False
        self.transcripts: list[tuple[str, str, bool]] = []
        self.events: list[tuple[str, dict[str, object]]] = []
        self.metric_events: list[tuple[str, str, int | bool | None]] = []
        self.stt_refinements: list[tuple[str, bool, bool]] = []
        self.external_questions: list[tuple[str, str, int]] = []
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
        is_refinement: bool = False,
    ) -> dict[str, object]:
        self.transcripts.append((source, text, is_final))
        self.memory.append(DialogueTurn(source, text, is_final=is_final), refine_latest=is_refinement)
        return {"source": source, "text": text, "isFinal": is_final}

    def publish_status(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))

    def record_audio_speech_started(self, source: str) -> None:
        self.metric_events.append(("speech_started", source, None))

    def record_audio_chunk(self, source: str, byte_count: int) -> None:
        self.metric_events.append(("audio_chunk", source, byte_count))

    def record_stt_result(
        self,
        source: str,
        is_final: bool,
        *,
        is_refinement: bool = False,
    ) -> int:
        self.metric_events.append(("stt", source, is_final))
        self.stt_refinements.append((source, is_final, is_refinement))
        return 1 if is_final and not is_refinement else 0

    def record_external_question(
        self,
        question_id: str,
        question: str,
        *,
        utterance_sequence: int = 0,
        **_payload,
    ) -> None:
        self.external_questions.append((question_id, question, utterance_sequence))

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


class FakeRecordingStore:
    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []
        self.writes: list[tuple[str, bytes, int]] = []
        self.finishes = 0
        self.recording_id = ""

    def start(self, **payload) -> dict[str, str]:
        self.starts.append(payload)
        self.recording_id = "recording_test"
        return {"recordingId": self.recording_id}

    def write(
        self,
        source: str,
        chunk: bytes,
        captured_at_ns: int,
        recording_id: str = "",
    ) -> bool:
        del recording_id
        self.writes.append((source, chunk, captured_at_ns))
        return True

    def record_event(self, _event: str, _payload: dict[str, object]) -> None:
        pass

    def finish(
        self,
        *,
        error: str = "",
        finished_monotonic_ns: int | None = None,
    ) -> dict[str, str]:
        self.finishes += 1
        self.recording_id = ""
        return {"recordingId": "recording_test", "error": error}

    def active_recording_id(self) -> str:
        return self.recording_id


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
    def test_buffered_capture_keeps_reading_and_marks_oldest_drops(self) -> None:
        source_chunks = [pcm_constant(index, 1600) for index in range(1, 7)]
        captured: list[bytes] = []
        dropped: list[tuple[int, int]] = []
        consumed: list[bytes] = []
        first_consumed = threading.Event()
        release_consumer = threading.Event()

        class TrackingSource(FakePcmSource):
            def chunks(self, _stop_event) -> Iterator[bytes]:
                for chunk in self._chunks:
                    captured.append(chunk)
                    yield chunk

        capture = _BufferedPcmCapture(
            TrackingSource(source_chunks),
            threading.Event(),
            capacity=2,
            on_drop=lambda frames, captured_at_ns: dropped.append((frames, captured_at_ns)),
        )

        def consume() -> None:
            for item in capture.chunks():
                consumed.append(item.pcm)
                if len(consumed) == 1:
                    first_consumed.set()
                    release_consumer.wait(1)

        consumer = threading.Thread(target=consume)
        consumer.start()
        self.assertTrue(first_consumed.wait(1))
        deadline = time.monotonic() + 1
        while len(captured) < len(source_chunks) and time.monotonic() < deadline:
            time.sleep(0.005)
        release_consumer.set()
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertEqual(captured, source_chunks)
        self.assertTrue(dropped)
        dropped_chunks = sum(frames for frames, _ in dropped) // 1600
        self.assertEqual(dropped_chunks + len(consumed), len(source_chunks))
        self.assertTrue(all(captured_at_ns > 0 for _, captured_at_ns in dropped))
        self.assertEqual(consumed[-1], source_chunks[-1])

    def test_buffered_capture_marks_recording_queue_overflow(self) -> None:
        source_chunks = [pcm_constant(index, 1600) for index in range(1, 7)]
        source_finished = threading.Event()
        recording_started = threading.Event()
        release_recording = threading.Event()
        recorded: list[bytes] = []
        recording_drops: list[tuple[int, int]] = []

        class TrackingSource(FakePcmSource):
            def chunks(self, _stop_event) -> Iterator[bytes]:
                for index, chunk in enumerate(self._chunks):
                    yield chunk
                    if index == 0:
                        recording_started.wait(1)
                source_finished.set()

        def record(item) -> None:
            recorded.append(item.pcm)
            if len(recorded) == 1:
                recording_started.set()
                release_recording.wait(1)

        capture = _BufferedPcmCapture(
            TrackingSource(source_chunks),
            threading.Event(),
            capacity=2,
            on_capture=record,
            on_capture_drop=lambda frames, captured_at_ns: recording_drops.append(
                (frames, captured_at_ns)
            ),
        )
        consumer = threading.Thread(target=lambda: list(capture.chunks()))
        consumer.start()

        self.assertTrue(recording_started.wait(1))
        self.assertTrue(source_finished.wait(1))
        release_recording.set()
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertEqual(recorded[0], source_chunks[0])
        self.assertEqual(recorded[-2:], source_chunks[-2:])
        self.assertEqual(sum(frames for frames, _ in recording_drops), 3 * 1600)
        self.assertTrue(all(captured_at_ns > 0 for _, captured_at_ns in recording_drops))

    def test_buffered_capture_waits_for_recording_worker_before_finishing(self) -> None:
        recording_started = threading.Event()
        release_recording = threading.Event()

        def blocked_recording(_item) -> None:
            recording_started.set()
            release_recording.wait(1)

        capture = _BufferedPcmCapture(
            FakePcmSource([pcm_constant(1000, 1600)]),
            threading.Event(),
            capacity=2,
            on_capture=blocked_recording,
        )
        consumer = threading.Thread(target=lambda: list(capture.chunks()))

        consumer.start()
        self.assertTrue(recording_started.wait(1))
        consumer.join(timeout=0.1)
        self.assertTrue(consumer.is_alive())
        release_recording.set()
        consumer.join(timeout=1)
        self.assertFalse(consumer.is_alive())

    def test_pcm_gain_amplifies_and_saturates(self) -> None:
        raw = b"".join(
            value.to_bytes(2, "little", signed=True)
            for value in (-20_000, -1_000, 1_000, 20_000)
        )

        amplified = apply_pcm16_gain(raw, 2.0)
        values = [
            int.from_bytes(amplified[index : index + 2], "little", signed=True)
            for index in range(0, len(amplified), 2)
        ]

        self.assertEqual(values, [-32_768, -2_000, 2_000, 32_767])

    def test_live_controller_starts_session_before_recording_with_revision(self) -> None:
        order: list[str] = []

        class OrderedSession(FakeSession):
            def start(self) -> dict[str, object]:
                order.append("session")
                return super().start()

        class OrderedStore(FakeRecordingStore):
            def start(self, **payload) -> dict[str, str]:
                order.append("recording")
                return super().start(**payload)

        session = OrderedSession()
        store = OrderedStore()
        controller = LiveAudioController(
            session,
            lambda _key: FakeRecognizer(),
            lambda _source, _config: FakePcmSource([pcm_constant(1000, 1600)]),
            recording_store=store,
        )

        controller.start(
            LiveAudioConfig(sources=("remote",), record_testing=True, vad_enabled=False),
            "test-key",
        )
        deadline = time.monotonic() + 1
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(order[:2], ["session", "recording"])
        self.assertTrue(store.starts[0]["app_revision"])

    def test_slow_recording_store_does_not_block_audio_source(self) -> None:
        source_finished_at: list[float] = []

        class TimedSource(FakePcmSource):
            def chunks(self, _stop_event) -> Iterator[bytes]:
                yield from self._chunks
                source_finished_at.append(time.monotonic())

        class SlowStore(FakeRecordingStore):
            def write(
                self,
                source: str,
                chunk: bytes,
                captured_at_ns: int,
                recording_id: str = "",
            ) -> bool:
                time.sleep(0.03)
                return super().write(source, chunk, captured_at_ns, recording_id)

        source_chunks = [pcm_constant(1000, 1600) for _ in range(20)]
        session = FakeSession()
        store = SlowStore()
        controller = LiveAudioController(
            session,
            lambda _key: FakeRecognizer(),
            lambda _source, _config: TimedSource(source_chunks),
            recording_store=store,
        )
        started_at = time.monotonic()

        controller.start(
            LiveAudioConfig(sources=("remote",), record_testing=True, vad_enabled=False),
            "test-key",
        )
        deadline = time.monotonic() + 1
        while not source_finished_at and time.monotonic() < deadline:
            time.sleep(0.005)

        self.assertTrue(source_finished_at)
        self.assertLess(source_finished_at[0] - started_at, 0.2)
        deadline = time.monotonic() + 2
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(store.writes), len(source_chunks))

    def test_microphone_gain_does_not_change_raw_recording(self) -> None:
        recognized: list[bytes] = []

        class CapturingRecognizer(FakeRecognizer):
            def stream_lpcm(self, chunks, *, language: str, sample_rate_hertz: int):
                recognized.extend(chunks)
                return iter(())

        raw = pcm_constant(1000, 1600)
        session = FakeSession()
        store = FakeRecordingStore()
        controller = LiveAudioController(
            session,
            lambda _key: CapturingRecognizer(),
            lambda _source, _config: FakePcmSource([raw]),
            recording_store=store,
        )

        controller.start(
            LiveAudioConfig(
                sources=("mic",),
                record_testing=True,
                vad_enabled=False,
                mic_gain=2.0,
            ),
            "test-key",
        )
        deadline = time.monotonic() + 1
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual([item[1] for item in store.writes], [raw])
        self.assertEqual(recognized, [pcm_constant(2000, 1600)])

    def test_live_controller_records_raw_chunks_before_vad(self) -> None:
        session = FakeSession()
        store = FakeRecordingStore()
        source_chunks = [
            pcm_constant(0, 1600),
            pcm_constant(1000, 1600),
            pcm_constant(0, 1600),
        ]
        controller = LiveAudioController(
            session,
            lambda _key: FakeRecognizer(),
            lambda _source, _config: FakePcmSource(source_chunks),
            recording_store=store,
        )

        controller.start(
            LiveAudioConfig(
                sources=("remote",),
                record_testing=True,
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

        self.assertEqual(len(store.starts), 1)
        self.assertEqual([item[1] for item in store.writes], source_chunks)
        self.assertEqual(store.finishes, 1)
        self.assertEqual(session.transcripts, [("remote", "ru-RU:16000:3", True)])
        self.assertTrue(all(item[2] > 0 for item in store.writes))

    def test_live_controller_does_not_record_when_testing_is_disabled(self) -> None:
        session = FakeSession()
        store = FakeRecordingStore()
        controller = LiveAudioController(
            session,
            lambda _key: FakeRecognizer(),
            lambda _source, _config: FakePcmSource([pcm_constant(1000, 1600)]),
            recording_store=store,
        )

        controller.start(
            LiveAudioConfig(sources=("remote",), vad_enabled=False),
            "test-key",
        )

        deadline = time.monotonic() + 2
        while controller.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(store.starts, [])
        self.assertEqual(store.writes, [])
        self.assertEqual(store.finishes, 0)

    def test_speechkit_controller_falls_back_when_remote_recognition_fails(self) -> None:
        session = FakeSession()
        fallback_called = threading.Event()
        fallback_reasons: list[str] = []

        class FailingRecognizer:
            def stream_lpcm(self, _chunks, *, language: str, sample_rate_hertz: int):
                raise RuntimeError("speechkit stream lost")

        def fallback_starter(_config: LiveAudioConfig, reason: str) -> None:
            fallback_reasons.append(reason)
            fallback_called.set()

        controller = LiveAudioController(
            session,
            lambda _key: FailingRecognizer(),
            lambda _source, _config: FakePcmSource([pcm_constant(1000, 1600)]),
            fallback_starter=fallback_starter,
        )

        controller.start(
            LiveAudioConfig(sources=("remote",), vad_enabled=False),
            "test-key",
        )

        self.assertTrue(fallback_called.wait(2))
        self.assertEqual(fallback_reasons, ["speechkit stream lost"])

    @unittest.skipUnless(os.name == "nt", "Windows audio regression")
    def test_lists_audio_devices_from_separate_windows_threads(self) -> None:
        errors: list[Exception] = []

        def read_devices() -> None:
            try:
                list_audio_devices()
            except Exception as error:
                errors.append(error)

        for _ in range(3):
            thread = threading.Thread(target=read_devices)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        com_errors = [error for error in errors if "0x800401f0" in str(error).lower()]
        self.assertEqual(com_errors, [])

    def test_normalizes_audio_sources(self) -> None:
        self.assertEqual(normalize_sources(["system", "me", "remote"]), ("remote", "mic"))

    def test_converts_float_frames_to_mono_pcm16(self) -> None:
        pcm = float_frames_to_pcm16([[0.5, 0.5], [-1.0, -1.0], [0.0, 0.0]])

        self.assertEqual(len(pcm), 6)
        self.assertEqual(int.from_bytes(pcm[:2], "little", signed=True), 16384)
        self.assertEqual(int.from_bytes(pcm[2:4], "little", signed=True), -32767)

    def test_converts_application_float_stereo_to_16khz_chunks(self) -> None:
        import numpy as np

        frames = np.full((9_600, 2), 0.5, dtype="<f4")
        converter = _FloatStereoConverter(target_rate=16_000, chunk_duration_ms=200)

        chunks = list(converter.feed(frames.tobytes()))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 6_400)
        self.assertAlmostEqual(int.from_bytes(chunks[0][:2], "little", signed=True), 16_384, delta=2)

    def test_application_resampler_suppresses_frequencies_above_16khz_nyquist(self) -> None:
        import numpy as np

        frames = 48_000
        timeline = np.arange(frames, dtype=np.float64) / 48_000

        def converted_rms(frequency: int) -> float:
            mono = (0.5 * np.sin(2 * np.pi * frequency * timeline)).astype("<f4")
            stereo = np.column_stack((mono, mono)).astype("<f4")
            converter = _FloatStereoConverter(target_rate=16_000, chunk_duration_ms=1_000)
            chunk = next(converter.feed(stereo.tobytes()))
            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float64)[200:]
            return float(np.sqrt(np.mean(samples * samples)))

        voice_band = converted_rms(1_000)
        aliased_high = converted_rms(10_000)

        self.assertGreater(voice_band, 10_000)
        self.assertLess(aliased_high, voice_band * 0.08)

    def test_application_resampler_keeps_packet_boundaries_transparent(self) -> None:
        import numpy as np

        rng = np.random.default_rng(42)
        frames = rng.uniform(-0.5, 0.5, size=(9_600, 2)).astype("<f4")
        whole = _FloatStereoConverter(target_rate=16_000, chunk_duration_ms=200)
        split = _FloatStereoConverter(target_rate=16_000, chunk_duration_ms=200)

        whole_chunk = next(whole.feed(frames.tobytes()))
        split_chunks: list[bytes] = []
        for start in range(0, len(frames), 317):
            split_chunks.extend(split.feed(frames[start : start + 317].tobytes()))

        self.assertEqual(split_chunks, [whole_chunk])

    def test_application_source_emits_silence_after_process_audio_stops(self) -> None:
        import numpy as np

        signal = np.full((960, 2), 0.5, dtype="<f4").tobytes()

        class FakeLoopbackSession:
            def __init__(self, _process_id: int) -> None:
                self.sent_signal = False

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read_packets(self):
                if self.sent_signal:
                    return iter(())
                self.sent_signal = True
                return iter((signal,))

        stop = threading.Event()
        source = ProcessLoopbackPcmSource(
            42,
            AudioCaptureConfig(
                process_id=42,
                sample_rate_hertz=16_000,
                chunk_duration_ms=20,
            ),
        )

        with (
            patch("mimir.audio.applications._ProcessLoopbackSession", FakeLoopbackSession),
            patch("mimir.audio.applications.process_exists", return_value=True),
        ):
            chunks = source.chunks(stop)
            signal_chunk = next(chunks)
            silence_chunk = next(chunks)
            stop.set()

        self.assertGreater(int.from_bytes(signal_chunk[:2], "little", signed=True), 0)
        self.assertEqual(silence_chunk[-200:], bytes(200))

    def test_application_source_reopens_stalled_process_capture(self) -> None:
        import numpy as np

        signal = np.full((960, 2), 0.5, dtype="<f4").tobytes()
        opened_sessions: list[int] = []
        diagnostics: list[tuple[str, dict[str, object]]] = []

        class FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def monotonic(self) -> float:
                self.value += 0.011
                return self.value

        class FakeLoopbackSession:
            def __init__(self, _process_id: int) -> None:
                self.index = len(opened_sessions)
                self.sent_signal = False
                opened_sessions.append(self.index)

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read_packets(self):
                if self.index == 0 or self.sent_signal:
                    return iter(())
                self.sent_signal = True
                return iter((signal,))

        stop = threading.Event()
        clock = FakeClock()
        source = ProcessLoopbackPcmSource(
            42,
            AudioCaptureConfig(
                process_id=42,
                sample_rate_hertz=16_000,
                chunk_duration_ms=20,
            ),
            refresh_after_silence_seconds=0.03,
            session_factory=FakeLoopbackSession,
            monotonic=clock.monotonic,
            diagnostic_callback=lambda event, payload: diagnostics.append((event, payload)),
        )

        with patch("mimir.audio.applications.process_exists", return_value=True):
            chunks = source.chunks(stop)
            received = [next(chunks) for _ in range(3)]
            stop.set()

        self.assertGreaterEqual(len(opened_sessions), 2)
        self.assertTrue(any(any(chunk) for chunk in received))
        self.assertEqual(diagnostics[0][0], "audio.application_capture.refresh")
        self.assertEqual(diagnostics[0][1]["refreshCount"], 1)

    def test_application_source_keeps_refreshing_during_long_silence(self) -> None:
        opened_sessions: list[int] = []
        diagnostics: list[tuple[str, dict[str, object]]] = []

        class FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def monotonic(self) -> float:
                self.value += 0.011
                return self.value

        class EmptyLoopbackSession:
            def __init__(self, _process_id: int) -> None:
                opened_sessions.append(len(opened_sessions))

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read_packets(self):
                return iter(())

        source = ProcessLoopbackPcmSource(
            42,
            AudioCaptureConfig(
                process_id=42,
                sample_rate_hertz=16_000,
                chunk_duration_ms=20,
            ),
            refresh_after_silence_seconds=0.03,
            max_consecutive_refreshes=2,
            session_factory=EmptyLoopbackSession,
            monotonic=FakeClock().monotonic,
            diagnostic_callback=lambda event, payload: diagnostics.append((event, payload)),
        )

        stop = threading.Event()
        with patch("mimir.audio.applications.process_exists", return_value=True):
            chunks = source.chunks(stop)
            for _ in range(20):
                next(chunks)
                if len(opened_sessions) >= 4:
                    break
            stop.set()

        self.assertGreaterEqual(len(opened_sessions), 4)
        events = [event for event, _payload in diagnostics]
        self.assertGreaterEqual(events.count("audio.application_capture.refresh"), 3)
        self.assertEqual(events.count("audio.application_capture.stalled"), 1)
        self.assertNotIn("audio.application_capture.failed", events)

    def test_application_source_retries_real_capture_errors(self) -> None:
        attempts = 0
        diagnostics: list[str] = []
        signal = struct.pack("<1920f", *([0.4] * 1920))

        class RecoveringLoopbackSession:
            def __init__(self, _process_id: int) -> None:
                nonlocal attempts
                attempts += 1

            def __enter__(self):
                if attempts < 3:
                    raise AudioCaptureError("временная ошибка")
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read_packets(self):
                return iter((signal,))

        stop = threading.Event()
        source = ProcessLoopbackPcmSource(
            42,
            AudioCaptureConfig(process_id=42, chunk_duration_ms=20),
            session_factory=RecoveringLoopbackSession,
            diagnostic_callback=lambda event, _payload: diagnostics.append(event),
        )

        with patch("mimir.audio.applications.process_exists", return_value=True):
            chunk = next(source.chunks(stop))
            stop.set()

        self.assertTrue(any(chunk))
        self.assertEqual(attempts, 3)
        self.assertEqual(diagnostics.count("audio.application_capture.retry"), 2)

    def test_application_source_resets_read_errors_after_successful_quiet_reads(self) -> None:
        opened_sessions: list[int] = []
        diagnostics: list[str] = []
        signal = struct.pack("<1920f", *([0.4] * 1920))

        class FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def monotonic(self) -> float:
                self.value += 0.011
                return self.value

        class IntermittentLoopbackSession:
            def __init__(self, _process_id: int) -> None:
                self.index = len(opened_sessions)
                self.sent_signal = False
                opened_sessions.append(self.index)

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read_packets(self):
                if self.index in {0, 2, 4}:
                    raise RuntimeError("временная ошибка чтения")
                if self.index < 5 or self.sent_signal:
                    return iter(())
                self.sent_signal = True
                return iter((signal,))

        stop = threading.Event()
        source = ProcessLoopbackPcmSource(
            42,
            AudioCaptureConfig(process_id=42, chunk_duration_ms=20),
            refresh_after_silence_seconds=0.03,
            max_consecutive_refreshes=3,
            session_factory=IntermittentLoopbackSession,
            monotonic=FakeClock().monotonic,
            diagnostic_callback=lambda event, _payload: diagnostics.append(event),
        )

        with patch("mimir.audio.applications.process_exists", return_value=True):
            chunks = source.chunks(stop)
            received = []
            for _ in range(30):
                chunk = next(chunks)
                received.append(chunk)
                if any(chunk):
                    break
            stop.set()

        self.assertTrue(any(any(chunk) for chunk in received))
        self.assertGreaterEqual(len(opened_sessions), 6)
        self.assertNotIn("audio.application_capture.failed", diagnostics)

    def test_remote_source_uses_selected_application_instead_of_output_device(self) -> None:
        source = default_source_factory(
            "remote",
            AudioCaptureConfig(device_id="ignored-output", process_id=42),
        )

        self.assertIsInstance(source, ProcessLoopbackPcmSource)
        self.assertEqual(source.process_id, 42)

    def test_remote_application_does_not_require_output_device_list(self) -> None:
        checks: list[dict[str, object]] = []

        def fail_if_called() -> list[dict[str, object]]:
            raise AssertionError("Список устройств не должен запрашиваться")

        add_device_checks(
            checks,
            ["remote"],
            {},
            fail_if_called,
            application_process_id=42,
            list_audio_applications=lambda: [
                {"processId": 42, "executable": "meeting.exe", "title": "Созвон"},
            ],
        )

        self.assertEqual(checks, [{"name": "remote_application", "ok": True, "detail": "Приложение созвона: Созвон"}])

    def test_remote_application_must_be_selected_and_running(self) -> None:
        checks: list[dict[str, object]] = []

        add_device_checks(
            checks,
            ["remote"],
            {},
            lambda: [],
            application_process_id=0,
            list_audio_applications=lambda: [],
        )

        self.assertFalse(checks[0]["ok"])
        self.assertEqual(checks[0]["detail"], "Выберите приложение созвона в настройках")

    def test_saved_application_is_used_when_request_has_no_process(self) -> None:
        class SavedApplication:
            process_id = 73

        class SavedTesting:
            enabled = True

        class SavedConfig:
            audio_mode = "yandex_realtime"
            audio_application = SavedApplication()
            testing = SavedTesting()

        mode, config = parse_live_audio_request({"sources": ["remote"]}, lambda: SavedConfig())

        self.assertEqual(mode, "yandex_realtime")
        self.assertEqual(config["application_process_id"], 73)
        self.assertTrue(config["record_testing"])

    @unittest.skipUnless(os.name == "nt", "Windows process check")
    def test_current_process_is_available_for_application_capture(self) -> None:
        self.assertTrue(process_exists(os.getpid()))

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
        self.assertEqual(ended.trailing_silence_ms, 250)
        self.assertTrue(ended.send_to_stt)

    def test_vad_sends_six_hundred_milliseconds_before_speech(self) -> None:
        gate = EnergyVadGate(16_000)
        quiet_chunks = [pcm_constant(40 + index, 3_200) for index in range(3)]
        for chunk in quiet_chunks:
            self.assertFalse(gate.process(chunk).send_to_stt)

        speech = pcm_constant(1_000, 3_200)
        decision = gate.process(speech)

        self.assertTrue(decision.speech_started)
        self.assertEqual(decision.audio_chunks, (*quiet_chunks, speech))

    def test_vad_adapts_to_quiet_speech_below_old_threshold(self) -> None:
        gate = EnergyVadGate(16_000)
        for _ in range(10):
            gate.process(pcm_constant(70, 3_200))

        decision = gate.process(pcm_constant(250, 3_200))

        self.assertTrue(decision.speech_started)
        self.assertLess(decision.speech_threshold, 350)

    def test_vad_keeps_first_hundred_milliseconds_until_speech_is_confirmed(self) -> None:
        gate = EnergyVadGate(
            16_000,
            EnergyVadConfig(pre_roll_ms=200, min_speech_ms=120),
        )
        gate.process(pcm_constant(0, 1_600))
        first = pcm_constant(600, 1_600)
        second = pcm_constant(700, 1_600)

        candidate = gate.process(first)
        confirmed = gate.process(second)

        self.assertFalse(candidate.send_to_stt)
        self.assertIn(first, confirmed.audio_chunks)
        self.assertEqual(confirmed.audio_chunks[-1], second)

    def test_vad_sends_exactly_two_seconds_of_tail_silence(self) -> None:
        gate = EnergyVadGate(16_000)
        gate.process(pcm_constant(1_000, 3_200))

        decisions = [gate.process(pcm_constant(0, 3_200)) for _ in range(10)]

        self.assertTrue(all(decision.send_to_stt for decision in decisions))
        self.assertFalse(any(decision.speech_ended for decision in decisions[:-1]))
        self.assertTrue(decisions[-1].speech_ended)

    def test_live_controller_finishes_current_utterance_before_forced_cancel(self) -> None:
        session = FakeSession()

        class WaitingSource:
            def chunks(self, stop_event) -> Iterator[bytes]:
                yield pcm_constant(1_000, 3_200)
                stop_event.wait(5)

        class FinalizingRecognizer(FakeRecognizer):
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def stream_lpcm(self, chunks, *, language: str, sample_rate_hertz: int):
                consumed = list(chunks)
                if consumed and not self.cancelled:
                    yield SpeechRecognitionResult("последняя фраза", True, end_of_utterance=True)

        recognizer = FinalizingRecognizer()
        controller = LiveAudioController(
            session,
            lambda _key: recognizer,
            lambda _source, _config: WaitingSource(),
        )
        controller.start(LiveAudioConfig(sources=("remote",)), "test-key")
        time.sleep(0.05)

        controller.stop()

        self.assertFalse(recognizer.cancelled)
        self.assertEqual(session.transcripts, [("remote", "последняя фраза", True)])

    def test_realtime_prompt_keeps_context_contract_without_language_rule(self) -> None:
        self.assertIn("DIALOGUE_CONTEXT", REALTIME_INSTRUCTIONS)
        self.assertIn("не участник созвона", REALTIME_INSTRUCTIONS)
        self.assertIn("Не выдумывай опыт", REALTIME_INSTRUCTIONS)
        self.assertNotIn("на языке вопроса", REALTIME_INSTRUCTIONS)
        self.assertNotIn("пиши по-русски", REALTIME_INSTRUCTIONS)

    def test_realtime_controller_reports_thread_that_is_still_stopping(self) -> None:
        release = threading.Event()
        thread = threading.Thread(target=release.wait, daemon=True)
        thread.start()
        controller = RealtimeAudioController(FakeSession(), lambda _key: FakeRecognizer())
        with controller._lock:  # noqa: SLF001 - моделируем зависшее завершение
            controller._thread = thread  # noqa: SLF001
            controller._running = False  # noqa: SLF001

        self.assertTrue(controller.has_live_threads())

        release.set()
        thread.join(timeout=1)
        self.assertFalse(controller.has_live_threads())

    def test_realtime_controller_updates_instructions_after_settings_change(self) -> None:
        class DynamicSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self.goal = "Старая цель"

            def realtime_instructions(self, base: str) -> str:
                return f"{base}\nЦель: {self.goal}"

        class FakeRealtimeClient:
            def __init__(self) -> None:
                self.instructions: list[str] = []

            async def append_audio(self, _pcm: bytes) -> None:
                pass

            async def add_dialogue_context(self, _text: str) -> None:
                pass

            async def update_instructions(self, instructions: str) -> None:
                self.instructions.append(instructions)

        session = DynamicSession()
        controller = RealtimeAudioController(session, lambda _key: FakeRecognizer())
        controller._last_realtime_instructions = session.realtime_instructions(REALTIME_INSTRUCTIONS)
        client = FakeRealtimeClient()

        async def run_sender() -> None:
            queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            sender = asyncio.create_task(
                controller._send_events(client, queue, stop_event, 0)
            )
            await asyncio.sleep(0.05)
            session.goal = "Новая цель"
            controller.refresh_settings()
            await asyncio.sleep(0.3)
            stop_event.set()
            await asyncio.wait_for(sender, timeout=1)

        asyncio.run(run_sender())

        self.assertEqual(len(client.instructions), 1)
        self.assertIn("Цель: Новая цель", client.instructions[0])

    def test_realtime_controller_skips_stale_queued_instructions(self) -> None:
        class FakeRealtimeClient:
            def __init__(self) -> None:
                self.instructions: list[str] = []

            async def append_audio(self, _pcm: bytes) -> None:
                pass

            async def add_dialogue_context(self, _text: str) -> None:
                pass

            async def update_instructions(self, instructions: str) -> None:
                self.instructions.append(instructions)

        controller = RealtimeAudioController(FakeSession(), lambda _key: FakeRecognizer())
        controller._last_realtime_instructions = "Новые настройки"
        client = FakeRealtimeClient()

        async def run_sender() -> None:
            queue: asyncio.Queue = asyncio.Queue()
            await queue.put(("session_instructions", "remote", "Старые настройки"))
            stop_event = threading.Event()
            sender = asyncio.create_task(controller._send_events(client, queue, stop_event, 0))
            await asyncio.sleep(0.05)
            stop_event.set()
            await asyncio.wait_for(sender, timeout=1)

        asyncio.run(run_sender())

        self.assertEqual(client.instructions, [])

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
        self.assertEqual(session.external_questions[0][2], 1)
        self.assertTrue(any(event == "answer_delta" for event, _payload in session.events))

    def test_realtime_microphone_refinement_keeps_refinement_flag(self) -> None:
        session = FakeSession()

        class RefiningRecognizer:
            def stream_lpcm(
                self,
                chunks: Iterable[bytes],
                *,
                language: str,
                sample_rate_hertz: int,
            ) -> Iterator[SpeechRecognitionResult]:
                del language, sample_rate_hertz
                if list(chunks):
                    yield SpeechRecognitionResult(
                        text="Уточненный итог",
                        is_final=True,
                        end_of_utterance=True,
                        is_refinement=True,
                    )

        controller = RealtimeAudioController(
            session,
            lambda _key: RefiningRecognizer(),
            source_factory=lambda _source, _config: FakePcmSource([pcm_constant(1000, 3200)]),
        )
        loop = asyncio.new_event_loop()
        try:
            controller._produce_mic_context(  # noqa: SLF001 - проверяем передачу признака
                RealtimeAudioConfig(sources=("mic",), vad_enabled=False),
                threading.Event(),
                loop,
                asyncio.Queue(),
                "test-key",
            )
        finally:
            loop.close()

        self.assertIn(("mic", True, True), session.stt_refinements)

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

    def test_realtime_controller_falls_back_when_remote_capture_fails(self) -> None:
        session = FakeSession()
        fallback_called = threading.Event()
        fallback_reasons: list[str] = []

        def missing_source(_source, _config):
            raise RuntimeError("loopback lost")

        def fallback_starter(_config: RealtimeAudioConfig, reason: str) -> None:
            fallback_reasons.append(reason)
            fallback_called.set()

        controller = RealtimeAudioController(
            session,
            lambda _key: FakeRecognizer(),
            source_factory=missing_source,
            fallback_starter=fallback_starter,
        )
        config = RealtimeAudioConfig(sources=("remote",))
        stop_event = threading.Event()
        controller._config = config
        controller._stop_event = stop_event
        controller._running = True

        controller._produce_remote_audio(config, stop_event, None, None)

        self.assertTrue(fallback_called.wait(1))
        self.assertTrue(stop_event.is_set())
        self.assertEqual(fallback_reasons, ["loopback lost"])
        self.assertTrue(
            any(
                event == "audio_error" and payload.get("phase") == "remote_producer"
                for event, payload in session.events
            )
        )


if __name__ == "__main__":
    unittest.main()
