import io
import unittest
import wave
from collections.abc import Iterable, Iterator

from mimir.models import SpeechRecognitionResult
from mimir.stt import AudioStreamConfig, SpeechKitStreamRunner, chunk_pcm, pcm_chunks_from_wav


class FakeRecognizer:
    def __init__(self) -> None:
        self.sessions = 0
        self.total_chunks = 0

    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str,
        sample_rate_hertz: int,
    ) -> Iterator[SpeechRecognitionResult]:
        self.sessions += 1
        consumed = list(chunks)
        self.total_chunks += len(consumed)
        yield SpeechRecognitionResult(
            text=f"{language}:{sample_rate_hertz}:{len(consumed)}",
            is_final=True,
            end_of_utterance=True,
        )


class SpeechKitStreamTests(unittest.TestCase):
    def test_chunks_pcm_by_duration(self) -> None:
        pcm = b"\0" * 16_000

        chunks = list(chunk_pcm(pcm, sample_rate_hertz=16_000, chunk_duration_ms=250))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(pcm))

    def test_runner_rotates_by_byte_limit(self) -> None:
        recognizer = FakeRecognizer()
        runner = SpeechKitStreamRunner(
            recognizer,
            AudioStreamConfig(rotate_after_bytes=5),
        )

        events = list(runner.run("remote", [b"1111", b"2222", b"3333"]))

        self.assertEqual(recognizer.sessions, 3)
        self.assertEqual([event.text for event in events], ["ru-RU:16000:1"] * 3)

    def test_reads_mono_pcm_wav(self) -> None:
        raw = io.BytesIO()
        with wave.open(raw, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(b"\0" * 3_200)

        sample_rate, chunks = pcm_chunks_from_wav(raw.getvalue(), chunk_duration_ms=100)

        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 3_200)


if __name__ == "__main__":
    unittest.main()
