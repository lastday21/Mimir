import unittest

from mimir.providers.base import ProviderError
from mimir.providers.yandex_speechkit import (
    YandexSpeechKitClient,
    build_streaming_requests,
    first_text,
    normalize_language,
    parse_streaming_response,
)

from yandex.cloud.ai.stt.v3 import stt_pb2


class YandexSpeechKitTests(unittest.TestCase):
    def test_normalizes_short_language_codes(self) -> None:
        self.assertEqual(normalize_language("ru"), "ru-RU")
        self.assertEqual(normalize_language("en"), "en-US")
        self.assertEqual(normalize_language("tr"), "tr-TR")

    def test_rejects_missing_key_before_network_call(self) -> None:
        with self.assertRaises(ProviderError):
            YandexSpeechKitClient("").recognize_lpcm(b"\0\0")

    def test_rejects_unsupported_sample_rate(self) -> None:
        with self.assertRaisesRegex(ProviderError, "sample rate"):
            YandexSpeechKitClient("test").recognize_lpcm(
                b"\0\0",
                sample_rate_hertz=44_100,
            )

    def test_extracts_first_streaming_alternative(self) -> None:
        self.assertEqual(first_text(["hello", "ignored"]), "hello")
        self.assertEqual(first_text([stt_pb2.Alternative(text="hello")]), "hello")
        self.assertEqual(first_text("привет"), "привет")

    def test_builds_grpc_streaming_options_before_audio_chunks(self) -> None:
        requests = list(build_streaming_requests(stt_pb2, [b"1234"], "ru", 16_000))

        self.assertEqual(requests[0].WhichOneof("Event"), "session_options")
        self.assertEqual(requests[1].WhichOneof("Event"), "chunk")
        model = requests[0].session_options.recognition_model
        self.assertEqual(model.audio_processing_type, stt_pb2.RecognitionModelOptions.REAL_TIME)
        self.assertEqual(model.audio_format.raw_audio.audio_encoding, stt_pb2.RawAudio.LINEAR16_PCM)
        self.assertEqual(model.language_restriction.language_code[0], "ru-RU")

    def test_parses_grpc_final_response(self) -> None:
        response = stt_pb2.StreamingResponse(
            final=stt_pb2.AlternativeUpdate(
                alternatives=[stt_pb2.Alternative(text="готово")]
            )
        )

        result = parse_streaming_response(response)

        self.assertIsNotNone(result)
        self.assertEqual(result.text, "готово")
        self.assertTrue(result.is_final)

    def test_marks_normalized_final_as_refinement(self) -> None:
        response = stt_pb2.StreamingResponse(
            final_refinement=stt_pb2.FinalRefinement(
                final_index=0,
                normalized_text=stt_pb2.AlternativeUpdate(
                    alternatives=[stt_pb2.Alternative(text="Я закончил проект")]
                ),
            )
        )

        result = parse_streaming_response(response)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_final)
        self.assertTrue(result.is_refinement)
        self.assertEqual(result.text, "Я закончил проект")


if __name__ == "__main__":
    unittest.main()
