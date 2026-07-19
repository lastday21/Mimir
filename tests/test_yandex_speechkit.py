import unittest

from mimir.providers.base import ProviderError
from mimir.providers.yandex_speechkit import (
    YandexSpeechKitClient,
    SpeechKitUtteranceAssembler,
    build_streaming_requests,
    first_text,
    is_retryable_grpc_error,
    normalize_language,
    parse_streaming_response,
)

from yandex.cloud.ai.stt.v3 import stt_pb2


class YandexSpeechKitTests(unittest.TestCase):
    def test_cancel_stops_active_grpc_call_and_channel(self) -> None:
        class FakeCall:
            cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        class FakeChannel:
            closed = False

            def close(self) -> None:
                self.closed = True

        client = YandexSpeechKitClient("test")
        call = FakeCall()
        channel = FakeChannel()
        client._active_call = call
        client._active_channel = channel

        client.cancel()

        self.assertTrue(call.cancelled)
        self.assertTrue(channel.closed)

    def test_retries_temporary_grpc_connection_error(self) -> None:
        class TemporaryError(Exception):
            def code(self):
                return type("Code", (), {"name": "UNKNOWN"})()

            def details(self) -> str:
                return "tcp handshaker shutdown"

        self.assertTrue(is_retryable_grpc_error(TemporaryError()))

    def test_does_not_retry_access_error(self) -> None:
        class AccessError(Exception):
            def code(self):
                return type("Code", (), {"name": "PERMISSION_DENIED"})()

            def details(self) -> str:
                return "permission denied"

        self.assertFalse(is_retryable_grpc_error(AccessError()))

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
        classifier = requests[0].session_options.eou_classifier.default_classifier
        self.assertEqual(classifier.type, stt_pb2.DefaultEouClassifier.DEFAULT)
        self.assertEqual(classifier.max_pause_between_words_hint_ms, 1_200)

    def test_parses_grpc_final_response(self) -> None:
        response = stt_pb2.StreamingResponse(
            final=stt_pb2.AlternativeUpdate(
                alternatives=[stt_pb2.Alternative(text="готово")]
            )
        )

        result = parse_streaming_response(response)

        self.assertIsNotNone(result)
        self.assertEqual(result.text, "готово")
        self.assertFalse(result.is_final)

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
        self.assertFalse(result.is_final)
        self.assertTrue(result.is_refinement)
        self.assertEqual(result.text, "Я закончил проект")

    def test_assembles_all_final_segments_before_end_of_utterance(self) -> None:
        assembler = SpeechKitUtteranceAssembler()

        partial = assembler.accept(
            stt_pb2.StreamingResponse(
                partial=stt_pb2.AlternativeUpdate(
                    alternatives=[stt_pb2.Alternative(text="сколько")]
                )
            )
        )
        first_final = assembler.accept(
            stt_pb2.StreamingResponse(
                audio_cursors=stt_pb2.AudioCursors(final_index=0),
                final=stt_pb2.AlternativeUpdate(
                    alternatives=[stt_pb2.Alternative(text="сколько будет")]
                ),
            )
        )
        second_final = assembler.accept(
            stt_pb2.StreamingResponse(
                audio_cursors=stt_pb2.AudioCursors(final_index=1),
                final=stt_pb2.AlternativeUpdate(
                    alternatives=[stt_pb2.Alternative(text="четыре плюс пять")]
                ),
            )
        )
        complete = assembler.accept(
            stt_pb2.StreamingResponse(eou_update=stt_pb2.EouUpdate(time_ms=1_500))
        )

        self.assertEqual(partial.text, "сколько")
        self.assertFalse(partial.is_final)
        self.assertEqual(first_final.text, "сколько будет")
        self.assertFalse(first_final.is_final)
        self.assertEqual(second_final.text, "сколько будет четыре плюс пять")
        self.assertFalse(second_final.is_final)
        self.assertEqual(complete.text, "сколько будет четыре плюс пять")
        self.assertTrue(complete.is_final)
        self.assertTrue(complete.end_of_utterance)

    def test_late_refinement_replaces_completed_utterance(self) -> None:
        assembler = SpeechKitUtteranceAssembler()
        assembler.accept(
            stt_pb2.StreamingResponse(
                audio_cursors=stt_pb2.AudioCursors(final_index=0),
                final=stt_pb2.AlternativeUpdate(
                    alternatives=[stt_pb2.Alternative(text="двадцать пять процентов")]
                ),
            )
        )
        assembler.accept(stt_pb2.StreamingResponse(eou_update=stt_pb2.EouUpdate(time_ms=1_000)))

        refinement = assembler.accept(
            stt_pb2.StreamingResponse(
                final_refinement=stt_pb2.FinalRefinement(
                    final_index=0,
                    normalized_text=stt_pb2.AlternativeUpdate(
                        alternatives=[stt_pb2.Alternative(text="25%")]
                    ),
                )
            )
        )

        self.assertEqual(refinement.text, "25%")
        self.assertTrue(refinement.is_final)
        self.assertTrue(refinement.is_refinement)


if __name__ == "__main__":
    unittest.main()
