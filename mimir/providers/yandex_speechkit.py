from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator

from ..models import SpeechRecognitionResult
from .base import ProviderError


RECOGNIZE_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
STREAMING_GRPC_TARGET = "stt.api.cloud.yandex.net:443"
DEFAULT_LANGUAGE = "ru-RU"
DEFAULT_SAMPLE_RATE = 16_000
MAX_SHORT_AUDIO_BYTES = 1_000_000


def normalize_language(language: str) -> str:
    value = language.strip()
    if not value:
        return DEFAULT_LANGUAGE
    return {
        "ru": "ru-RU",
        "en": "en-US",
        "tr": "tr-TR",
    }.get(value, value)


class YandexSpeechKitClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def recognize_lpcm(
        self,
        audio: bytes,
        *,
        language: str = DEFAULT_LANGUAGE,
        sample_rate_hertz: int = DEFAULT_SAMPLE_RATE,
    ) -> str:
        if not self.api_key:
            raise ProviderError("Yandex SpeechKit API key is not configured")
        if not audio:
            raise ProviderError("Audio payload is empty")
        if len(audio) > MAX_SHORT_AUDIO_BYTES:
            raise ProviderError("Audio is too large for short SpeechKit recognition")
        if sample_rate_hertz not in {8_000, 16_000, 48_000}:
            raise ProviderError("SpeechKit LPCM sample rate must be 8000, 16000, or 48000 Hz")

        query = urllib.parse.urlencode(
            {
                "lang": normalize_language(language),
                "topic": "general",
                "format": "lpcm",
                "sampleRateHertz": str(sample_rate_hertz),
            }
        )
        request = urllib.request.Request(
            f"{RECOGNIZE_URL}?{query}",
            data=audio,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/octet-stream",
                "x-data-logging-enabled": "false",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Yandex SpeechKit returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(f"Yandex SpeechKit is not reachable: {error.reason}") from error

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProviderError("Yandex SpeechKit returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ProviderError("Yandex SpeechKit returned unexpected JSON")
        if "error_code" in payload:
            message = payload.get("error_message") or payload.get("error_code")
            raise ProviderError(f"Yandex SpeechKit error: {message}")
        result = payload.get("result")
        if not isinstance(result, str):
            raise ProviderError("Yandex SpeechKit response did not contain text")
        return result.strip()

    def stream_lpcm(
        self,
        chunks: Iterable[bytes],
        *,
        language: str = DEFAULT_LANGUAGE,
        sample_rate_hertz: int = DEFAULT_SAMPLE_RATE,
    ) -> Iterator[SpeechRecognitionResult]:
        if not self.api_key:
            raise ProviderError("Yandex SpeechKit API key is not configured")
        if sample_rate_hertz not in {8_000, 16_000, 48_000}:
            raise ProviderError("SpeechKit LPCM sample rate must be 8000, 16000, or 48000 Hz")

        try:
            import grpc
            from yandex.cloud.ai.stt.v3 import stt_pb2, stt_service_pb2_grpc
        except ImportError as error:
            raise ProviderError(
                "SpeechKit streaming requires `grpcio` and `yandexcloud` packages"
            ) from error

        channel = grpc.secure_channel(STREAMING_GRPC_TARGET, grpc.ssl_channel_credentials())
        stub = stt_service_pb2_grpc.RecognizerStub(channel)
        metadata = (
            ("authorization", f"Api-Key {self.api_key}"),
            ("x-data-logging-enabled", "false"),
        )
        requests = build_streaming_requests(stt_pb2, chunks, language, sample_rate_hertz)
        last_final = ""
        try:
            for response in stub.RecognizeStreaming(requests, metadata=metadata, timeout=330):
                result = parse_streaming_response(response)
                if result is None:
                    continue
                if result.is_final:
                    if result.text == last_final:
                        continue
                    last_final = result.text
                yield result
        except grpc.RpcError as error:
            detail = error.details() if hasattr(error, "details") else str(error)
            raise ProviderError(f"Yandex SpeechKit gRPC streaming failed: {detail}") from error
        finally:
            channel.close()


def build_streaming_requests(
    stt_pb2: object,
    chunks: Iterable[bytes],
    language: str,
    sample_rate_hertz: int,
) -> Iterator[object]:
    yield stt_pb2.StreamingRequest(
        session_options=stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                model="general",
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=sample_rate_hertz,
                        audio_channel_count=1,
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=False,
                    literature_text=False,
                    phone_formatting_mode=stt_pb2.TextNormalizationOptions.PHONE_FORMATTING_MODE_DISABLED,
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[normalize_language(language)],
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
            ),
            eou_classifier=stt_pb2.EouClassifierOptions(
                default_classifier=stt_pb2.DefaultEouClassifier(
                    type=stt_pb2.DefaultEouClassifier.HIGH,
                    max_pause_between_words_hint_ms=700,
                )
            ),
        )
    )
    for chunk in chunks:
        if chunk:
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))


def parse_streaming_response(response: object) -> SpeechRecognitionResult | None:
    event = response.WhichOneof("Event")
    if event == "partial":
        text = text_from_update(response.partial)
        if text:
            return SpeechRecognitionResult(text=text, is_final=False)
    if event == "final":
        text = text_from_update(response.final)
        if text:
            return SpeechRecognitionResult(text=text, is_final=True, end_of_utterance=True)
    if event == "final_refinement":
        text = text_from_update(response.final_refinement.normalized_text)
        if text:
            return SpeechRecognitionResult(
                text=text,
                is_final=True,
                end_of_utterance=True,
                is_refinement=True,
            )
    if event == "status_code":
        code_type = getattr(response.status_code, "code_type", 0)
        message = getattr(response.status_code, "message", "")
        if message and code_type not in {0, 1}:
            raise ProviderError(f"Yandex SpeechKit status: {message}")
    return None


def text_from_update(update: object) -> str:
    alternatives = getattr(update, "alternatives", None)
    return first_text(alternatives)


def first_text(alternatives: object) -> str:
    if isinstance(alternatives, str):
        return alternatives.strip()
    try:
        if alternatives and len(alternatives) > 0:
            first = alternatives[0]
            return str(getattr(first, "text", first) or "").strip()
    except (IndexError, TypeError):
        return ""
    return ""
