from __future__ import annotations

import json
import threading
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
STREAMING_MAX_ATTEMPTS = 2
STREAMING_RETRY_DELAY_SECONDS = 0.75


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
        self._stream_lock = threading.Lock()
        self._active_call: object | None = None
        self._active_channel: object | None = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._stream_lock:
            call = self._active_call
            channel = self._active_channel
        cancel = getattr(call, "cancel", None)
        if callable(cancel):
            cancel()
        close = getattr(channel, "close", None)
        if callable(close):
            close()

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

        metadata = (
            ("authorization", f"Api-Key {self.api_key}"),
            ("x-data-logging-enabled", "false"),
        )
        chunk_iterator = iter(chunks)
        for attempt in range(STREAMING_MAX_ATTEMPTS):
            channel = grpc.secure_channel(STREAMING_GRPC_TARGET, grpc.ssl_channel_credentials())
            stub = stt_service_pb2_grpc.RecognizerStub(channel)
            requests = build_streaming_requests(
                stt_pb2,
                chunk_iterator,
                language,
                sample_rate_hertz,
            )
            assembler = SpeechKitUtteranceAssembler()
            call: object | None = None
            try:
                call = stub.RecognizeStreaming(requests, metadata=metadata, timeout=330)
                with self._stream_lock:
                    self._active_call = call
                    self._active_channel = channel
                for response in call:
                    validate_streaming_status(response)
                    result = assembler.accept(response)
                    if result is not None:
                        yield result
                final_result = assembler.flush()
                if final_result is not None:
                    yield final_result
                return
            except grpc.RpcError as error:
                if self._cancelled.is_set():
                    return
                if attempt + 1 < STREAMING_MAX_ATTEMPTS and is_retryable_grpc_error(error):
                    if self._cancelled.wait(STREAMING_RETRY_DELAY_SECONDS):
                        return
                    continue
                detail = error.details() if hasattr(error, "details") else str(error)
                raise ProviderError(f"Yandex SpeechKit gRPC streaming failed: {detail}") from error
            finally:
                with self._stream_lock:
                    if self._active_call is call:
                        self._active_call = None
                        self._active_channel = None
                cancel = getattr(call, "cancel", None)
                if callable(cancel):
                    cancel()
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
                    type=stt_pb2.DefaultEouClassifier.DEFAULT,
                    max_pause_between_words_hint_ms=1_200,
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
            return SpeechRecognitionResult(text=text, is_final=False, end_of_utterance=False)
    if event == "final_refinement":
        text = text_from_update(response.final_refinement.normalized_text)
        if text:
            return SpeechRecognitionResult(
                text=text,
                is_final=False,
                end_of_utterance=False,
                is_refinement=True,
            )
    return None


class SpeechKitUtteranceAssembler:
    def __init__(self) -> None:
        self._segments: dict[int, str] = {}
        self._segment_order: list[int] = []
        self._partial = ""
        self._closed = False
        self._last_emitted: tuple[str, bool, bool] | None = None

    def accept(self, response: object) -> SpeechRecognitionResult | None:
        event = response.WhichOneof("Event")
        if event == "partial":
            self._start_next_utterance()
            self._partial = text_from_update(response.partial)
            return self._emit(self._combined_text(), is_final=False)

        if event == "final":
            self._start_next_utterance()
            text = text_from_update(response.final)
            if not text:
                return None
            self._partial = ""
            self._store_segment(streaming_final_index(response, self._segment_order), text)
            return self._emit(self._combined_text(), is_final=False)

        if event == "final_refinement":
            text = text_from_update(response.final_refinement.normalized_text)
            if not text:
                return None
            index = int(getattr(response.final_refinement, "final_index", 0))
            self._store_segment(index, text)
            return self._emit(
                self._combined_text(),
                is_final=self._closed,
                is_refinement=self._closed,
            )

        if event == "eou_update":
            text = self._combined_text()
            if not text:
                return None
            self._closed = True
            self._partial = ""
            return self._emit(text, is_final=True)

        return None

    def flush(self) -> SpeechRecognitionResult | None:
        if self._closed:
            return None
        text = self._combined_text()
        if not text:
            return None
        self._closed = True
        return self._emit(text, is_final=True)

    def _start_next_utterance(self) -> None:
        if not self._closed:
            return
        self._segments = {}
        self._segment_order = []
        self._partial = ""
        self._closed = False
        self._last_emitted = None

    def _store_segment(self, index: int, text: str) -> None:
        if index not in self._segments:
            self._segment_order.append(index)
        self._segments[index] = text.strip()

    def _combined_text(self) -> str:
        parts = [self._segments[index] for index in self._segment_order]
        if self._partial:
            parts.append(self._partial)
        return " ".join(part.strip() for part in parts if part.strip()).strip()

    def _emit(
        self,
        text: str,
        *,
        is_final: bool,
        is_refinement: bool = False,
    ) -> SpeechRecognitionResult | None:
        normalized = text.strip()
        if not normalized:
            return None
        key = (normalized, is_final, is_refinement)
        if key == self._last_emitted:
            return None
        self._last_emitted = key
        return SpeechRecognitionResult(
            text=normalized,
            is_final=is_final,
            end_of_utterance=is_final,
            is_refinement=is_refinement,
        )


def streaming_final_index(response: object, existing: list[int]) -> int:
    try:
        if response.HasField("audio_cursors"):
            return int(response.audio_cursors.final_index)
    except (AttributeError, ValueError):
        pass
    return max(existing, default=-1) + 1


def validate_streaming_status(response: object) -> None:
    if response.WhichOneof("Event") != "status_code":
        return
    code_type = getattr(response.status_code, "code_type", 0)
    message = getattr(response.status_code, "message", "")
    if message and code_type not in {0, 1}:
        raise ProviderError(f"Yandex SpeechKit status: {message}")


def is_retryable_grpc_error(error: object) -> bool:
    code = None
    code_reader = getattr(error, "code", None)
    if callable(code_reader):
        try:
            code = code_reader()
        except Exception:
            code = None
    code_name = str(getattr(code, "name", code) or "").upper()
    if any(
        marker in code_name
        for marker in (
            "CANCELLED",
            "INVALID_ARGUMENT",
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
            "RESOURCE_EXHAUSTED",
        )
    ):
        return False
    if any(
        marker in code_name
        for marker in ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "UNKNOWN")
    ):
        return True

    detail_reader = getattr(error, "details", None)
    try:
        detail = str(detail_reader() if callable(detail_reader) else error).lower()
    except Exception:
        detail = str(error).lower()
    return any(
        marker in detail
        for marker in (
            "failed to connect",
            "connection reset",
            "connection refused",
            "handshaker shutdown",
            "temporarily unavailable",
            "timed out",
        )
    )


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
