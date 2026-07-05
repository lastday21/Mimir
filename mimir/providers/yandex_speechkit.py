from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .base import ProviderError


RECOGNIZE_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
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
