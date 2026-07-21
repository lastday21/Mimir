from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .audio.applications import ProcessLoopbackPcmSource
from .audio.capture import AudioCaptureConfig, AudioCaptureError, SoundcardPcmSource
from .providers.base import ProviderError
from .providers.yandex_ai import YandexAIStudioClient
from .providers.yandex_speechkit import STREAMING_GRPC_TARGET, YandexSpeechKitClient


PROBE_TIMEOUT_SECONDS = 8
PROBE_AUDIO_DURATION_MS = 200


def probe_speechkit_connection(api_key: str) -> str:
    try:
        import grpc
    except ImportError as error:
        raise ProviderError("Не установлена библиотека gRPC для SpeechKit") from error
    channel = grpc.secure_channel(STREAMING_GRPC_TARGET, grpc.ssl_channel_credentials())
    try:
        grpc.channel_ready_future(channel).result(timeout=PROBE_TIMEOUT_SECONDS)
    except grpc.FutureTimeoutError as error:
        raise ProviderError("Не удалось подключиться к потоку SpeechKit gRPC") from error
    finally:
        channel.close()
    silence = bytes(16_000 * 2 * PROBE_AUDIO_DURATION_MS // 1_000)
    YandexSpeechKitClient(api_key).recognize_lpcm(
        silence,
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
    )
    return "Соединение со SpeechKit установлено"


def probe_yandex_model(api_key: str, folder_id: str, model: str) -> str:
    client = YandexAIStudioClient(api_key, folder_id)
    models = client.list_models(timeout_seconds=PROBE_TIMEOUT_SECONDS)
    selected = model.strip()
    normalized = client.normalize_model(selected)
    available = {item.id.strip().casefold() for item in models}
    selected_available = normalized.casefold() in available or selected.casefold() in available
    if not selected_available:
        selected_available = any(item.endswith(f"/{selected.casefold()}") for item in available)
    if not selected_available:
        raise ProviderError(f"Выбранная модель недоступна: {selected or 'не указана'}")
    return f"Модель доступна: {selected}"


def probe_audio_source(
    source: str,
    device_id: str | None,
    application_process_id: int,
    *,
    source_factory: Callable[[str, str | None, int], Any] | None = None,
) -> str:
    factory = source_factory or build_probe_source
    pcm_source = factory(source, device_id, application_process_id)
    stop_event = threading.Event()
    chunks = iter(pcm_source.chunks(stop_event))
    try:
        chunk = next(chunks)
    except StopIteration as error:
        raise AudioCaptureError("Источник звука не передал пробный фрагмент") from error
    finally:
        stop_event.set()
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
    if not isinstance(chunk, bytes) or len(chunk) < 2:
        raise AudioCaptureError("Источник звука вернул пустой пробный фрагмент")
    return "Захват звука открывается"


def build_probe_source(source: str, device_id: str | None, application_process_id: int) -> Any:
    config = AudioCaptureConfig(
        sample_rate_hertz=16_000,
        chunk_duration_ms=PROBE_AUDIO_DURATION_MS,
        device_id=device_id,
        process_id=application_process_id or None,
    )
    if source == "remote":
        return ProcessLoopbackPcmSource(application_process_id, config)
    if source == "mic":
        return SoundcardPcmSource(source, config)
    raise AudioCaptureError("Неизвестный источник звука")
