from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, Self

from ..live_trace import trace_live_event
from .base import ProviderError


REALTIME_MODEL = "speech-realtime-250923"
REALTIME_URL = "wss://ai.api.cloud.yandex.net/v1/realtime"


class RealtimeClientProtocol(Protocol):
    async def __aenter__(self) -> Self:
        ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        ...

    async def setup_session(self, instructions: str, sample_rate_hertz: int) -> None:
        ...

    async def append_audio(self, pcm: bytes) -> None:
        ...

    async def add_mic_context(self, text: str) -> None:
        ...

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class YandexRealtimeConfig:
    api_key: str
    folder_id: str
    model: str = REALTIME_MODEL

    @property
    def url(self) -> str:
        folder = self.folder_id.strip()
        if not folder:
            raise ProviderError("Yandex folder ID is required for Realtime API")
        return f"{REALTIME_URL}?model=gpt://{folder}/{self.model}"


class YandexRealtimeClient:
    def __init__(self, config: YandexRealtimeConfig) -> None:
        key = config.api_key.strip()
        if not key:
            raise ProviderError("Yandex AI Studio API key is not configured")
        self.config = config
        self._session: Any | None = None
        self._ws: Any | None = None

    async def __aenter__(self) -> Self:
        try:
            import aiohttp
        except ImportError as error:
            raise ProviderError("Yandex Realtime API requires `aiohttp`. Reinstall with `pip install -e .`.") from error

        self._session = aiohttp.ClientSession()
        trace_live_event("realtime.connect", model=self.config.model, folderId=self.config.folder_id)
        self._ws = await self._session.ws_connect(
            self.config.url,
            headers={"Authorization": f"Api-Key {self.config.api_key}"},
            heartbeat=20.0,
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        trace_live_event("realtime.disconnect", excType=str(exc_type) if exc_type else "")
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def setup_session(self, instructions: str, sample_rate_hertz: int) -> None:
        trace_live_event(
            "realtime.session.update",
            model=self.config.model,
            sampleRateHertz=sample_rate_hertz,
            outputModalities=["text"],
            turnDetection={"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 400},
            instructions=instructions,
        )
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "instructions": instructions,
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": sample_rate_hertz,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "silence_duration_ms": 400,
                            },
                        },
                    },
                },
            }
        )

    async def append_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        trace_live_event("realtime.out.input_audio_buffer.append", bytes=len(pcm))
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def add_mic_context(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        trace_live_event("realtime.out.conversation.item.create", source="mic", text=normalized)
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"MIC_CONTEXT: Пользователь ответил собеседнику: {normalized}",
                        }
                    ],
                    "metadata": {
                        "source": "mic",
                        "purpose": "context_only",
                    },
                },
            }
        )

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        try:
            import aiohttp
        except ImportError as error:
            raise ProviderError("Yandex Realtime API requires `aiohttp`. Reinstall with `pip install -e .`.") from error

        ws = self._require_ws()
        async for message in ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                payload = json.loads(message.data)
                trace_live_event("realtime.in.event", **trace_realtime_payload(payload))
                yield payload
            elif message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                trace_live_event("realtime.closed", messageType=str(message.type))
                return

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._require_ws().send_json(payload)

    def _require_ws(self) -> Any:
        if self._ws is None:
            raise ProviderError("Yandex Realtime websocket is not connected")
        return self._ws


def trace_realtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(payload.get("type") or "")
    item: dict[str, Any] = {"type": event_type}
    for key in ("transcript", "delta"):
        if key in payload:
            item[key] = payload[key]
    if event_type == "error":
        item["error"] = payload.get("error")
    return item
