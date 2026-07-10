from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from ..models import ChatMessage, ModelInfo
from .base import ProviderError
from .http import request_json


class YandexAIStudioClient:
    def __init__(
        self,
        api_key: str,
        folder_id: str,
        base_url: str = "https://ai.api.cloud.yandex.net/v1",
    ) -> None:
        self.api_key = api_key.strip()
        self.folder_id = folder_id.strip()
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("Yandex AI Studio API key is missing")
        if not self.folder_id:
            raise ProviderError("Yandex Cloud folder ID is missing")
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
            "x-project": self.folder_id,
            "OpenAI-Project": self.folder_id,
        }

    def list_models(self) -> list[ModelInfo]:
        data = request_json("GET", f"{self.base_url}/models", headers=self._headers(), timeout=30)
        raw_models = data.get("data", [])
        if not isinstance(raw_models, list):
            return []
        models: list[ModelInfo] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            name = str(item.get("name") or model_id)
            context = item.get("context_window") or item.get("context_length")
            models.append(
                ModelInfo(
                    id=model_id,
                    name=name,
                    provider="yandex_ai_studio",
                    context_window=context if isinstance(context, int) else None,
                )
            )
        return models

    def chat(self, model: str, messages: list[ChatMessage]) -> str:
        model_id = self.normalize_model(model)
        data = request_json(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            body={
                "model": model_id,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "stream": False,
            },
            timeout=120,
        )
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
        raise ProviderError("Yandex AI Studio returned no assistant message")

    def stream_chat(self, model: str, messages: list[ChatMessage]) -> Iterator[str]:
        model_id = self.normalize_model(model)
        body = {
            "model": model_id,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": True,
        }
        headers = {
            **self._headers(),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-data-logging-enabled": "false",
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        break
                    chunk = parse_openai_delta(payload)
                    if chunk:
                        yield chunk
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Yandex AI Studio returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(f"Yandex AI Studio is not reachable: {error.reason}") from error
        except TimeoutError as error:
            raise ProviderError("Yandex AI Studio request timed out") from error
        except OSError as error:
            raise ProviderError(f"Yandex AI Studio request failed: {error}") from error

    def normalize_model(self, model: str) -> str:
        model = model.strip()
        if "://" in model:
            return model
        return f"gpt://{self.folder_id}/{model}"


def parse_openai_delta(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        return str(delta.get("content") or "")
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""
