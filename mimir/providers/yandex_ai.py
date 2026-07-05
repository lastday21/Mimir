from __future__ import annotations

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

    def normalize_model(self, model: str) -> str:
        model = model.strip()
        if "://" in model:
            return model
        return f"gpt://{self.folder_id}/{model}"
