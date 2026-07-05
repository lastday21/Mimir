from __future__ import annotations

from ..models import ChatMessage, ModelInfo
from .base import ProviderError
from .http import request_json


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[ModelInfo]:
        data = request_json("GET", f"{self.base_url}/api/tags", timeout=5)
        models = data.get("models", [])
        if not isinstance(models, list):
            return []
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            context = details.get("context_length")
            result.append(ModelInfo(id=name, name=name, provider="ollama", context_window=context if isinstance(context, int) else None))
        return result

    def chat(self, model: str, messages: list[ChatMessage]) -> str:
        if not model:
            raise ProviderError("Ollama model is not selected")
        data = request_json(
            "POST",
            f"{self.base_url}/api/chat",
            body={
                "model": model,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "stream": False,
            },
            timeout=120,
        )
        message = data.get("message")
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(data.get("response") or "")
