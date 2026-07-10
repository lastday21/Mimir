from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

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

    def stream_chat(self, model: str, messages: list[ChatMessage]) -> Iterator[str]:
        if not model:
            raise ProviderError("Ollama model is not selected")
        body = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    chunk, done = parse_ollama_delta(line)
                    if chunk:
                        yield chunk
                    if done:
                        break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(f"Ollama is not reachable: {error.reason}") from error
        except TimeoutError as error:
            raise ProviderError("Ollama request timed out") from error
        except OSError as error:
            raise ProviderError(f"Ollama request failed: {error}") from error


def parse_ollama_delta(payload: str) -> tuple[str, bool]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "", False
    message = data.get("message")
    chunk = ""
    if isinstance(message, dict):
        chunk = str(message.get("content") or "")
    else:
        chunk = str(data.get("response") or "")
    return chunk, bool(data.get("done"))
