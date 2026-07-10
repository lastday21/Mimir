from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from .models import ChatMessage
from .ollama_fallback import select_preferred_model
from .providers.base import ProviderError
from .session_types import AnswerStreamChunk


class AnswerProviderGateway:
    def __init__(
        self,
        load_config: Callable[[], Any],
        read_secret: Callable[[str], str | None],
        yandex_client: type[Any],
        ollama_client: type[Any],
        trace_event: Callable[..., None],
    ) -> None:
        self.load_config = load_config
        self.read_secret = read_secret
        self.yandex_client = yandex_client
        self.ollama_client = ollama_client
        self.trace_event = trace_event

    def provider_name(self, override: str | None = None, config: Any | None = None) -> str:
        if override:
            return override
        try:
            return (config or self.load_config()).llm_provider
        except Exception:
            return "unknown"

    def stream(
        self,
        messages: list[ChatMessage],
        override: str | None = None,
    ) -> Iterator[AnswerStreamChunk]:
        config = self.load_config()
        provider = self.provider_name(override, config)
        if provider == "ollama":
            client = self.ollama_client(config.ollama_base_url)
            model = self.ollama_model(config, client, override)
            for chunk in client.stream_chat(model, messages):
                yield AnswerStreamChunk(chunk, provider="ollama")
            return

        key = self.read_secret("yandex_ai_studio") or ""
        primary_started = False
        try:
            client = self.yandex_client(key, config.yandex_folder_id)
            for chunk in client.stream_chat(config.llm_model, messages):
                if chunk:
                    primary_started = True
                yield AnswerStreamChunk(chunk, provider="yandex_ai_studio")
        except ProviderError as error:
            if primary_started:
                raise
            yield from self.stream_ollama_fallback(messages, error)

    def stream_ollama_fallback(
        self,
        messages: list[ChatMessage],
        primary_error: ProviderError,
    ) -> Iterator[AnswerStreamChunk]:
        config = self.load_config()
        client = self.ollama_client(config.ollama_base_url)
        try:
            preferred = select_preferred_model(client.list_models())
        except ProviderError as error:
            raise ProviderError(
                f"Yandex AI Studio failed: {primary_error}. Ollama fallback failed: {error}"
            ) from error
        if preferred is None:
            raise ProviderError(
                f"Yandex AI Studio failed: {primary_error}. Ollama fallback has no local models"
            )
        reason = str(primary_error)
        self.trace_event(
            "answer.fallback",
            fromProvider="yandex_ai_studio",
            toProvider="ollama",
            model=preferred.id,
            reason=reason,
        )
        try:
            for chunk in client.stream_chat(preferred.id, messages):
                yield AnswerStreamChunk(
                    chunk,
                    provider="ollama",
                    fallback_used=True,
                    fallback_reason=reason,
                )
        except ProviderError as error:
            raise ProviderError(
                f"Yandex AI Studio failed: {primary_error}. Ollama fallback failed: {error}"
            ) from error

    def ollama_model(self, config: Any, client: Any, override: str | None) -> str:
        forced_ollama = override == "ollama" and config.llm_provider != "ollama"
        if not forced_ollama:
            return str(config.llm_model)
        preferred = select_preferred_model(client.list_models())
        if preferred is None:
            raise ProviderError("Ollama fallback has no local models")
        return preferred.id
