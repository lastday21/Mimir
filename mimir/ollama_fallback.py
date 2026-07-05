from __future__ import annotations

from .models import ModelInfo


def select_preferred_model(models: list[ModelInfo]) -> ModelInfo | None:
    if not models:
        return None
    return max(models, key=model_rank)


def sort_models(models: list[ModelInfo]) -> list[ModelInfo]:
    return sorted(models, key=model_rank, reverse=True)


def is_recommended_model(model_name: str) -> bool:
    search_text = model_name.lower()
    return "qwen3" in search_text and (
        contains_model_size(search_text, "4b") or contains_model_size(search_text, "8b")
    )


def model_rank(model: ModelInfo) -> tuple[int, int, int]:
    search_text = f"{model.id} {model.name}".lower()
    family_score = family_rank(search_text)
    size_score = local_size_score(search_text)
    context_score = min(model.context_window or 0, 128_000)
    return family_score + size_score, size_score, context_score


def family_rank(search_text: str) -> int:
    if "qwen3" in search_text:
        return 500
    if "qwen2.5" in search_text or "qwen2" in search_text:
        return 420
    if "qwen" in search_text:
        return 380
    if "llama3.2" in search_text:
        return 240
    if "llama3" in search_text:
        return 220
    if any(name in search_text for name in ("mistral", "gemma", "phi")):
        return 180
    return 100


def local_size_score(search_text: str) -> int:
    if contains_model_size(search_text, "8b"):
        return 90
    if contains_model_size(search_text, "4b"):
        return 80
    if contains_model_size(search_text, "7b"):
        return 70
    if contains_model_size(search_text, "14b"):
        return 50
    if contains_model_size(search_text, "3b"):
        return 45
    if contains_model_size(search_text, "1.5b"):
        return 25
    if contains_model_size(search_text, "32b"):
        return 20
    return 35


def contains_model_size(search_text: str, size: str) -> bool:
    return any(f"{separator}{size}" in search_text for separator in (":", "-", "_", " "))
