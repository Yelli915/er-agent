"""Model construction shared by the web backend and the benchmark; never touches the network."""

from __future__ import annotations

import importlib
from typing import Any

import httpx

from er_finder.config import Settings


def build_models(settings: Settings, http_client: httpx.Client) -> tuple[Any, Any]:
    """Return the main chat model and the classifier built on its own cheaper model."""
    ChatOpenAI = importlib.import_module("langchain_openai").ChatOpenAI  # noqa: N806
    InputClassifier = importlib.import_module("er_finder.guardrails.triage").InputClassifier  # noqa: N806
    timeout = settings.model_timeout
    model = ChatOpenAI(
        model=settings.main_model,
        reasoning_effort=settings.main_reasoning_effort,
        max_tokens=settings.main_max_output_tokens,
        http_client=http_client,
        timeout=timeout,
        max_retries=0,
    )
    classifier_model = ChatOpenAI(
        model=settings.classifier_model,
        temperature=settings.classifier_temperature,
        max_tokens=settings.classifier_max_tokens,
        http_client=http_client,
        timeout=timeout,
        max_retries=0,
    )
    return model, InputClassifier(classifier_model)
