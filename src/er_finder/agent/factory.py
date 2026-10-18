"""Model and runner construction shared by the web backend and the benchmark.

Importing this module never touches the network; only the factories do."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
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


def default_runner_factory(**kwargs: Any) -> Any:
    """Construct the real model, classifier and runner only after explicit connection."""
    runner_module = importlib.import_module("er_finder.agent.runner")
    factory_module = importlib.import_module("er_finder.agent.factory")
    provider_module = importlib.import_module("er_finder.web.provider")
    settings = Settings()
    http_client = httpx.Client(timeout=settings.model_timeout)
    provider = None
    try:
        provider = provider_module.LiveProvider(kakao_key=os.environ.get("KAKAO_REST_API_KEY", ""))
        model, classifier = factory_module.build_models(settings, http_client)
        runner = runner_module.ERFinder(
            provider=provider, model=model, classifier=classifier, **kwargs
        )
        return ManagedRunner(runner, provider, http_client)
    except Exception:
        try:
            if provider is not None:
                provider.close()
        finally:
            http_client.close()
        raise


class ManagedRunner:
    """Close only explicitly owned resources; leave SDK private clients untouched."""

    def __init__(self, runner: Any, provider: Any, http_client: httpx.Client):
        self._runner, self._provider, self._http_client = runner, provider, http_client

    def chat(self, text: str, *, force_refresh: bool = False) -> Any:
        return self._runner.chat(text, force_refresh=force_refresh)

    def approve(self, approved: bool) -> Any:
        return self._runner.approve(approved)

    def set_progress(self, listener: Callable[[str], None] | None) -> None:
        self._runner.on_progress = listener

    def end_session(self) -> None:
        try:
            self._runner.end_session()
        finally:
            self._provider.clear_cache()

    def close(self) -> None:
        try:
            self.end_session()
        finally:
            try:
                self._provider.close()
            finally:
                self._http_client.close()
