"""Standalone classification must not construct the unused main chat model."""

from types import SimpleNamespace

from er_finder.agent import factory
from er_finder.config import Settings


def test_build_classifier_constructs_only_the_classifier_model(monkeypatch):
    calls = []

    def chat_model(**kwargs):
        calls.append(kwargs)
        return object()

    original = factory.importlib.import_module
    monkeypatch.setattr(factory.importlib, "import_module", lambda name: (
        SimpleNamespace(ChatGoogleGenerativeAI=chat_model)
        if name == "langchain_google_genai" else original(name)
    ))
    settings = Settings()
    classifier = factory.build_classifier(settings)
    assert classifier is not None
    assert [call["model"] for call in calls] == [settings.classifier_model]
