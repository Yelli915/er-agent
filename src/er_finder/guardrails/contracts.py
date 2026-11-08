"""Input assessment boundary shared by rule-only and model-backed classifiers."""

from typing import Protocol

from er_finder.guardrails.input_guard import InputAssessment


class Classifier(Protocol):
    def assess(self, text: str) -> InputAssessment: ...
