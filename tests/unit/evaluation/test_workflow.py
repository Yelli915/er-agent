"""The comparison must isolate reply execution from input-model variation."""

import json

from scripts import eval_workflow

from er_finder.guardrails.input_guard import assess_input
from er_finder.guardrails.triage import InputClassifier
from er_finder.usage import ClassifierUsage


def test_frozen_classifier_calls_source_once_and_returns_detached_assessments():
    class ChangingClassifier:
        calls = 0
        last_usage = ClassifierUsage(calls=1)

        def assess(self, text):
            self.calls += 1
            result = assess_input(text)
            result.location_query = str(self.calls)
            return result

    source = ChangingClassifier()
    frozen = eval_workflow.FrozenClassifier(source)
    first = frozen.assess("same input")
    first.location_query = "mutated by first execution"
    second = frozen.assess("same input")
    assert source.calls == frozen.acquisition_calls == 1
    assert second.location_query == "1"
    assert frozen.last_usage.calls == 0


def test_workflow_cli_compares_both_paths_with_fixed_inputs(tmp_path, monkeypatch):
    output = tmp_path / "workflow.json"
    monkeypatch.setattr("sys.argv", ["eval_workflow", "--output", str(output)])
    assert eval_workflow.main() == 0
    report = json.loads(output.read_text("utf-8"))
    assert report["passed"]
    assert all(row["same_reply"] for row in report["cases"])
    assert all(
        row["variants"][1]["graph_model_calls"] == 0 for row in report["cases"]
    )


def test_offline_parsing_does_not_claim_ai_was_evaluated():
    report = eval_workflow.evaluate_parsing(InputClassifier(), False)
    assert report["live_status"] == "NOT_RUN"
    assert report["passed"] is None
    assert all(len(row["variants"]) == 1 for row in report["cases"])
