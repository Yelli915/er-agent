"""Regression guard for triage quality on the labeled golden set (offline, rules only).

The rule engine is a keyword matcher; the cases below are sentences it is known to miss and
that only the classifier model can catch. They are frozen so a *new* miss fails the build and
a fixed miss forces this list to shrink. Live classifier quality is measured with
scripts/eval_golden.py --live, not here.
"""

import pytest

from er_finder.evaluation import golden
from er_finder.guardrails.input_guard import assess_input

KNOWN_RULE_UNDER_TRIAGE = {
    "crit-13", "crit-14", "crit-18", "crit-19", "crit-20", "urg-07", "urg-08",
}  # fmt: skip
KNOWN_RULE_WRONG_CONDITION = {"crit-06", "crit-10", "crit-13"}
KNOWN_RULE_MISSED_BLOCKS = {"blk-06"}


@pytest.fixture(scope="module")
def cases():
    return golden.load_cases()


@pytest.fixture(scope="module")
def report(cases):
    return golden.evaluate(assess_input, cases)


def test_golden_set_is_well_formed(cases):
    severities = {c["severity"] for c in cases if not c.get("blocked")}
    assert severities == {"standard", "urgent", "critical"}
    assert sum(1 for c in cases if c["severity"] == "critical") >= 20
    assert any(c.get("blocked") for c in cases)


def test_no_new_under_triage(report):
    new = set(report["under_triage"]) - KNOWN_RULE_UNDER_TRIAGE
    assert not new, f"새로 과소 분류된 사례: {sorted(new)}"


def test_known_gaps_are_removed_once_fixed(report):
    fixed = (
        (KNOWN_RULE_UNDER_TRIAGE - set(report["under_triage"]))
        | (KNOWN_RULE_WRONG_CONDITION - set(report["wrong_condition"]))
        | (KNOWN_RULE_MISSED_BLOCKS - set(report["missed_blocks"]))
    )
    assert not fixed, f"고쳐진 사례는 이 파일의 목록에서 제거하세요: {sorted(fixed)}"


def test_no_new_wrong_condition_or_missed_block(report):
    assert set(report["wrong_condition"]) <= KNOWN_RULE_WRONG_CONDITION
    assert set(report["missed_blocks"]) <= KNOWN_RULE_MISSED_BLOCKS


def test_legitimate_requests_are_never_blocked(report):
    assert report["false_blocks"] == []


def test_evaluate_counts_crashes_as_under_triage():
    def broken(_):
        raise RuntimeError("x")

    cases = [{"id": "c1", "text": "t", "severity": "critical"}]
    result = golden.evaluate(broken, cases)
    assert result["errors"] == ["c1"] and result["under_triage"] == ["c1"]
    assert result["critical_recall"] == 0


def test_evaluate_separates_under_from_over_triage():
    from types import SimpleNamespace

    def fake(text):
        sev = {"a": "standard", "b": "critical"}[text]
        return SimpleNamespace(triage=SimpleNamespace(severity=sev, condition=None), blocked=False)

    cases = [
        {"id": "u", "text": "a", "severity": "urgent"},
        {"id": "o", "text": "b", "severity": "urgent"},
    ]
    result = golden.evaluate(fake, cases)
    assert result["under_triage"] == ["u"] and result["over_triage"] == ["o"]
