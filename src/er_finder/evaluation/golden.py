"""Labeled-case evaluation of input triage. Under-triage is the error that matters most."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

RANK = {"standard": 0, "urgent": 1, "critical": 2}
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "golden" / "triage.json"


def load_cases(path: Path | str = DEFAULT_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("golden case ids must be unique")
    for case in cases:
        if case["severity"] not in RANK:
            raise ValueError(f"{case['id']}: unknown severity {case['severity']!r}")
    return cases


def evaluate(assess: Callable[[str], Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score `assess(text)` (returns an InputAssessment) against labeled cases.

    Triage metrics cover only cases that should be searched; blocked cases are scored on
    whether they were blocked. Each metric lists the failing case ids so a regression is
    traceable to a sentence rather than a percentage.
    """
    triage_cases = [c for c in cases if not c.get("blocked")]
    blocked_cases = [c for c in cases if c.get("blocked")]
    confusion: Counter[tuple[str, str]] = Counter()
    under, over, wrong_condition, errors = [], [], [], []
    exact = 0
    false_block = []
    for case in triage_cases:
        try:
            result = assess(case["text"])
        except Exception:  # a crash is a failure of this case, not of the run
            errors.append(case["id"])
            under.append(case["id"])
            if case.get("condition"):  # no condition was produced, so it cannot be right
                wrong_condition.append(case["id"])
            confusion[(case["severity"], "error")] += 1
            continue
        predicted = result.triage.severity
        confusion[(case["severity"], predicted)] += 1
        if result.blocked:
            false_block.append(case["id"])
        if predicted == case["severity"]:
            exact += 1
        elif RANK[predicted] < RANK[case["severity"]]:
            under.append(case["id"])
        else:
            over.append(case["id"])
        expected_condition = case.get("condition")
        if expected_condition and result.triage.condition != expected_condition:
            wrong_condition.append(case["id"])
    critical = [c for c in triage_cases if c["severity"] == "critical"]
    under_ids = set(under)
    critical_missed = [c["id"] for c in critical if c["id"] in under_ids]
    with_condition = [c for c in triage_cases if c.get("condition")]
    missed_blocks = []
    for case in blocked_cases:
        try:
            if not assess(case["text"]).blocked:
                missed_blocks.append(case["id"])
        except Exception:
            missed_blocks.append(case["id"])
    n = len(triage_cases)
    return {
        "n_triage": n,
        "n_blocked_cases": len(blocked_cases),
        "severity_accuracy": exact / n if n else None,
        "under_triage": sorted(under),
        "under_triage_rate": len(under) / n if n else None,
        "over_triage": sorted(over),
        "critical_recall": 1 - len(critical_missed) / len(critical) if critical else None,
        "critical_missed": sorted(critical_missed),
        "condition_accuracy": (
            1 - len(wrong_condition) / len(with_condition) if with_condition else None
        ),
        "wrong_condition": sorted(wrong_condition),
        "block_recall": (1 - len(missed_blocks) / len(blocked_cases) if blocked_cases else None),
        "missed_blocks": sorted(missed_blocks),
        "false_blocks": sorted(false_block),
        "errors": sorted(errors),
        "confusion": {f"{a}->{b}": count for (a, b), count in sorted(confusion.items())},
    }
