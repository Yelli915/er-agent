"""Aggregation and orchestration of the live benchmark, with no model or network calls."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from er_finder.evaluation import benchmark
from er_finder.evaluation.benchmark import Prices, Scenario, percentile, run_benchmark, summarize

ROOT = Path(__file__).resolve().parents[3]


def test_percentile_interpolates_and_handles_edges():
    assert percentile([], 50) is None
    assert percentile([5], 90) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(range(1, 11), 90) == pytest.approx(9.1)
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3
    with pytest.raises(ValueError):
        percentile([1], 101)


def _record(elapsed, ok=True, calls=1, cost=0.01, error=None):
    return dict(
        elapsed_seconds=elapsed,
        succeeded=ok,
        model_calls=calls,
        classifier_calls=1,
        tokens_complete=True,
        cost_usd=cost,
        error_type=error,
    )


def test_summary_keeps_failures_in_n_but_not_in_latency():
    rows = [_record(float(i)) for i in range(1, 10)] + [_record(99.0, ok=False, error="Timeout")]
    summary = summarize(rows)
    assert summary["n"] == 10 and summary["succeeded"] == 9
    assert summary["failure_rate"] == pytest.approx(0.1)
    assert summary["latency_p50"] == 5 and summary["latency_max"] == 9
    assert summary["error_types"] == ["Timeout"]
    assert summary["cost_usd_total"] == pytest.approx(0.10)


def test_summary_of_nothing_is_safe():
    summary = summarize([])
    assert summary["n"] == 0 and summary["latency_p50"] is None and summary["failure_rate"] is None


def test_cost_needs_complete_tokens_and_prices():
    prices = Prices(1.0, 4.0, 0.5, 2.0)
    base = dict(
        main_input_tokens=1000,
        main_output_tokens=500,
        classifier_input_tokens=200,
        classifier_output_tokens=50,
        tokens_complete=True,
    )
    assert benchmark.estimate_cost(base, prices) == pytest.approx(
        (1000 * 1 + 500 * 4 + 200 * 0.5 + 50 * 2) / 1e6
    )
    assert benchmark.estimate_cost({**base, "tokens_complete": False}, prices) is None
    assert benchmark.estimate_cost(base, None) is None


def test_prices_parse_validates_shape():
    assert Prices.parse("1,2,3,4").classifier_output == 4
    for bad in ("1,2,3", "1,2,3,-4"):
        with pytest.raises(ValueError):
            Prices.parse(bad)


class FakeRunner:
    def __init__(self, log, fail=False):
        self.log, self.fail, self.closed = log, fail, False

    def chat(self, text):
        self.log.append(text)
        if self.fail:
            raise RuntimeError("boom")
        diagnostics = SimpleNamespace(
            succeeded=True,
            model_calls=1,
            classifier_calls=1,
            input_tokens=100,
            output_tokens=50,
            classifier_input_tokens=20,
            classifier_output_tokens=5,
            tokens_complete=True,
            classifier_tokens_complete=True,
            classifier_elapsed_seconds=0.1,
            error_type=None,
        )
        return SimpleNamespace(reply=SimpleNamespace(hospitals=[1, 2, 3]), diagnostics=diagnostics)

    def close(self):
        self.closed = True


def test_each_run_uses_a_fresh_runner_and_only_the_last_turn_is_measured():
    log, runners = [], []

    def factory():
        runners.append(FakeRunner(log))
        return runners[-1]

    scenario = Scenario("requery", ("first", "second"))
    result = run_benchmark(factory, [scenario], 3, prices=Prices(1, 1, 1, 1))
    assert len(result.runs) == 3 and len(runners) == 3 and all(r.closed for r in runners)
    assert log == ["first", "second"] * 3
    assert all(r.hospitals == 3 and r.cost_usd is not None for r in result.runs)
    assert result.summary()["requery"]["n"] == 3


def test_exception_is_recorded_as_a_failed_run_and_the_runner_is_closed():
    runners = []

    def factory():
        runners.append(FakeRunner([], fail=True))
        return runners[-1]

    result = run_benchmark(factory, [Scenario("standard", ("x",))], 2)
    assert [r.error_type for r in result.runs] == ["RuntimeError", "RuntimeError"]
    assert all(not r.succeeded for r in result.runs) and all(r.closed for r in runners)
    assert result.summary()["standard"]["failure_rate"] == 1


def test_factory_failure_is_one_failed_run_and_later_runs_still_happen():
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 2:
            raise OSError("construction failed")
        return FakeRunner([])

    result = run_benchmark(factory, [Scenario("standard", ("x",))], 3)
    assert [r.error_type for r in result.runs] == [None, "OSError", None]
    assert result.summary()["standard"]["n"] == 3


def test_budget_stops_the_run_and_says_why():
    result = run_benchmark(
        lambda: FakeRunner([]),
        [Scenario("standard", ("x",))],
        10,
        prices=Prices(1000, 1000, 1000, 1000),
        budget_usd=0.05,
    )
    assert result.aborted and len(result.runs) < 10


def test_runs_must_be_positive():
    with pytest.raises(ValueError):
        run_benchmark(lambda: FakeRunner([]), [], 0)


def test_compare_renders_numbers_from_both_summaries():
    def doc(p50, calls):
        return {
            "summary": {
                "standard": summarize([_record(p50), _record(p50)]) | {"model_calls_mean": calls}
            }
        }

    table = benchmark.compare(doc(21.0, 5), doc(8.0, 1))
    assert "| standard | 기준선 | 2 | 0% | 21.0 |" in table
    assert "| standard | 개선 후 | 2 | 0% | 8.0 |" in table


def test_cli_compare_reads_saved_json(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps({"summary": {"standard": summarize([_record(2.0)])}}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--compare", str(path), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0 and "| standard | 기준선 |" in result.stdout


def test_default_scenarios_use_public_coordinates_only():
    assert set(benchmark.SCENARIOS) == {"standard", "urgent", "requery"}
    assert all(benchmark.LOCATION in s.turns[0] for s in benchmark.SCENARIOS.values())


def _result_document(**meta):
    base = {
        "commit": "abc",
        "measured_at": "2026-10-18T00:00:00+00:00",
        "runs_per_scenario": 3,
        "main_model": "m",
        "scenarios": {"standard": ["증상"]},
    }
    return {"meta": base | meta, "runs": [{"scenario": "standard"}], "summary": {"standard": {}}}


def test_check_result_accepts_a_clean_document():
    assert benchmark.check_result(_result_document()) == []


def test_check_result_flags_missing_parts_and_secrets():
    assert benchmark.check_result({"meta": {}, "runs": [], "summary": {}})
    assert benchmark.check_result(_result_document(note="sk-" + "a" * 24))
    assert benchmark.check_result(_result_document(note="Bearer " + "x" * 20))
    assert benchmark.check_result(_result_document(note="serviceKey=" + "Ab1" * 6))
    assert benchmark.check_result(_result_document(note="tok-12345678"), secrets=["tok-12345678"])


def test_cli_check_fails_on_a_leaked_key(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_result_document(note="sk-" + "a" * 24)), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--check", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 1 and "커밋하지 마세요" in result.stderr
