"""Repeated latency/cost measurement. Pure aggregation lives here; live wiring is injected."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

# Public coordinates (Gangnam Station) and synthetic symptoms only; no real user data.
LOCATION = "37.4979, 127.0276"


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[str, ...]
    # Only the last turn is measured; earlier turns warm the session (cache, symptom context).
    description: str = ""


SCENARIOS: dict[str, Scenario] = {
    "standard": Scenario(
        "standard",
        (f"{LOCATION} 근처에서 발목이 아파요",),
        "일반 증상 한 번 조회",
    ),
    "urgent": Scenario(
        "urgent",
        (f"{LOCATION} 근처에서 가슴이 답답하고 식은땀이 나요",),
        "중증 수용 조회(심근경색 조건)",
    ),
    "requery": Scenario(
        "requery",
        (f"{LOCATION} 근처에서 발목이 아파요", "같은 위치에서 다시 보여줘"),
        "같은 세션 재질의(두 번째 턴만 측정)",
    ),
}


@dataclass(frozen=True)
class Prices:
    """USD per one million tokens."""

    main_input: float
    main_output: float
    classifier_input: float
    classifier_output: float

    @classmethod
    def parse(cls, text: str) -> Prices:
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4 or any(p < 0 for p in parts):
            raise ValueError("가격은 main_in,main_out,cls_in,cls_out (USD/1M 토큰) 4개여야 합니다.")
        return cls(*parts)


@dataclass
class RunRecord:
    scenario: str
    run: int
    elapsed_seconds: float
    succeeded: bool
    hospitals: int
    model_calls: int
    classifier_calls: int
    main_input_tokens: int | None
    main_output_tokens: int | None
    classifier_input_tokens: int | None
    classifier_output_tokens: int | None
    tokens_complete: bool
    classifier_elapsed_seconds: float
    cost_usd: float | None
    error_type: str | None = None


def estimate_cost(record: dict[str, Any] | RunRecord, prices: Prices | None) -> float | None:
    data = asdict(record) if isinstance(record, RunRecord) else record
    keys = (
        "main_input_tokens",
        "main_output_tokens",
        "classifier_input_tokens",
        "classifier_output_tokens",
    )
    if prices is None or not data["tokens_complete"]:
        return None
    t = {k: data.get(k) or 0 for k in keys}
    return (
        t["main_input_tokens"] * prices.main_input
        + t["main_output_tokens"] * prices.main_output
        + t["classifier_input_tokens"] * prices.classifier_input
        + t["classifier_output_tokens"] * prices.classifier_output
    ) / 1_000_000


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (same as numpy's default)."""
    data = sorted(values)
    if not data:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be within 0..100")
    pos = (len(data) - 1) * q / 100
    low, high = math.floor(pos), math.ceil(pos)
    return data[low] + (data[high] - data[low]) * (pos - low)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one scenario. Failed runs stay in n and the failure rate, never dropped.

    Latency percentiles cover successful runs only: a failed run's elapsed time measures the
    failure, not the search. The failure rate is reported next to them so nothing is hidden.
    """
    n = len(records)
    ok = [r for r in records if r["succeeded"]]
    elapsed = [r["elapsed_seconds"] for r in ok]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    return {
        "n": n,
        "succeeded": len(ok),
        "failure_rate": (n - len(ok)) / n if n else None,
        "latency_p50": percentile(elapsed, 50),
        "latency_p90": percentile(elapsed, 90),
        "latency_mean": _mean(elapsed),
        "latency_min": min(elapsed) if elapsed else None,
        "latency_max": max(elapsed) if elapsed else None,
        "model_calls_mean": _mean([r["model_calls"] for r in ok]),
        "classifier_calls_mean": _mean([r["classifier_calls"] for r in ok]),
        "tokens_complete_runs": sum(1 for r in records if r["tokens_complete"]),
        "cost_usd_mean": _mean(costs),
        "cost_usd_total": sum(costs) if costs else None,
        "error_types": sorted({r["error_type"] for r in records if r.get("error_type")}),
    }


def measure_turn(runner: Any, text: str) -> tuple[Any, float]:
    from time import perf_counter

    started = perf_counter()
    result = runner.chat(text)
    return result, perf_counter() - started


def record_from_result(
    scenario: str, run: int, result: Any, elapsed: float, prices: Prices | None
) -> RunRecord:
    d = result.diagnostics
    record = RunRecord(
        scenario=scenario,
        run=run,
        elapsed_seconds=elapsed,
        succeeded=bool(d and d.succeeded),
        hospitals=len(result.reply.hospitals),
        model_calls=d.model_calls if d else 0,
        classifier_calls=d.classifier_calls if d else 0,
        main_input_tokens=d.input_tokens if d else None,
        main_output_tokens=d.output_tokens if d else None,
        classifier_input_tokens=d.classifier_input_tokens if d else None,
        classifier_output_tokens=d.classifier_output_tokens if d else None,
        tokens_complete=bool(d and d.tokens_complete and d.classifier_tokens_complete),
        classifier_elapsed_seconds=d.classifier_elapsed_seconds if d else 0.0,
        cost_usd=None,
        error_type=d.error_type if d else None,
    )
    record.cost_usd = estimate_cost(record, prices)
    return record


@dataclass
class BenchmarkResult:
    runs: list[RunRecord] = field(default_factory=list)
    aborted: str | None = None

    def summary(self) -> dict[str, Any]:
        names = dict.fromkeys(r.scenario for r in self.runs)
        return {
            name: summarize([asdict(r) for r in self.runs if r.scenario == name]) for name in names
        }


def run_benchmark(
    runner_factory: Callable[[], Any],
    scenarios: list[Scenario],
    runs: int,
    *,
    prices: Prices | None = None,
    budget_usd: float | None = None,
    on_record: Callable[[RunRecord], None] | None = None,
) -> BenchmarkResult:
    """Run every scenario `runs` times, each time on a fresh runner (new session, empty cache)."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    result = BenchmarkResult()
    spent = 0.0
    for scenario in scenarios:
        for index in range(1, runs + 1):
            runner = None
            warmup_cost = 0.0  # warm-up turns are paid for even though they are not measured
            try:
                runner = runner_factory()
                for text in scenario.turns[:-1]:
                    warmup = runner.chat(text)
                    warmup_record = record_from_result(scenario.name, index, warmup, 0.0, prices)
                    warmup_cost += warmup_record.cost_usd or 0.0
                chat_result, elapsed = measure_turn(runner, scenario.turns[-1])
            except Exception as error:
                spent += warmup_cost
                result.runs.append(
                    RunRecord(
                        scenario.name, index, 0.0, False, 0, 0, 0, None, None, None, None,
                        False, 0.0, None, type(error).__name__,
                    )
                )  # fmt: skip
                if budget_usd is not None and spent > budget_usd:
                    result.aborted = f"budget exceeded: ${spent:.4f} > ${budget_usd:.4f}"
                    return result
                continue
            finally:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
            record = record_from_result(scenario.name, index, chat_result, elapsed, prices)
            result.runs.append(record)
            if on_record:
                on_record(record)
            spent += warmup_cost + (record.cost_usd or 0.0)
            if budget_usd is not None and spent > budget_usd:
                result.aborted = f"budget exceeded: ${spent:.4f} > ${budget_usd:.4f}"
                return result
    return result


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(
        r"(?i)(service[_-]?key|api[_-]?key|authorization)\"?\s*[:=]\s*\"?[A-Za-z0-9%+/=_-]{12,}"
    ),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
)
_REQUIRED_META = ("commit", "measured_at", "runs_per_scenario", "main_model", "scenarios")


def check_result(document: dict[str, Any], secrets: Iterable[str] = ()) -> list[str]:
    """Problems that make a result file unsafe or unusable to commit; empty means fine.

    `secrets` are literal values (e.g. the current environment's keys) that must not appear.
    """
    problems = [f"meta.{key} 누락" for key in _REQUIRED_META if key not in document.get("meta", {})]
    for key in ("runs", "summary"):
        if not document.get(key):
            problems.append(f"{key} 비어 있음")
    text = json.dumps(document, ensure_ascii=False)
    problems += [f"비밀값 패턴 발견: {p.pattern[:20]}…" for p in _SECRET_PATTERNS if p.search(text)]
    problems += ["환경 변수의 비밀값이 포함됨" for v in secrets if len(v) >= 8 and v in text]
    return problems


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def compare(baseline: dict[str, Any], after: dict[str, Any]) -> str:
    """Markdown table of two benchmark JSON files; every number comes from their summaries."""
    lines = [
        "| 시나리오 | 구분 | n | 실패율 | P50(초) | P90(초) | 메인 호출 | 분류기 호출 "
        "| 턴당 비용(USD) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in dict.fromkeys([*baseline["summary"], *after["summary"]]):
        for label, doc in (("기준선", baseline), ("개선 후", after)):
            s = doc["summary"].get(name)
            if s is None:
                continue
            lines.append(
                f"| {name} | {label} | {s['n']} | {_fmt(s['failure_rate'] * 100, 0)}% "
                f"| {_fmt(s['latency_p50'])} | {_fmt(s['latency_p90'])} "
                f"| {_fmt(s['model_calls_mean'])} | {_fmt(s['classifier_calls_mean'])} "
                f"| {_fmt(s['cost_usd_mean'], 5)} |"
            )
    return "\n".join(lines)
