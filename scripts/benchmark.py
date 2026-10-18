"""Repeated live benchmark (opt-in) and comparison of saved results.

Live runs call OpenAI and E-Gen and cost money, so nothing runs without --live.
    python scripts/benchmark.py --live --runs 20 --output docs/benchmarks/baseline.json
    python scripts/benchmark.py --compare \
        docs/benchmarks/baseline.json docs/benchmarks/after.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er_finder.evaluation import benchmark  # noqa: E402

BLOCKED = (
    "BLOCKED: 실제 OpenAI·E-Gen을 호출하며 비용이 발생하므로 --live 없이는 실행하지 않습니다. "
    "P50/P90 지연, 모델 호출 수, 토큰, 비용은 python scripts/benchmark.py --live --runs 20 으로 "
    "측정하고, 오프라인 확인은 python scripts/verify.py 를 사용하세요."
)
REQUIRED_KEYS = ("OPENAI_API_KEY", "EGEN_SERVICE_KEY")


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def _live_factory(server_planned: bool):
    """Build runners the way the web backend does, so both measure the same pipeline."""
    from er_finder.config import Settings
    from er_finder.web.live import default_runner_factory, load_environment

    load_environment()
    return (lambda: default_runner_factory(server_planned_calls=server_planned)), Settings()


def run_live(args) -> int:
    from er_finder.web.live import credential_status

    credential_status()  # loads the project .env without overriding the environment
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k, "").strip()]
    if missing:
        print(f"BLOCKED: 필요한 키가 설정되지 않았습니다: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        prices = benchmark.Prices.parse(args.prices) if args.prices else None
    except ValueError as error:
        print(f"--prices 형식 오류: {error}", file=sys.stderr)
        return 2
    if args.budget_usd is not None and prices is None:
        print("--budget-usd 를 쓰려면 --prices 가 필요합니다.", file=sys.stderr)
        return 2
    scenarios = [benchmark.SCENARIOS[name] for name in args.scenario]
    factory, settings = _live_factory(not args.no_server_planned)

    def progress(record):
        status = "ok" if record.succeeded else f"FAIL({record.error_type})"
        print(
            f"[{record.scenario} #{record.run}] {record.elapsed_seconds:.1f}s "
            f"model={record.model_calls} cls={record.classifier_calls} {status}",
            file=sys.stderr,
        )

    result = benchmark.run_benchmark(
        factory,
        scenarios,
        args.runs,
        prices=prices,
        budget_usd=args.budget_usd,
        on_record=progress,
    )
    document = {
        "meta": {
            "commit": _commit(),
            "label": args.label,
            "measured_at": datetime.now(UTC).isoformat(),
            "runs_per_scenario": args.runs,
            "python": platform.python_version(),
            "main_model": settings.main_model,
            "reasoning_effort": settings.main_reasoning_effort,
            "classifier_model": settings.classifier_model,
            "server_planned_calls": not args.no_server_planned,
            "prices_usd_per_million": asdict(prices) if prices else None,
            "aborted": result.aborted,
            "scenarios": {s.name: s.turns for s in scenarios},
        },
        "runs": [asdict(r) for r in result.runs],
        "summary": result.summary(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}", file=sys.stderr)
    return 1 if result.aborted else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="실제 API를 호출해 측정(비용 발생)")
    parser.add_argument("--runs", type=int, default=20, help="시나리오당 반복 횟수(기본 20)")
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=sorted(benchmark.SCENARIOS),
        default=list(benchmark.SCENARIOS),
    )
    parser.add_argument("--output", default="benchmark-live.json")
    parser.add_argument("--label", default="", help="결과에 붙일 이름(baseline, after 등)")
    parser.add_argument(
        "--no-server-planned",
        action="store_true",
        help="기준선 측정: 서버가 정한 검색 단계에서도 모델을 호출(개선 전 동작)",
    )
    parser.add_argument("--prices", help="USD/1M 토큰: main_in,main_out,cls_in,cls_out")
    parser.add_argument("--budget-usd", type=float, help="누적 비용이 넘으면 중단(--prices 필요)")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "AFTER"))
    parser.add_argument(
        "--check", metavar="JSON", help="결과 파일에 비밀값이 없고 필수 항목이 있는지 커밋 전 점검"
    )
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Korean output must survive a Windows codepage
    if args.check:
        document = json.loads(Path(args.check).read_text(encoding="utf-8"))
        secrets = [os.environ[k] for k in REQUIRED_KEYS if os.environ.get(k)]
        problems = benchmark.check_result(document, secrets)
        for problem in problems:
            print(f"문제: {problem}", file=sys.stderr)
        print("OK" if not problems else "커밋하지 마세요.", file=sys.stderr)
        return 1 if problems else 0
    if args.compare:
        docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare]
        print(benchmark.compare(*docs))
        return 0
    if not args.live:
        print(BLOCKED, file=sys.stderr)
        return 2
    if args.runs < 1:
        print("--runs 는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
