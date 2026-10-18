"""Score input triage against the labeled golden set.

python scripts/eval_golden.py              # rule engine only, offline
python scripts/eval_golden.py --live       # rules + the real classifier model (small cost)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er_finder import golden  # noqa: E402
from er_finder.guardrails.input_guard import assess_input  # noqa: E402


def _print(name: str, report: dict) -> None:
    print(f"== {name} ==")
    for key in ("severity_accuracy", "critical_recall", "condition_accuracy", "block_recall"):
        value = report[key]
        print(f"{key:20s} {'-' if value is None else f'{value:.3f}'}")
    for key in ("under_triage", "over_triage", "wrong_condition", "missed_blocks", "false_blocks"):
        print(f"{key:20s} {report[key] or '-'}")
    print(f"confusion            {report['confusion']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="실제 분류기 모델 호출(소액 비용)")
    parser.add_argument("--cases", default=str(golden.DEFAULT_PATH))
    parser.add_argument("--output", help="결과 JSON 저장 경로")
    args = parser.parse_args(argv)
    cases = golden.load_cases(args.cases)
    reports = {"rules": golden.evaluate(assess_input, cases)}
    _print("rules", reports["rules"])
    if args.live:
        from er_finder.web.live import credential_status

        credential_status()
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            print("BLOCKED: OPENAI_API_KEY 가 설정되지 않았습니다.", file=sys.stderr)
            return 2
        import httpx

        from er_finder.agent.factory import build_models
        from er_finder.config import Settings

        settings = Settings()
        with httpx.Client(timeout=settings.model_timeout) as client:
            _, classifier = build_models(settings, client)
            reports["rules+classifier"] = golden.evaluate(classifier.assess, cases)
            reports["rules+classifier"]["classifier_model"] = settings.classifier_model
        _print("rules+classifier", reports["rules+classifier"])
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
