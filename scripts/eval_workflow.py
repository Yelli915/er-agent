"""Compare final-model vs deterministic replies offline; optionally evaluate real parsing."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er_finder.agent.runner import ERFinder  # noqa: E402
from er_finder.demo import DemoProvider, ScriptedModel  # noqa: E402
from er_finder.guardrails.triage import InputClassifier  # noqa: E402
from er_finder.search.service import SearchSession  # noqa: E402
from er_finder.usage import ClassifierUsage  # noqa: E402


class FrozenClassifier:
    """Acquire each input once, then replay detached assessments to both variants."""

    def __init__(self, source):
        self.source = source
        self.cache = {}
        self.acquisition_calls = 0
        self.last_usage = ClassifierUsage()

    def assess(self, text):
        if text not in self.cache:
            self.cache[text] = deepcopy(self.source.assess(text))
            self.acquisition_calls += getattr(self.source, "last_usage", ClassifierUsage()).calls
        return deepcopy(self.cache[text])


def evaluate_parsing(classifier, live):
    cases = json.loads((ROOT / "tests/fixtures/scenarios/parsing.json").read_text("utf-8"))
    rows = []
    for case in cases:
        outcomes = []
        sources = [("rules", InputClassifier())]
        if live:
            sources.append(("live", classifier))
        for name, source in sources:
            session = SearchSession(DemoProvider())
            session.begin(case["text"], assessment=source.assess(case["text"]))
            outcomes.append({
                "variant": name,
                "passed": session.location_query == case["query"]
                          and session.assessment.needs_location == case["needs_location"],
            })
        rows.append({"id": case["id"], "variants": outcomes})
    return {
        "mode": "parsing", "cases": rows,
        "live_status": "RUN" if live else "NOT_RUN",
        "passed": all(v["passed"] for r in rows for v in r["variants"] if v["variant"] == "live")
                  if live else None,
        "note": "Authored parsing regression cases; not held-out evidence of AI benefit.",
    }


def comparable(reply):
    data = reply.model_dump()
    data.pop("data_timestamp")
    for hospital in data["hospitals"]:
        hospital.pop("beds_updated_at")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "verification-results/workflow.json")
    parser.add_argument("--live", action="store_true", help="Call the real input model (paid)")
    parser.add_argument("--mode", choices=["workflow", "parsing"], default="workflow")
    args = parser.parse_args()
    classifier = InputClassifier()
    if args.live:
        from er_finder.agent.factory import build_classifier
        from er_finder.config import Settings
        from er_finder.web.live import load_environment

        load_environment()
        classifier = build_classifier(Settings())

    if args.mode == "parsing":
        report = evaluate_parsing(classifier, args.live)
        save_report(args.output, report)
        print(json.dumps({"live_status": report["live_status"], "passed": report["passed"]}))
        return int(report["passed"] is False)

    cases = json.loads((ROOT / "tests/fixtures/scenarios/dialogue.json").read_text("utf-8"))
    records = []
    for case in cases:
        variants = []
        frozen = FrozenClassifier(classifier)
        for deterministic in (False, True):
            finder = ERFinder(
                DemoProvider(), ScriptedModel(), classifier=frozen,
                deterministic_reply=deterministic,
            )
            try:
                results = [finder.chat(turn) for turn in case["turns"]]
                last = results[-1]
                passed = (
                    all(r.diagnostics.succeeded for r in results)
                    and bool(last.reply.hospitals) == case["has_candidates"]
                    and bool(last.pending_approval) == case.get("pending", False)
                    and last.reply.call_119_first == case.get("critical", False)
                )
                if "query" in case:
                    passed = passed and finder.session.location_query == case["query"]
                variants.append({
                    "variant": "deterministic" if deterministic else "final_model",
                    "passed": passed,
                    "graph_model_calls": sum(r.diagnostics.model_calls for r in results),
                    "classifier_calls": sum(r.diagnostics.classifier_calls for r in results),
                    "reply": comparable(last.reply),
                })
            finally:
                finder.close()
        records.append({
            "id": case["id"], "same_reply": variants[0]["reply"] == variants[1]["reply"],
            "input_acquisition_calls": frozen.acquisition_calls,
            "variants": [{k: v for k, v in row.items() if k != "reply"} for row in variants],
        })
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "provider": "invented demo data", "final_model": "scripted",
            "input_model": "live" if args.live else "rules only",
            "ai_parsing_benefit_verified": False,
            "comparison_control": "Same frozen input assessments replayed to both variants. "
                                  "Acquisition calls reported separately from graph calls.",
            "note": "Authored regression cases, not a held-out or clinical evaluation. "
                    "No real latency, token savings or cost claims from scripted runs.",
        },
        "cases": records,
        "passed": all(r["same_reply"] and all(v["passed"] for v in r["variants"]) for r in records),
    }
    save_report(args.output, report)
    print(json.dumps({"passed": report["passed"], "cases": len(records)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
