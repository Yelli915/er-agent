from types import SimpleNamespace

from er_finder.guardrails.evidence import DIAGNOSIS_NOTICE
from er_finder.guardrails.middleware import EvidenceCheckMiddleware
from er_finder.models import DISCLAIMER, ERSearchReply


def _reply(next_action: str) -> ERSearchReply:
    return ERSearchReply(
        severity="urgent",
        call_119_first=False,
        search_radius_km=5,
        hospitals=[],
        no_candidate_reason="반경 5km 안에 후보가 없습니다.",
        data_timestamp="2026-01-01T00:00:00+09:00",
        next_action=next_action,
        disclaimer=DISCLAIMER,
    )


def _run(next_action: str) -> ERSearchReply:
    session = SimpleNamespace(
        check_evidence=lambda proposal: _reply(next_action), snapshot=lambda: {}
    )
    guard = EvidenceCheckMiddleware(session)
    return guard.after_agent({"structured_response": None}, None)["structured_response"]


def test_sanitized_next_action_stays_within_the_contract_length():
    long_safe = "가까운 응급실에 전화로 먼저 확인하세요. " * 3
    reply = _run(f"{long_safe}이 약을 드세요.")
    assert len(reply.next_action) <= 80
    assert reply.next_action.endswith(DIAGNOSIS_NOTICE)


def test_clean_next_action_passes_through():
    reply = _run("병원에 전화로 확인하세요.")
    assert reply.next_action == "병원에 전화로 확인하세요."
