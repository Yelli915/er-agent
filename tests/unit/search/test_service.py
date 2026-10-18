from datetime import UTC, datetime

import pytest
from tests.unit.search.provider_fixture import OfflineSearchProvider

from er_finder.search import SearchSession, ToolOrderError


def finish(s):
    for _ in range(24):
        calls = s.next_calls()
        if not calls:
            return s.make_reply()
        for call in calls:
            getattr(s, call["name"])(**call["args"])
    raise AssertionError("search did not terminate")


def test_gangnam_urgent_filters_and_orders_fixture_candidates():
    s = SearchSession(OfflineSearchProvider())
    s.begin("가슴이 답답하고 식은땀이 나요. 강남구 역삼동")
    r = finish(s)
    assert r.severity == "urgent"
    assert [h.hpid for h in r.hospitals] == ["TEST-GN-1", "TEST-GN-2", "TEST-GN-5"]
    assert all(h.accepts_condition == "yes" and h.er_beds_available > 0 for h in r.hospitals)
    assert r.search_radius_km == 5


def test_walk_skips_severe_and_starts_three():
    s = SearchSession(OfflineSearchProvider(), transport="walk")
    s.begin("경포대 근처, 아이 이마가 찢어졌어요")
    r = finish(s)
    assert r.hospitals and r.search_radius_km == 3
    assert "get_severe_acceptance" not in [c["name"] for c in s.audit]


def test_expansion_exactly_three_and_119():
    s = SearchSession(OfflineSearchProvider())
    s.begin("영월군 상동읍, 할머니가 숨쉬기 힘들어해요")
    r = finish(s)
    assert not r.hospitals and r.call_119_first and r.no_candidate_reason
    assert [c["radius_km"] for c in s.audit if c["name"] == "list_nearby_ers"] == [5, 10, 20, 30]


def test_location_reuse_and_reset():
    s = SearchSession(OfflineSearchProvider())
    s.begin("강남구 역삼동 손가락이 부었어요")
    finish(s)
    assert s.location is not None
    s.begin("소아과 응급실만 다시 보여줘")
    r = finish(s)
    assert "geocode" not in [c["name"] for c in s.audit]
    assert "소아" in r.next_action
    s.clear()
    assert s.location is None and not s.candidates
    assert s.facilities == s.beds == s.acceptance == s.details == {}
    assert s.audit == []
    assert s.status()["draft"]["hospitals"] == []


def test_missing_unknown_location_and_invalid_tool_order():
    s = SearchSession(OfflineSearchProvider())
    s.begin("손가락이 부었어요")
    assert not s.next_calls()
    assert "위치" in s.make_reply().next_action
    s.begin("우리 동네 응급실 알려줘")
    finish(s)
    assert [c["name"] for c in s.audit] == ["geocode"]
    with pytest.raises(ToolOrderError):
        s.list_nearby_ers(37.5, 127, 5)


def test_model_cannot_change_coordinates_skip_radius_or_forge_ids():
    s = SearchSession(OfflineSearchProvider())
    s.begin("강남구 역삼동 발목 삐었어요")
    s.geocode(s.location_query)
    with pytest.raises(ToolOrderError):
        s.list_nearby_ers(0, 0, 30)
    s.list_nearby_ers(s.location["lat"], s.location["lon"], 5)
    with pytest.raises(ToolOrderError):
        s.get_er_bed_status([{"sido": "서울특별시", "sigungu": "강남구"}], ["FORGED"])


def test_untrusted_output_is_checked_per_hospital():
    s = SearchSession(OfflineSearchProvider())
    s.begin("강남구 역삼동 발목 삐었어요")
    r = finish(s)
    raw = r.model_dump()
    raw["hospitals"][0]["name"] = r.hospitals[1].name
    raw["hospitals"][0]["er_beds_available"] = 999
    raw["hospitals"][0]["er_tel"] = "02-999-9999"
    raw["next_action"] = "심근경색입니다. 약을 드세요."
    checked = s.check_evidence(raw)
    assert checked.hospitals[0].name == r.hospitals[0].name
    assert checked.hospitals[0].er_beds_available == r.hospitals[0].er_beds_available
    assert checked.hospitals[0].er_tel == r.hospitals[0].er_tel
    assert "심근경색입니다" not in checked.next_action
    assert s.evidence_corrections >= 3


def test_timestamp_staleness_and_no_invented_telephone():
    s = SearchSession(OfflineSearchProvider(), now=lambda: datetime(2027, 1, 1, tzinfo=UTC))
    s.begin("강남구 역삼동 손가락 부음")
    r = finish(s)
    assert all(h.stale for h in r.hospitals)
    assert all(h.er_tel is None for h in r.hospitals)


def test_severe_trauma_api_limitation_message():
    s = SearchSession(OfflineSearchProvider())

    s.begin("강남구 역삼동 중증외상 환자입니다")

    s.triage = s.triage.model_copy(update={"condition": "중증외상"})

    s.location = {
        "lat": 37.5,
        "lon": 127.0,
        "sido": "서울특별시",
        "sigungu": "강남구",
    }
    s.beds = {
        "TEST-ER": {
            "er_beds_available": 3,
        }
    }
    s.acceptance = {
        "TEST-ER": {
            "acceptable": None,
        }
    }
    s.candidates = []

    reply = s.make_reply()

    assert reply.hospitals == []
    assert "중증외상" in reply.no_candidate_reason
    assert "정확히 확인할 수 없어" in reply.no_candidate_reason
    assert "119" in reply.next_action


def test_severe_trauma_without_beds_uses_general_no_candidate_message():
    s = SearchSession(OfflineSearchProvider())

    s.begin("강남구 역삼동 중증외상 환자입니다")

    s.triage = s.triage.model_copy(update={"condition": "중증외상"})

    s.location = {
        "lat": 37.5,
        "lon": 127.0,
        "sido": "서울특별시",
        "sigungu": "강남구",
    }

    s.beds = {
        "TEST-ER": {
            "er_beds_available": 0,
        }
    }
    s.acceptance = {
        "TEST-ER": {
            "acceptable": None,
        }
    }
    s.candidates = []

    reply = s.make_reply()

    assert reply.hospitals == []
    assert "반경" in reply.no_candidate_reason
    assert "후보가 없습니다" in reply.no_candidate_reason
    assert "정확히 확인할 수 없어" not in reply.no_candidate_reason
    assert "119" in reply.next_action


@pytest.mark.parametrize(
    "proposal", [{"hospitals": None}, {"hospitals": "invalid"}, {"hospitals": [{"hpid": []}]}, None]
)
def test_malformed_proposal_preserves_verified_source_reply(proposal):
    session = SearchSession(OfflineSearchProvider())
    session.begin("강남구 역삼동 손가락이 부었어요")
    canonical = finish(session)
    assert session.check_evidence(proposal) == canonical


def test_lookup_failure_stops_retries_and_does_not_claim_no_hospitals():
    session = SearchSession(OfflineSearchProvider())
    session.begin("강남구 역삼동 손가락이 부었어요")
    finish(session)
    session.fail_lookup()
    assert session.next_calls() == []
    reply = session.make_reply()
    assert not reply.hospitals
    assert "조회" in reply.no_candidate_reason and "실패" in reply.no_candidate_reason
    assert "후보가 없습니다" not in reply.no_candidate_reason
    session.begin("강남구 역삼동 손가락이 부었어요")
    assert session.lookup_error is None and session.next_calls()
