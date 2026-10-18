from datetime import UTC, datetime, timedelta

import pytest

from er_finder.search.candidates import is_stale, select_candidates

NOW = datetime(2027, 1, 1, tzinfo=UTC)


def iso(delta_seconds=0):
    return (NOW + timedelta(seconds=delta_seconds)).isoformat()


def make_facility(name, distance_km, er_tel=None, address=None):
    return {"name": name, "distance_km": distance_km, "er_tel": er_tel, "address": address}


def passthrough_or_unknown(value):
    # Preserve safe_data's required string fallback without exercising its rules.
    return value if isinstance(value, str) and value.strip() else "확인 불가"


@pytest.fixture(autouse=True)
def passthrough_safe_data(monkeypatch):
    # select_candidates only orchestrates safety scrubbing; the scrubbing
    # rules themselves belong to er_finder.guardrails and are tested there.
    monkeypatch.setattr("er_finder.search.candidates.safe_data", passthrough_or_unknown)


def test_is_stale_missing_or_malformed_timestamp():
    assert is_stale(None, NOW) is True
    assert is_stale("", NOW) is True
    assert is_stale("not-a-timestamp", NOW) is True


def test_is_stale_naive_timestamp_is_treated_as_stale():
    assert is_stale(datetime(2027, 1, 1).isoformat(), NOW) is True


def test_is_stale_freshness_window():
    assert is_stale(iso(-60), NOW) is False
    assert is_stale(iso(-901), NOW) is True
    assert is_stale(iso(60), NOW) is False
    assert is_stale(iso(61), NOW) is True


def test_filters_out_hospitals_without_available_beds():
    facilities = {"H1": make_facility("병원1", 1.0), "H2": make_facility("병원2", 2.0)}
    beds = {
        "H1": {"er_beds_available": 0, "beds_updated_at": iso()},
        "H2": {"er_beds_available": None, "beds_updated_at": iso()},
    }
    assert select_candidates(facilities, beds, {}, {}, condition=None, now=NOW) == []


def test_boolean_bed_count_is_rejected_even_though_true_is_truthy():
    facilities = {"H1": make_facility("병원1", 1.0)}
    beds = {"H1": {"er_beds_available": True, "beds_updated_at": iso()}}
    assert select_candidates(facilities, beds, {}, {}, condition=None, now=NOW) == []


def test_filters_out_hospitals_that_reject_the_condition():
    facilities = {
        "H1": make_facility("수용", 1.0),
        "H2": make_facility("거부", 2.0),
        "H3": make_facility("미확인", 3.0),
    }
    beds = {hpid: {"er_beds_available": 3, "beds_updated_at": iso()} for hpid in facilities}
    acceptance = {"H1": {"acceptable": True}, "H2": {"acceptable": False}}

    result = select_candidates(facilities, beds, acceptance, {}, condition="흉통", now=NOW)

    assert [h.hpid for h in result] == ["H1"]
    assert result[0].accepts_condition == "yes"


def test_without_a_condition_acceptance_is_not_checked_and_marked_unknown():
    facilities = {"H1": make_facility("병원1", 1.0)}
    beds = {"H1": {"er_beds_available": 1, "beds_updated_at": iso()}}
    acceptance = {"H1": {"acceptable": False}}

    result = select_candidates(facilities, beds, acceptance, {}, condition=None, now=NOW)

    assert len(result) == 1
    assert result[0].accepts_condition == "unknown"


def test_sorts_by_distance_and_caps_at_three_results():
    facilities = {
        "H1": make_facility("1", 5.0),
        "H2": make_facility("2", 1.0),
        "H3": make_facility("3", 3.0),
        "H4": make_facility("4", 2.0),
    }
    beds = {hpid: {"er_beds_available": 1, "beds_updated_at": iso()} for hpid in facilities}

    result = select_candidates(facilities, beds, {}, {}, condition=None, now=NOW)

    assert [h.hpid for h in result] == ["H2", "H4", "H3"]


def test_ties_broken_by_hpid():
    facilities = {"H2": make_facility("2", 1.0), "H1": make_facility("1", 1.0)}
    beds = {hpid: {"er_beds_available": 1, "beds_updated_at": iso()} for hpid in facilities}

    result = select_candidates(facilities, beds, {}, {}, condition=None, now=NOW)

    assert [h.hpid for h in result] == ["H1", "H2"]


def test_phone_number_is_taken_from_detail_then_bed_then_facility():
    facilities = {"H1": make_facility("병원1", 1.0, er_tel="facility-tel")}
    beds = {"H1": {"er_beds_available": 1, "beds_updated_at": iso(), "er_tel": "bed-tel"}}
    details = {"H1": {"er_tel": "detail-tel"}}

    result = select_candidates(facilities, beds, {}, details, condition=None, now=NOW)

    assert result[0].er_tel == "detail-tel"


def test_phone_number_scrubbed_to_unverifiable_is_cleared_to_none(monkeypatch):
    monkeypatch.setattr(
        "er_finder.search.candidates.safe_data",
        lambda value: "확인 불가" if value == "facility-tel" else passthrough_or_unknown(value),
    )
    facilities = {"H1": make_facility("병원1", 1.0, er_tel="facility-tel")}
    beds = {"H1": {"er_beds_available": 1, "beds_updated_at": iso()}}

    result = select_candidates(facilities, beds, {}, {}, condition=None, now=NOW)

    assert result[0].er_tel is None


def test_is_cached_flag_is_propagated_and_stale_age_sets_is_stale():
    facilities = {"H1": make_facility("병원1", 1.0)}
    beds = {
        "H1": {
            "er_beds_available": 1,
            "beds_updated_at": iso(-901),  # older than the 15-minute freshness window
            "is_cached": True,
        }
    }

    result = select_candidates(facilities, beds, {}, {}, condition=None, now=NOW)

    assert result[0].is_cached is True
    assert result[0].is_stale is True


def test_explicit_is_stale_flag_is_or_ed_with_age_based_staleness():
    facilities = {"H1": make_facility("병원1", 1.0)}
    beds = {
        "H1": {
            "er_beds_available": 1,
            "beds_updated_at": iso(),  # fresh by age alone
            "is_stale": True,
        }
    }

    result = select_candidates(facilities, beds, {}, {}, condition=None, now=NOW)

    assert result[0].is_stale is True
