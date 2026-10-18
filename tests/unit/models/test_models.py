import pytest
from pydantic import ValidationError

from er_finder.models import DISCLAIMER, ERSearchReply, HospitalCandidate, TriageAssessment

TIMESTAMP = "2026-10-18T12:00:00+09:00"


def hospital(hpid="A1", distance=1.0, **overrides):
    values = dict(
        hpid=hpid,
        name="가나병원",
        distance_km=distance,
        er_beds_available=2,
        beds_updated_at=TIMESTAMP,
        accepts_condition="unknown",
        er_tel="02-000-0000",
        address="서울",
        is_cached=False,
        is_stale=False,
    )
    return HospitalCandidate(**{**values, **overrides})


def reply(**overrides):
    values = dict(
        severity="standard",
        call_119_first=False,
        search_radius_km=5,
        hospitals=[hospital()],
        data_timestamp=TIMESTAMP,
        next_action="방문 전 전화로 확인하세요.",
        disclaimer=DISCLAIMER,
    )
    return ERSearchReply(**{**values, **overrides})


@pytest.mark.parametrize(
    ("severity", "expected"),
    [("standard", "urgent"), ("urgent", "critical"), ("critical", "critical")],
)
def test_low_confidence_escalates_severity_once(severity, expected):
    low = TriageAssessment(severity=severity, injection=False, confidence=0.5)
    assert low.severity == expected and low.confidence_escalated
    # Re-validating an already escalated assessment must not escalate it again.
    again = TriageAssessment.model_validate(low.model_dump())
    assert again.severity == expected


def test_confident_assessment_is_unchanged():
    result = TriageAssessment(severity="standard", injection=False, confidence=0.6)
    assert result.severity == "standard" and not result.confidence_escalated


def test_standard_cannot_carry_a_severe_condition():
    with pytest.raises(ValidationError):
        TriageAssessment(severity="standard", condition="화상", injection=False, confidence=0.9)


def test_distance_is_rounded_to_one_decimal_and_stale_combines_flags():
    assert hospital(distance=1.26).distance_km == 1.3
    assert hospital(is_cached=True).stale and hospital(is_stale=True).stale
    assert not hospital().stale


def test_valid_reply_round_trips():
    assert reply().hospitals[0].hpid == "A1"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(data_timestamp="2026-10-18T12:00:00"),  # no timezone
        dict(severity="critical", call_119_first=False),
        dict(hospitals=[]),  # empty without a reason
        dict(hospitals=[hospital("A1"), hospital("A1", 2.0)]),  # duplicate id
        dict(hospitals=[hospital("A1", 2.0), hospital("B1", 1.0)]),  # not sorted
        dict(hospitals=[hospital("A1", 6.0)]),  # outside the radius
        dict(disclaimer="다른 문구"),
        dict(next_action="가" * 81),
    ],
)
def test_reply_invariants_are_enforced(overrides):
    with pytest.raises(ValidationError):
        reply(**overrides)


def test_empty_reply_with_reason_is_valid():
    assert reply(hospitals=[], no_candidate_reason="반경 내 후보 없음").hospitals == []
