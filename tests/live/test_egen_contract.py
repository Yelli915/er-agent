"""E-Gen response shape against the real service: structure, never specific data values.

Beds and acceptance change by the minute, so these tests assert that fields exist with the
expected types and that the parser still recognises the live tags (e.g. MKioskTy casing).
"""

import pytest

pytestmark = pytest.mark.live

GANGNAM = (37.4979, 127.0276)


@pytest.fixture(scope="module")
def nearby(egen):
    rows = egen.list_nearby_ers(*GANGNAM, radius_km=10, raise_on_error=True)
    assert rows, "강남역 반경 10km에서 응급의료기관이 하나도 조회되지 않았습니다."
    return rows


def test_nearby_rows_keep_the_fields_search_depends_on(nearby):
    for row in nearby:
        assert isinstance(row["hpid"], str) and row["hpid"]
        assert isinstance(row["name"], str) and row["name"]
        assert isinstance(row["lat"], float) and isinstance(row["lon"], float)
        assert isinstance(row["distance_km"], (int, float)) and row["distance_km"] >= 0
    assert any(row["sido"] and row["sigungu"] is not None for row in nearby), (
        "어떤 기관에서도 시도/시군구를 얻지 못했습니다. 주소 필드 형식이 바뀌었을 수 있습니다."
    )
    assert nearby == sorted(nearby, key=lambda row: row["distance_km"])


@pytest.fixture(scope="module")
def region(nearby):
    row = next(r for r in nearby if r["sido"] and r["sigungu"] is not None)
    return row["sido"], row["sigungu"], [r["hpid"] for r in nearby]


def test_bed_status_returns_integer_or_unknown(egen, region):
    sido, sigungu, hpids = region
    rows = egen.get_er_bed_status(sido, sigungu, hpids, raise_on_error=True)
    assert rows, "같은 지역 병상 조회에서 목록의 기관이 하나도 반환되지 않았습니다."
    for row in rows:
        assert row["er_beds_available"] is None or (
            isinstance(row["er_beds_available"], int) and row["er_beds_available"] >= 0
        )
        assert row["beds_updated_at"] is None or isinstance(row["beds_updated_at"], str)
        assert isinstance(row["is_stale"], bool)


@pytest.mark.parametrize("condition", ["심근경색", "뇌졸중"])
def test_severe_acceptance_recognises_live_mkioskty_tags(egen, region, condition):
    sido, sigungu, _ = region
    rows = egen.get_severe_acceptance(sido, sigungu, condition, raise_on_error=True)
    assert rows
    assert all(row["acceptable"] in (True, False, None) for row in rows)
    # Every row unknown means the tags were not parsed (the MKioskTy/mkioskty casing bug).
    assert any(row["acceptable"] is not None for row in rows), (
        f"{condition}: 모든 기관의 수용 여부가 확인 불가입니다. 태그 형식이 바뀌었을 수 있습니다."
    )


def test_detail_matches_requested_hospital(egen, nearby):
    hpid = nearby[0]["hpid"]
    detail = egen.get_er_detail(hpid, raise_on_error=True)
    assert detail["hpid"] == hpid
    assert isinstance(detail.get("address"), str) and detail["address"]
