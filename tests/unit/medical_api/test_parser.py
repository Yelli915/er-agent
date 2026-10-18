"""parser.py 단위 테스트."""

import pytest

from er_finder.medical_api import parser


def test_get_items_returns_rows_for_real_response(egen_fixture):
    xml_text = egen_fixture("nearby_gangnam.xml")
    items = parser.get_items(xml_text)
    assert len(items) > 0


def test_get_items_raises_on_result_code_error(egen_fixture):
    xml_text = egen_fixture("error_result_code.xml")
    with pytest.raises(parser.EgenResponseError):
        parser.get_items(xml_text)


@pytest.mark.parametrize("filename", ["empty_items_no_tag.xml", "empty_items_blank.xml"])
def test_get_items_empty_items_are_normalized_to_empty_list(egen_fixture, filename):
    xml_text = egen_fixture(filename)
    assert parser.get_items(xml_text) == []


def test_parse_nearby_real_response(egen_fixture):
    xml_text = egen_fixture("nearby_gangnam.xml")
    rows = parser.parse_nearby(xml_text)
    assert len(rows) > 0

    first = rows[0]
    assert first["hpid"]
    assert first["name"]
    assert first["er_tel"]  # dutyTel1로 채워짐 (design.md §B.1.3)
    assert isinstance(first["lat"], float)
    assert isinstance(first["lon"], float)
    assert isinstance(first["distance_km"], float)


def test_parse_bed_status_missing_values(egen_fixture):
    xml_text = egen_fixture("bed_status_missing_values.xml")
    rows = parser.parse_bed_status(xml_text)
    by_hpid = {row["hpid"]: row for row in rows}

    assert by_hpid["A9900001"]["er_beds_available"] is None  # hvec 음수 -> None
    assert by_hpid["A9900002"]["er_beds_available"] is None  # hvec 빈 문자열 -> None
    assert by_hpid["A9900003"]["beds_updated_at"] is None  # hvidate 태그 없음 -> None


def test_parse_severe_has_all_28_codes(egen_fixture):
    xml_text = egen_fixture("severe_gangnam.xml")
    rows = parser.parse_severe(xml_text)
    assert len(rows) > 0

    first = rows[0]
    assert set(first["mkioskty"].keys()) == set(range(1, 29))
    assert all(value in (True, False, None) for value in first["mkioskty"].values())


def test_parse_er_detail_real_response(egen_fixture):
    xml_text = egen_fixture("detail_A1100057.xml")
    detail = parser.parse_er_detail(xml_text)

    assert detail["name"]
    assert detail["address"]
    assert detail["hours"]
    # dutyTel3 대신 dutyTel1을 쓰기로 했으니 er_tel/main_tel이 같은 값이어야 한다.
    assert detail["er_tel"] == detail["main_tel"]


def test_parse_er_detail_no_result_returns_empty_dict():
    xml_text = (
        '<?xml version="1.0"?><response><header><resultCode>00</resultCode>'
        "<resultMsg>NORMAL SERVICE.</resultMsg></header>"
        "<body><totalCount>0</totalCount></body></response>"
    )
    assert parser.parse_er_detail(xml_text) == {}


def test_hvidate_to_iso():
    assert parser.hvidate_to_iso("20260910143000") == "2026-09-10T14:30:00+09:00"
    assert parser.hvidate_to_iso(None) is None
    assert parser.hvidate_to_iso("이상한값") is None


def test_yn_to_bool():
    assert parser.yn_to_bool("Y") is True
    assert parser.yn_to_bool("N") is False
    assert parser.yn_to_bool(None) is None


def test_to_bed_count():
    assert parser.to_bed_count("-1") is None
    assert parser.to_bed_count("") is None
    assert parser.to_bed_count("3") == 3


def test_normalize_sido():
    assert parser.normalize_sido("강원") == "강원특별자치도"
    assert parser.normalize_sido("서울") == "서울특별시"
    assert parser.normalize_sido("이미정식이름") == "이미정식이름"
