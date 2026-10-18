"""Recorded E-Gen fields and synthetic transport failures, without live requests."""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from er_finder.medical_api import cache, client, parser


def xml(items: str) -> str:
    return (
        "<response><header><resultCode>00</resultCode></header>"
        f"<body><items>{items}</items></body></response>"
    )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.5", "1.5", "-1", ""])
def test_invalid_bed_counts_are_unknown_without_raising(value):
    assert (
        parser.parse_bed_status(xml(f"<item><hpid>A1</hpid><hvec>{value}</hvec></item>"))[0][
            "er_beds_available"
        ]
        is None
    )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_coordinates_are_unknown(value):
    assert parser.to_float(value) is None


def test_recorded_nearby_addresses_keep_each_hospitals_own_region(egen_fixture):
    rows = parser.parse_nearby(egen_fixture("nearby_gangnam.xml"))
    by_id = {row["hpid"]: row for row in rows}
    assert by_id["A1100141"]["address"] == "서울특별시 강남구 남부순환로 2649, 베드로병원 (도곡동)"
    assert (by_id["A1100141"]["sido"], by_id["A1100141"]["sigungu"]) == ("서울특별시", "강남구")
    assert (by_id["A1122033"]["sido"], by_id["A1122033"]["sigungu"]) == ("서울특별시", "서초구")
    assert (by_id["A1100004"]["sido"], by_id["A1100004"]["sigungu"]) == ("서울특별시", "용산구")


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("강원도 영월군 영월읍 중앙1로 59", ("강원특별자치도", "영월군")),
        ("전라북도 전주시 덕진구 건지로 20", ("전북특별자치도", "전주시")),
        ("세종특별자치시 보듬7로 20", ("세종특별자치시", "")),
        ("주소 확인 불가", (None, None)),
        (None, (None, None)),
    ],
)
def test_region_extraction_normalizes_only_actual_administrative_tokens(address, expected):
    assert parser.region_from_address(address) == expected


def test_severe_recordings_preserve_positive_negative_and_unknown(egen_fixture):
    rows = parser.parse_severe(egen_fixture("severe_gangneung.xml"))
    by_id = {row["hpid"]: row for row in rows}
    assert by_id["A2200008"]["mkioskty"][1] is True
    assert by_id["A2200005"]["mkioskty"][1] is False
    assert by_id["A2200011"]["mkioskty"][1] is None


def test_lowercase_severe_fields_remain_supported():
    row = parser.parse_severe(xml("<item><hpid>A1</hpid><mkioskty1>N</mkioskty1></item>"))[0]
    assert row["mkioskty"][1] is False


def test_bed_timestamp_preserves_korean_timezone_and_freshness():
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    fresh = parser.hvidate_to_iso(now.strftime("%Y%m%d%H%M%S"))
    assert datetime.fromisoformat(fresh).utcoffset() == timedelta(hours=9)
    assert client._is_stale(fresh) is False
    assert client._is_stale((now - timedelta(minutes=16)).isoformat()) is True
    assert client._is_stale((now + timedelta(minutes=2)).isoformat()) is True
    assert client._is_stale("not a date") is True


def test_parser_errors_never_include_raw_server_messages():
    with pytest.raises(parser.EgenResponseError) as error:
        parser.parse_nearby(
            "<response><header><resultCode>30</resultCode>"
            "<resultMsg>serviceKey=synthetic-secret</resultMsg></header></response>"
        )
    assert "synthetic-secret" not in str(error.value)


@respx.mock
def test_nearby_enrichment_reuses_recorded_address_without_detail_requests(egen_fixture):
    route = respx.get(client.BASE_URL + "/getEgytLcinfoInqire").mock(
        return_value=httpx.Response(200, text=egen_fixture("nearby_gangnam.xml"))
    )
    rows = client.list_nearby_ers(37.497942, 127.027621, detail_cache={}, raise_on_error=True)
    assert {row["sigungu"] for row in rows} == {"강남구", "서초구", "용산구", "동작구"}
    assert not any(row["region_lookup_failed"] for row in rows)
    assert route.call_count == len(respx.calls) == 1


@respx.mock
def test_missing_nearby_address_uses_detail_once_across_radius_expansion(egen_fixture):
    root = ET.fromstring(egen_fixture("nearby_gangnam.xml"))
    item = root.find("body/items/item")
    item.remove(item.find("dutyAddr"))
    respx.get(client.BASE_URL + "/getEgytLcinfoInqire").mock(
        return_value=httpx.Response(200, text=ET.tostring(root, encoding="unicode"))
    )
    detail = respx.get(
        client.BASE_URL + "/getEgytBassInfoInqire", params={"HPID": "A1100057"}
    ).mock(return_value=httpx.Response(200, text=egen_fixture("detail_A1100057.xml")))
    detail_cache = {}
    rows = client.list_nearby_ers(37.497942, 127.027621, detail_cache=detail_cache)
    assert rows[0]["sigungu"] == "강남구"
    assert rows[0]["address"].startswith("서울특별시 강남구 논현로")
    client.list_nearby_ers(37.497942, 127.027621, 10, detail_cache=detail_cache)
    client.get_er_detail("A1100057", detail_cache=detail_cache)
    assert detail.call_count == 1


@respx.mock
def test_missing_regions_have_a_bounded_detail_budget_and_failure_marker(egen_fixture, monkeypatch):
    monkeypatch.setattr(client, "MAX_RETRIES", 0)
    root = ET.fromstring(egen_fixture("nearby_gangnam.xml"))
    for item in root.findall("body/items/item"):
        item.remove(item.find("dutyAddr"))
    respx.get(client.BASE_URL + "/getEgytLcinfoInqire").mock(
        return_value=httpx.Response(200, text=ET.tostring(root, encoding="unicode"))
    )
    detail = respx.get(client.BASE_URL + "/getEgytBassInfoInqire").mock(
        return_value=httpx.Response(503)
    )
    rows = client.list_nearby_ers(37.497942, 127.027621, detail_cache={}, raise_on_error=True)
    assert detail.call_count == 3
    assert all(row["region_lookup_failed"] for row in rows)
    assert all(row["sigungu"] is None for row in rows)


@pytest.mark.parametrize(
    "body", ["<broken", "<response><header><resultCode>30</resultCode></header></response>"]
)
@respx.mock
def test_strict_schema_failures_are_distinct_from_normal_empty_results(body):
    route = respx.get(client.BASE_URL + "/getEgytLcinfoInqire").mock(
        return_value=httpx.Response(200, text=body)
    )
    with pytest.raises(client.EgenAPIError):
        client.list_nearby_ers(37.5, 127.0, raise_on_error=True)
    route.mock(return_value=httpx.Response(200, text=xml("")))
    assert client.list_nearby_ers(37.5, 127.0, raise_on_error=True) == []


@respx.mock
def test_http_failure_has_a_safe_error_and_redacts_service_key_logs(monkeypatch, caplog):
    monkeypatch.setattr(client, "SERVICE_KEY", "synthetic+secret/==")
    respx.get(client.BASE_URL + "/getEgytBassInfoInqire").mock(return_value=httpx.Response(403))
    caplog.set_level(logging.INFO, logger="httpx")
    with pytest.raises(client.EgenAPIError) as error:
        client.get_er_detail("A1", raise_on_error=True)
    assert "synthetic" not in str(error.value)
    assert "synthetic" not in caplog.text
    assert "http" not in str(error.value).lower()


@respx.mock
def test_official_http_endpoint_upgrades_to_https_and_request_limits_are_bounded(monkeypatch):
    monkeypatch.setattr(client, "BASE_URL", "http://apis.data.go.kr/B552657/ErmctInfoInqireService")
    monkeypatch.setattr(client, "TIMEOUT", float("inf"))
    monkeypatch.setattr(client, "MAX_RETRIES", 100)
    monkeypatch.setattr(client, "BACKOFF", 0)
    route = respx.get(
        "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytBassInfoInqire"
    ).mock(return_value=httpx.Response(503))
    with pytest.raises(client.EgenAPIError):
        client.get_er_detail("A1", raise_on_error=True)
    assert route.call_count == 3
    assert 0 < route.calls[0].request.extensions["timeout"]["read"] <= 10


@respx.mock
def test_bed_cache_metadata_marks_only_cache_reads_and_failed_refreshes(egen_fixture, monkeypatch):
    monkeypatch.setattr(client, "MAX_RETRIES", 0)
    route = respx.get(client.BASE_URL + "/getEmrrmRltmUsefulSckbdInfoInqire").mock(
        return_value=httpx.Response(200, text=egen_fixture("bed_status_gangnam.xml"))
    )
    store = {}
    first = client.get_er_bed_status(
        "서울특별시", "강남구", ["A1100015"], store, raise_on_error=True
    )
    assert first[0]["is_cached"] is False
    second = client.get_er_bed_status(
        "서울특별시", "강남구", ["A1100015"], store, raise_on_error=True
    )
    assert second[0]["is_cached"] is True
    assert second[0]["lookup_failed"] is False
    key = cache.make_key("서울특별시", "강남구")
    store[key] = (store[key][0], 0)
    route.mock(return_value=httpx.Response(503))
    fallback = client.get_er_bed_status(
        "서울특별시", "강남구", ["A1100015"], store, raise_on_error=True
    )
    assert fallback[0]["is_cached"] is True
    assert fallback[0]["is_stale"] is True
    assert fallback[0]["lookup_failed"] is True


@respx.mock
def test_strict_bed_and_severe_errors_do_not_return_unknown_success(monkeypatch):
    monkeypatch.setattr(client, "MAX_RETRIES", 0)
    respx.get(client.BASE_URL + "/getEmrrmRltmUsefulSckbdInfoInqire").mock(
        return_value=httpx.Response(503)
    )
    respx.get(client.BASE_URL + "/getSrsillDissAceptncPosblInfoInqire").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(client.EgenAPIError):
        client.get_er_bed_status("서울특별시", "강남구", ["A1"], raise_on_error=True)
    with pytest.raises(client.EgenAPIError):
        client.get_severe_acceptance("서울특별시", "강남구", "심근경색", raise_on_error=True)


@pytest.mark.parametrize("timestamp", ["202609111152", "2026091", "202609111152290"])
def test_incomplete_or_extra_timestamp_digits_are_unknown(timestamp):
    assert parser.hvidate_to_iso(timestamp) is None


@respx.mock
def test_strict_failed_refresh_with_no_matching_cached_hospital_still_raises(monkeypatch):
    monkeypatch.setattr(client, "MAX_RETRIES", 0)
    respx.get(client.BASE_URL + "/getEmrrmRltmUsefulSckbdInfoInqire").mock(
        return_value=httpx.Response(503)
    )
    store = {cache.make_key("서울특별시", "강남구"): ([], 0)}
    with pytest.raises(client.EgenAPIError):
        client.get_er_bed_status("서울특별시", "강남구", ["A1"], store, raise_on_error=True)


def test_a_missing_response_body_is_not_a_successful_empty_result():
    with pytest.raises(parser.EgenResponseError):
        parser.parse_nearby("<response><header><resultCode>00</resultCode></header></response>")


@respx.mock
def test_detail_cannot_supply_another_hospitals_address(egen_fixture):
    respx.get(client.BASE_URL + "/getEgytBassInfoInqire").mock(
        return_value=httpx.Response(200, text=egen_fixture("detail_A1100057.xml"))
    )
    with pytest.raises(client.EgenAPIError):
        client.get_er_detail("A1100141", raise_on_error=True)


@respx.mock
def test_detail_failure_is_short_cached_without_suppressing_strict_error(monkeypatch):
    monkeypatch.setattr(client, "MAX_RETRIES", 0)
    route = respx.get(client.BASE_URL + "/getEgytBassInfoInqire").mock(
        return_value=httpx.Response(503)
    )
    store = {}
    for _ in range(2):
        with pytest.raises(client.EgenAPIError):
            client.get_er_detail("A1", detail_cache=store, raise_on_error=True)
    assert route.call_count == 1


@respx.mock
def test_missing_credentials_fail_locally_without_spending_a_request(monkeypatch):
    monkeypatch.setattr(client, "SERVICE_KEY", "")
    with pytest.raises(client.EgenAPIError):
        client.list_nearby_ers(37.5, 127, raise_on_error=True)
    assert client.list_nearby_ers(37.5, 127) == []
    assert len(respx.calls) == 0
