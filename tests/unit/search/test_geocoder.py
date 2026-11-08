import json
from pathlib import Path

import httpx
import pytest

from er_finder.search.geocoder import KakaoGeocoder


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("강원도 강릉시 저동 94", "강원특별자치도"),
        ("전라북도 전주시 덕진구 건지로 20", "전북특별자치도"),
    ],
)
def test_legacy_region_names_use_shared_normalization(address, expected):
    document = {"address_name": address, "x": "127.0", "y": "37.5"}
    geocoder = KakaoGeocoder(
        kakao_key="key",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"documents": [document]})
            )
        ),
    )
    assert geocoder.geocode(address)["sido"] == expected


def test_geocode_uses_address_then_keyword_fallback_and_parses_region():
    paths: list[str] = []
    fixture_path = Path(__file__).parents[2] / "fixtures/kakao/sample_response.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload["fixture_provenance"]["kind"] == "synthetic"

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.headers["Authorization"] == "KakaoAK kakao-secret"
        if request.url.path.endswith("/address.json"):
            return httpx.Response(200, json={"meta": {"total_count": 0}, "documents": []})
        return httpx.Response(200, json=payload)

    geocoder = KakaoGeocoder(
        kakao_key="kakao-secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    result = geocoder.geocode("역삼역")

    assert result == {
        "found": True,
        "lat": 37.5007,
        "lon": 127.0365,
        "address": "서울 강남구 테헤란로 156",
        "sido": "서울특별시",
        "sigungu": "강남구",
    }
    assert paths == ["/v2/local/search/address.json", "/v2/local/search/keyword.json"]


def test_geocode_address_result_prefers_documented_region_fields():
    payload = {
        "meta": {"total_count": 1},
        "documents": [
            {
                "address_name": "강원특별자치도 강릉시 저동 94",
                "x": "128.907",
                "y": "37.795",
                "address": {
                    "address_name": "강원특별자치도 강릉시 저동 94",
                    "region_1depth_name": "강원특별자치도",
                    "region_2depth_name": "강릉시",
                },
                "road_address": None,
            }
        ],
    }
    geocoder = KakaoGeocoder(
        kakao_key="key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )

    result = geocoder.geocode("강릉시 저동 94")

    assert result["found"] is True
    assert result["sido"] == "강원특별자치도"
    assert result["sigungu"] == "강릉시"


def test_geocode_fails_closed_for_no_key_bad_json_and_invalid_coordinates():
    no_key = KakaoGeocoder(
        kakao_key="",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    assert no_key.geocode("역삼역") == {
        "found": False,
        "lat": None,
        "lon": None,
        "address": None,
        "sido": None,
        "sigungu": None,
    }

    bad_json = KakaoGeocoder(
        kakao_key="key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))
        ),
    )
    assert bad_json.geocode("역삼역")["found"] is False

    bad_coords_payload = {"documents": [{"address_name": "서울 강남구", "x": "nan", "y": "37.5"}]}
    bad_coords = KakaoGeocoder(
        kakao_key="key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bad_coords_payload))
        ),
    )
    assert bad_coords.geocode("강남구")["found"] is False


def test_geocode_retries_transport_errors_only():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, json={"documents": []})

    geocoder = KakaoGeocoder(
        kakao_key="key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    assert geocoder.geocode("역삼역")["found"] is False
    assert attempts == 4  # 3 address attempts, then one successful keyword request
