"""Live-provider glue uses existing parsers and only offline fixtures."""

import importlib
from pathlib import Path

import pytest


class Geocoder:
    def __init__(self):
        self.closed = 0

    def geocode(self, query):
        return {"found": True, "address": query, "lat": 37.5, "lon": 127.0}

    def close(self):
        self.closed += 1


@pytest.fixture
def provider_module():
    return importlib.import_module("er_finder.web.provider")


@pytest.fixture
def medical(monkeypatch):
    monkeypatch.setenv("EGEN_SERVICE_KEY", "offline-provider-key")
    module = importlib.import_module("er_finder.medical_api.client")
    fixtures = Path(__file__).resolve().parents[2] / "fixtures" / "egen"
    calls = []

    def response(endpoint, params):
        calls.append(endpoint)
        files = {
            "/getEgytLcinfoInqire": "nearby_gangnam.xml",
            "/getEmrrmRltmUsefulSckbdInfoInqire": "bed_status_gangnam.xml",
            "/getSrsillDissAceptncPosblInfoInqire": "severe_gangnam.xml",
            "/getEgytBassInfoInqire": "detail_A1100057.xml",
        }
        return (fixtures / files[endpoint]).read_text(encoding="utf-8")

    monkeypatch.setattr(module, "_call", response)
    return module, calls


def test_provider_delegates_geocoding_and_preserves_medical_fields(provider_module, medical):
    client, _ = medical
    provider = provider_module.LiveProvider(geocoder=Geocoder(), medical_client=client)
    assert provider.geocode("강남역")["address"] == "강남역"
    assert provider.list_nearby_ers(37.5, 127.0, 5) == client.list_nearby_ers(37.5, 127.0, 5)
    assert provider.get_severe_acceptance("서울", "강남구", "심근경색") == (
        client.get_severe_acceptance("서울", "강남구", "심근경색")
    )
    assert provider.get_er_detail("A1100057") == client.get_er_detail("A1100057")


def test_bed_cache_is_per_provider_and_refresh_forces_existing_client_to_fetch(
    provider_module,
    medical,
):
    client, calls = medical
    one = provider_module.LiveProvider(geocoder=Geocoder(), medical_client=client)
    two = provider_module.LiveProvider(geocoder=Geocoder(), medical_client=client)
    first = one.get_er_bed_status("서울", "강남구", ["A1100015"])
    assert first and not first[0]["is_cached"]
    cached = one.get_er_bed_status("서울", "강남구", ["A1100015"])
    assert len(calls) == 1 and cached[0]["is_cached"]
    refreshed = one.get_er_bed_status("서울", "강남구", ["A1100015"], force_refresh=True)
    assert len(calls) == 2 and not refreshed[0]["is_cached"]
    two.get_er_bed_status("서울", "강남구", ["A1100015"])
    assert len(calls) == 3


def test_failed_force_refresh_retains_only_client_stale_fallback(
    provider_module, medical, monkeypatch
):
    client, _ = medical
    provider = provider_module.LiveProvider(geocoder=Geocoder(), medical_client=client)
    previous = provider.get_er_bed_status("서울", "강남구", ["A1100015"])

    def unavailable(endpoint, params):
        raise client.EgenAPIError()

    monkeypatch.setattr(client, "_call", unavailable)
    result = provider.get_er_bed_status("서울", "강남구", ["A1100015"], force_refresh=True)
    assert result[0]["er_beds_available"] == previous[0]["er_beds_available"]
    assert result[0]["stale"] and result[0]["is_stale"] and result[0]["is_cached"]


def test_provider_close_is_idempotent_and_clears_cached_records(provider_module, medical):
    client, calls = medical
    geocoder = Geocoder()
    provider = provider_module.LiveProvider(geocoder=geocoder, medical_client=client)
    provider.get_er_bed_status("서울", "강남구", ["A1100015"])
    provider.clear_cache()
    provider.get_er_bed_status("서울", "강남구", ["A1100015"])
    assert len(calls) == 2
    provider.close()
    provider.close()
    assert geocoder.closed == 1
    with pytest.raises(RuntimeError):
        provider.geocode("강남역")
