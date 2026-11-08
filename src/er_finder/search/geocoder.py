"""Kakao Local address and keyword geocoder."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import httpx

from er_finder.http_retry import SAFE_TIMEOUT, request_with_transport_retries
from er_finder.regions import normalize_sido

_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def empty_geocode() -> dict[str, Any]:
    return {
        "found": False,
        "lat": None,
        "lon": None,
        "address": None,
        "sido": None,
        "sigungu": None,
    }


def region_from_address(address: str) -> tuple[str | None, str | None]:
    parts = address.split()
    if not parts:
        return None, None
    sido = normalize_sido(parts[0])
    sigungu = parts[1] if len(parts) > 1 else None
    return sido, sigungu


class KakaoGeocoder:
    def __init__(
        self,
        *,
        kakao_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._key = kakao_key.strip()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=SAFE_TIMEOUT, follow_redirects=False)
        self._sleep = sleep or time.sleep

    def geocode(self, query: str) -> dict[str, Any]:
        query = query.strip()[:200]
        if not query or not self._key:
            return empty_geocode()

        document = self._lookup(_ADDRESS_URL, query)
        if document is None:
            document = self._lookup(_KEYWORD_URL, query)
        if document is None:
            return empty_geocode()

        try:
            lon = float(document["x"])
            lat = float(document["y"])
        except (KeyError, TypeError, ValueError):
            return empty_geocode()
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not (-90 <= lat <= 90)
            or not (-180 <= lon <= 180)
        ):
            return empty_geocode()

        nested_address = document.get("address") or {}
        road_address = document.get("road_address") or {}
        address = (
            road_address.get("address_name")
            or nested_address.get("address_name")
            or document.get("road_address_name")
            or document.get("address_name")
        )
        if not isinstance(address, str) or not address.strip():
            return empty_geocode()
        sido = nested_address.get("region_1depth_name") or road_address.get("region_1depth_name")
        sigungu = nested_address.get("region_2depth_name") or road_address.get("region_2depth_name")
        parsed_sido, parsed_sigungu = region_from_address(address)
        sido = sido or parsed_sido
        sigungu = sigungu or parsed_sigungu

        return {
            "found": True,
            "lat": lat,
            "lon": lon,
            "address": address,
            "sido": sido,
            "sigungu": sigungu,
        }

    def _lookup(self, url: str, query: str) -> dict[str, Any] | None:
        try:
            response = request_with_transport_retries(
                self._client,
                "GET",
                url,
                params={"query": query, "page": "1", "size": "1"},
                headers={"Authorization": f"KakaoAK {self._key}"},
                timeout=SAFE_TIMEOUT,
                sleep=self._sleep,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except (httpx.TransportError, ValueError, TypeError):
            return None
        documents = payload.get("documents") if isinstance(payload, dict) else None
        if not isinstance(documents, list) or not documents or not isinstance(documents[0], dict):
            return None
        return documents[0]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
