"""Connect the runner's provider port to the team's existing real API clients."""

from __future__ import annotations

import importlib
import os
from threading import RLock
from typing import Any

from er_finder.medical_api.cache import make_key
from er_finder.medical_api.parser import normalize_sido
from er_finder.search.coordinates import parse_coordinates


class LiveProvider:
    """Own the geocoder and one browser's bed cache; retain source response semantics."""

    def __init__(
        self,
        *,
        kakao_key: str | None = None,
        geocoder: Any = None,
        medical_client: Any = None,
    ) -> None:
        self._medical = (
            medical_client
            if medical_client is not None
            else (importlib.import_module("er_finder.medical_api.client"))
        )
        self._geocoder = (
            geocoder
            if geocoder is not None
            else (
                importlib.import_module("er_finder.search.geocoder").KakaoGeocoder(
                    kakao_key=kakao_key
                    if kakao_key is not None
                    else os.environ.get("KAKAO_REST_API_KEY", "")
                )
            )
        )
        self._bed_cache: dict[str, Any] = {}
        self._detail_cache: dict[str, Any] = {}
        self._lock = RLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("연결이 종료되었습니다. 다시 연결해 주세요.")

    def geocode(self, query: str) -> dict[str, Any]:
        self._ensure_open()
        coordinates = parse_coordinates(query)
        if coordinates is not None:
            return dict(found=True, lat=coordinates[0], lon=coordinates[1])
        found: dict[str, Any] = self._geocoder.geocode(query)
        return found

    def list_nearby_ers(self, lat: float, lon: float, radius_km: int = 5) -> list[dict[str, Any]]:
        self._ensure_open()
        with self._lock:
            nearby: list[dict[str, Any]] = self._medical.list_nearby_ers(
                lat, lon, radius_km, detail_cache=self._detail_cache, raise_on_error=True
            )
            return nearby

    def get_er_bed_status(
        self,
        sido: str,
        sigungu: str,
        hpids: list[str],
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        with self._lock:
            key = make_key(normalize_sido(sido), sigungu)
            if force_refresh and key in self._bed_cache:
                # Expire, rather than remove, so the client's own failure path
                # may return its explicitly stale cached response.
                rows, _ = self._bed_cache[key]
                self._bed_cache[key] = (rows, 0.0)
            before = self._bed_cache.get(key)
            rows = self._medical.get_er_bed_status(
                sido,
                sigungu,
                hpids,
                bed_cache=self._bed_cache,
                raise_on_error=True,
            )
            cached = before is not None and self._bed_cache.get(key) is before
            return [
                {
                    **row,
                    "is_cached": bool(row.get("is_cached", cached)),
                    "is_stale": bool(row.get("is_stale", row.get("stale", False))),
                }
                for row in rows
            ]

    def get_severe_acceptance(
        self, sido: str, sigungu: str, condition: str
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        severe: list[dict[str, Any]] = self._medical.get_severe_acceptance(
            sido, sigungu, condition, raise_on_error=True
        )
        return severe

    def get_er_detail(self, hpid: str) -> dict[str, Any]:
        self._ensure_open()
        with self._lock:
            detail: dict[str, Any] = self._medical.get_er_detail(
                hpid, detail_cache=self._detail_cache, raise_on_error=True
            )
            return detail

    def clear_cache(self) -> None:
        with self._lock:
            self._bed_cache.clear()
            self._detail_cache.clear()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.clear_cache()
            self._geocoder.close()
