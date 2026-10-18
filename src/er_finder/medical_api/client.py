"""E-Gen calls and per-session caches.

The default interface retains empty-result fallbacks. Live consumers can pass
raise_on_error=True to distinguish a failed request from a successful empty result.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv

from er_finder.http_retry import request_with_transport_retries

from . import cache, parser

load_dotenv()


def _bounded_number(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(upper, max(lower, number)) if math.isfinite(number) else default


def _secure_url(value: str) -> str:
    # Upgrade the legacy official URL before sending any credential.
    return re.sub(r"^http://apis\.data\.go\.kr(?=/|$)", "https://apis.data.go.kr", value)


BASE_URL = _secure_url(
    os.environ.get("EGEN_BASE_URL", "https://apis.data.go.kr/B552657/ErmctInfoInqireService")
)
SERVICE_KEY = unquote(os.environ.get("EGEN_SERVICE_KEY", ""))
TIMEOUT = _bounded_number(os.environ.get("ER_REQUEST_TIMEOUT"), 5, 0.1, 10)
MAX_RETRIES = int(_bounded_number(os.environ.get("ER_MAX_RETRIES"), 2, 0, 2))
BACKOFF = _bounded_number(os.environ.get("ER_RETRY_BACKOFF"), 1, 0, 1)
BED_CACHE_TTL = _bounded_number(os.environ.get("ER_BED_CACHE_TTL"), 60, 1, 300)
STALE_MINUTES = _bounded_number(os.environ.get("ER_STALE_MINUTES"), 15, 1, 15)
DETAIL_CACHE_TTL = 300
DETAIL_FAILURE_TTL = 60
DETAIL_CACHE_LIMIT = 256
REGION_DETAIL_LIMIT = 3


class _ServiceKeyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = re.sub(
            r"([?&]serviceKey=)[^&\s\"']+", r"\1[REDACTED]", record.getMessage(), flags=re.I
        )
        record.args = ()
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(getattr(item, "name", None) == "egen-service-key" for item in _httpx_logger.filters):
    _httpx_logger.addFilter(_ServiceKeyFilter("egen-service-key"))


class EgenAPIError(RuntimeError):
    """A safe boundary error: never include response bodies, URLs or service keys."""

    def __init__(self) -> None:
        super().__init__("E-Gen 조회를 완료하지 못했습니다. 연결과 서비스 설정을 확인하세요.")


CONDITION_TO_MKIOSKTY = {
    "심근경색": [1],
    "뇌출혈": [3, 4],
    "뇌졸중": [2],
    "화상": [19],
    "분만": [16],
}


def _build_params(extra: dict[str, Any], num_of_rows: int = 100) -> dict[str, Any]:
    return {"serviceKey": SERVICE_KEY, "pageNo": 1, "numOfRows": num_of_rows, **extra}


def _region_params(sido: str, sigungu: str) -> dict[str, Any]:
    params = {"STAGE1": parser.normalize_sido(sido)}
    if sigungu:
        params["STAGE2"] = sigungu
    return params


def _call(endpoint: str, params: dict[str, Any]) -> str:
    try:
        if not SERVICE_KEY.strip():
            raise EgenAPIError()
        response = request_with_transport_retries(
            None,
            "GET",
            _secure_url(BASE_URL).rstrip("/") + endpoint,
            params=params,
            timeout=_bounded_number(TIMEOUT, 5, 0.1, 10),
            max_retries=int(_bounded_number(MAX_RETRIES, 2, 0, 2)),
            backoff=_bounded_number(BACKOFF, 1, 0, 1),
        )
        return response.text
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        raise EgenAPIError() from None


def _request(endpoint: str, extra: dict[str, Any], parse: Callable[[str], Any]) -> Any:
    try:
        return parse(_call(endpoint, _build_params(extra)))
    except parser.EgenResponseError:
        raise EgenAPIError() from None


def _lookup(
    endpoint: str,
    extra: dict[str, Any],
    parse: Callable[[str], Any],
    *,
    default: list[Any] | dict[str, Any],
    raise_on_error: bool,
) -> Any:
    """Return the parsed rows, or the empty default unless the caller wants the failure."""
    try:
        return _request(endpoint, extra, parse)
    except EgenAPIError:
        if raise_on_error:
            raise
        return default


def _is_stale(beds_updated_at: str | None) -> bool:
    if not beds_updated_at:
        return True
    try:
        updated_at = datetime.fromisoformat(beds_updated_at)
        if updated_at.tzinfo is None:
            return True
        age = datetime.now(parser.KOREA_TZ) - updated_at
        return age > timedelta(minutes=STALE_MINUTES) or age < -timedelta(minutes=1)
    except (TypeError, ValueError):
        return True


def _cached_detail(detail_cache: dict[str, Any], hpid: str) -> dict[str, Any] | None:
    record = cache.get(detail_cache, hpid, ttl=DETAIL_CACHE_TTL)
    if record is not None and record["lookup_failed"]:
        return cache.get(detail_cache, hpid, ttl=DETAIL_FAILURE_TTL)
    return record


def list_nearby_ers(
    lat: float,
    lon: float,
    radius_km: int = 5,
    *,
    detail_cache: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> list[dict[str, Any]]:
    rows = _lookup(
        "/getEgytLcinfoInqire",
        {"WGS84_LAT": lat, "WGS84_LON": lon},
        parser.parse_nearby,
        default=[],
        raise_on_error=raise_on_error,
    )

    nearby = [
        row
        for row in rows
        if row["hpid"]
        and row["lat"] is not None
        and -90 <= row["lat"] <= 90
        and row["lon"] is not None
        and -180 <= row["lon"] <= 180
        and row["distance_km"] is not None
        and 0 <= row["distance_km"] <= radius_km
    ]
    nearby.sort(key=lambda row: row["distance_km"])
    details = detail_cache if detail_cache is not None else {}
    extra_calls = 0
    for row in nearby:
        if not row["sido"] or row["sigungu"] is None:
            cached = _cached_detail(details, row["hpid"])
            if cached is not None or extra_calls < REGION_DETAIL_LIMIT:
                if cached is None:
                    extra_calls += 1
                try:
                    detail = get_er_detail(row["hpid"], detail_cache=details, raise_on_error=True)
                except EgenAPIError:
                    detail = {}
                if detail.get("address"):
                    row["address"] = detail["address"]
                    row["sido"], row["sigungu"] = parser.region_from_address(row["address"])
                row["er_tel"] = row["er_tel"] or detail.get("er_tel")
        row["region_lookup_failed"] = not row["sido"] or row["sigungu"] is None
    return nearby


def get_er_bed_status(
    sido: str,
    sigungu: str,
    hpids: list[str],
    bed_cache: dict[str, Any] | None = None,
    *,
    raise_on_error: bool = False,
) -> list[dict[str, Any]]:
    if bed_cache is None:
        bed_cache = {}
    key = cache.make_key(parser.normalize_sido(sido), sigungu)
    rows = cache.get(bed_cache, key, ttl=BED_CACHE_TTL)
    is_cached = rows is not None
    lookup_failed = False
    if rows is None:
        try:
            rows = _request(
                "/getEmrrmRltmUsefulSckbdInfoInqire",
                _region_params(sido, sigungu),
                parser.parse_bed_status,
            )
        except EgenAPIError:
            expired = bed_cache.get(key)
            if expired is None:
                if raise_on_error:
                    raise
                return []
            rows = expired[0]
            is_cached = lookup_failed = True
        else:
            cache.save(bed_cache, key, rows)

    wanted = set(hpids)
    result = []
    for row in rows:
        if row["hpid"] in wanted:
            stale = lookup_failed or _is_stale(row["beds_updated_at"])
            result.append(
                {
                    **row,
                    "stale": stale,
                    "is_stale": stale,
                    "is_cached": is_cached,
                    "lookup_failed": lookup_failed,
                }
            )
    if lookup_failed and not result and raise_on_error:
        raise EgenAPIError()
    return result


def _combine_mkioskty(mkioskty: dict[int, Any], codes: list[int]) -> bool | None:
    values = [mkioskty.get(code) for code in codes]
    if any(value is True for value in values):
        return True
    if all(value is False for value in values):
        return False
    return None


def get_severe_acceptance(
    sido: str, sigungu: str, condition: str, *, raise_on_error: bool = False
) -> list[dict[str, Any]]:
    codes = CONDITION_TO_MKIOSKTY.get(condition)
    if not codes:
        return []
    rows = _lookup(
        "/getSrsillDissAceptncPosblInfoInqire",
        _region_params(sido, sigungu),
        parser.parse_severe,
        default=[],
        raise_on_error=raise_on_error,
    )
    return [
        {
            "hpid": row["hpid"],
            "name": row["name"],
            "acceptable": _combine_mkioskty(row["mkioskty"], codes),
        }
        for row in rows
        if row["hpid"]
    ]


def get_er_detail(
    hpid: str, *, detail_cache: dict[str, Any] | None = None, raise_on_error: bool = False
) -> dict[str, Any]:
    if detail_cache is not None:
        record = _cached_detail(detail_cache, hpid)
        if record is not None:
            if record["lookup_failed"] and raise_on_error:
                raise EgenAPIError()
            return dict(record["data"])
    lookup_failed = False
    detail: dict[str, Any]
    try:
        detail = _request("/getEgytBassInfoInqire", {"HPID": hpid}, parser.parse_er_detail)
        if detail.get("hpid") and detail["hpid"] != hpid:
            raise EgenAPIError()
    except EgenAPIError:
        detail = {}
        lookup_failed = True
    if detail_cache is not None:
        if hpid not in detail_cache and len(detail_cache) >= DETAIL_CACHE_LIMIT:
            detail_cache.pop(next(iter(detail_cache)))
        cache.save(detail_cache, hpid, {"data": dict(detail), "lookup_failed": lookup_failed})
    if lookup_failed and raise_on_error:
        raise EgenAPIError() from None
    return detail
