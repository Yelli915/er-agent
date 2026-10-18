"""병상 조회 결과를 잠깐 저장해두는 캐시

- 같은 (시도, 시군구)는 60초 동안 캐시를 그대로 사용
- API 호출이 계속 실패했을 때 캐시를 대신 쓰면 stale=True로 표시

"""

from __future__ import annotations

import time
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_TTL_SECONDS = 60


def make_key(sido: str, sigungu: str) -> str:
    """시도·시군구로 캐시 키를 만든다"""
    return f"{sido}:{sigungu}"


def get(cache: dict[str, tuple[T, float]], key: str, ttl: float = DEFAULT_TTL_SECONDS) -> T | None:
    """캐시에서 값을 꺼낸다. 없거나 ttl초가 지났으면 None"""
    if key not in cache:
        return None
    value, saved_at = cache[key]
    if time.time() - saved_at >= ttl:
        return None
    return value


def save(cache: dict[str, tuple[T, float]], key: str, value: T) -> None:
    """캐시에 값을 저장한다."""
    cache[key] = (value, time.time())


def make_stale(rows: list[Any]) -> list[Any]:
    """캐시를 폴백으로 쓸 때 각 병원 dict에 stale=True를 강제로 붙인다"""
    result = []
    for row in rows:
        row_with_stale = dict(row)
        row_with_stale["stale"] = True
        result.append(row_with_stale)
    return result
