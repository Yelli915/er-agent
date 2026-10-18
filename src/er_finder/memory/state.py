"""Session State — 대화 중에만 유효하고 세션이 끝나면 사라지는 값.

create_agent의 state_schema로 등록되어 도구·미들웨어가 매 turn 읽고 쓴다.

candidates/triage 타입은 이후 models.py의 Pydantic 스키마로 교체될 수 있어
지금은 최소 TypedDict로만 정의해둔다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class Location(TypedDict):
    lat: float
    lon: float
    sido: str
    sigungu: str


class TriageState(TypedDict, total=False):
    severity: Literal["critical", "urgent", "standard"]
    condition: str | None
    injection: bool
    confidence: float


class HospitalCandidate(TypedDict, total=False):
    hpid: str
    name: str
    distance_km: float
    er_beds_available: int | None
    beds_updated_at: str | None
    accepts_condition: Literal["yes", "no", "unknown"]
    er_tel: str | None
    address: str
    is_cached: bool
    is_stale: bool


BED_CACHE_TTL_SECONDS = 60


@dataclass
class BedCacheEntry:
    """시도·시군구 한 조합에 대한 병상 조회 결과 캐시."""

    beds: list[HospitalCandidate]
    fetched_at: float = field(default_factory=time.time)

    def is_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.fetched_at) > BED_CACHE_TTL_SECONDS


@dataclass
class ERFinderState:
    """대화(세션) 도중 계속 바뀌는 값의 모음.

    Context와 달리 매 turn 값이 바뀌는 게 정상이라 frozen을 걸지 않았다.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    current_location: Location | None = None
    triage: TriageState | None = None
    search_radius_km: int = 5
    candidates: list[HospitalCandidate] = field(default_factory=list)
    bed_cache: dict[tuple[str, str], BedCacheEntry] = field(default_factory=dict)

    # ---------- bed_cache 헬퍼 (60초 TTL) ----------

    def get_cached_beds(self, sido: str, sigungu: str) -> list[HospitalCandidate] | None:
        """캐시가 있고 만료 전이면 반환하고, 없거나 만료됐으면 None."""
        entry = self.bed_cache.get((sido, sigungu))
        if entry is None or entry.is_expired():
            return None
        return entry.beds

    def set_cached_beds(self, sido: str, sigungu: str, beds: list[HospitalCandidate]) -> None:
        self.bed_cache[(sido, sigungu)] = BedCacheEntry(beds=beds)

    # ---------- 반경 확대 헬퍼 ----------

    def expand_radius(self, sequence: list[int]) -> bool:
        """search_radius_km을 sequence(이동수단별 반경 순서)의 다음 단계로 넓힌다.

        더 넓힐 단계가 없으면 False. "현재값보다 큰 첫 값"으로 이동하므로
        현재값이 sequence에 없어도 반경이 줄어들지 않는다.
        """
        for candidate in sequence:
            if candidate > self.search_radius_km:
                self.search_radius_km = candidate
                return True
        return False

    # ---------- 위치 갱신 헬퍼 ----------

    def reset_location_scoped(self, *, initial_radius_km: int = 5) -> None:
        """위치가 바뀌면 이전 위치 기준으로 쌓인 반경·후보·캐시를 초기화한다.

        initial_radius_km은 이동수단별 반경 시작값(walk=3, car/transit=5)이다.
        State는 transport를 모르므로 호출자가 넘겨줘야 한다.
        """
        self.search_radius_km = initial_radius_km
        self.candidates = []
        self.bed_cache = {}
