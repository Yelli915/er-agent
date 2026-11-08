"""Search ports and internal messages; provider rows remain untrusted API data."""

from typing import Any, Protocol, TypedDict


class Region(TypedDict):
    sido: str
    sigungu: str


class PlannedCall(TypedDict):
    name: str
    args: dict[str, Any]


class SearchProvider(Protocol):
    """Raw provider rows are validated and normalized by SearchSession."""

    def geocode(self, query: str) -> dict[str, Any]: ...

    def list_nearby_ers(
        self, lat: float, lon: float, radius_km: int = 5
    ) -> list[dict[str, Any]]: ...

    def get_er_bed_status(
        self, sido: str, sigungu: str, hpids: list[str], *, force_refresh: bool = False
    ) -> list[dict[str, Any]]: ...

    def get_severe_acceptance(
        self, sido: str, sigungu: str, condition: str
    ) -> list[dict[str, Any]]: ...

    def get_er_detail(self, hpid: str) -> dict[str, Any]: ...

    def close(self) -> None: ...
