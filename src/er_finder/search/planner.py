"""Workflow ordering: which tool calls the server accepts next."""

from typing import TYPE_CHECKING, Any

from er_finder.messages import (
    TOOL_ORDER_MISMATCH,
)
from er_finder.models import HospitalCandidate
from er_finder.search.candidates import select_candidates
from er_finder.search.errors import ToolOrderError

if TYPE_CHECKING:
    from er_finder.search.service import SearchSession


class CallPlanner:
    """Decides the next server-approved calls from a SearchSession's state."""

    def __init__(self, session: "SearchSession") -> None:
        self.session = session
        self.refreshed = False

    def regions(self) -> list[dict[str, str]]:
        s = self.session
        return [
            dict(sido=sido, sigungu=sigungu)
            for sido, sigungu in sorted(
                {
                    (r["sido"], r["sigungu"])
                    for r in s.facilities.values()
                    if r.get("sido") and r.get("sigungu") is not None
                }
            )
        ]

    def refresh_candidates(self) -> list[HospitalCandidate]:
        s = self.session
        self.refreshed = True
        if s.lookup_error:
            s.candidates = []
            return s.candidates
        s.candidates = select_candidates(
            s.facilities,
            s.beds,
            s.acceptance,
            s.details,
            s.triage.condition,
            s.now(),
        )
        return s.candidates

    def next_calls(self) -> list[dict[str, Any]]:
        s = self.session
        if s.lookup_error:
            return []
        if s.assessment.blocked or s.assessment.greeting:
            return []
        if not s.location:
            if s.location_query and not s.geo_attempted:
                return [{"name": "geocode", "args": {"query": s.location_query}}]
            return []
        if not s.listed:
            return [
                {
                    "name": "list_nearby_ers",
                    "args": {
                        "lat": s.location["lat"],
                        "lon": s.location["lon"],
                        "radius_km": s.radius,
                    },
                }
            ]
        regions = self.regions()
        new_regions = [r for r in regions if (r["sido"], r["sigungu"]) not in s.severe_region_cache]
        if not s.beds_checked:
            calls = [
                {
                    "name": "get_er_bed_status",
                    "args": {"regions": regions, "hpids": sorted(s.facilities)},
                }
            ]
            if s.triage.condition and new_regions:
                calls.append(
                    {
                        "name": "get_severe_acceptance",
                        "args": {"regions": new_regions, "condition": s.triage.condition},
                    }
                )
            return calls
        positive_beds = any(
            isinstance(b.get("er_beds_available"), int) and b["er_beds_available"] > 0
            for b in s.beds.values()
        )
        if s.triage.condition and not s.severe_checked and positive_beds:
            return [
                {
                    "name": "get_severe_acceptance",
                    "args": {
                        "regions": new_regions,
                        "condition": s.triage.condition,
                    },
                }
            ]
        candidates = self.refresh_candidates()
        if not candidates and s.radius_index < len(s.radii) - 1:
            return [
                {
                    "name": "list_nearby_ers",
                    "args": {
                        "lat": s.location["lat"],
                        "lon": s.location["lon"],
                        "radius_km": s.radii[s.radius_index + 1],
                    },
                }
            ]
        return [
            {"name": "get_er_detail", "args": {"hpid": h.hpid}}
            for h in candidates
            if h.hpid not in s.details
        ]

    def _require(self, name: str, arguments: dict[str, Any]) -> None:
        if not any(c["name"] == name and c["args"] == arguments for c in self.next_calls()):
            raise ToolOrderError(TOOL_ORDER_MISMATCH)
