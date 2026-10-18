"""Workflow ordering: which tool calls the server accepts next."""

from er_finder.search.candidates import select_candidates
from er_finder.search.errors import ToolOrderError


class CallPlanner:
    """Mixin for SearchSession; reads search state, never mutates provider data."""

    def regions(self):
        return [
            dict(sido=s, sigungu=g)
            for s, g in sorted(
                {
                    (r["sido"], r["sigungu"])
                    for r in self.facilities.values()
                    if r.get("sido") and r.get("sigungu") is not None
                }
            )
        ]

    def _candidates(self):
        if self.lookup_error:
            self.candidates = []
            return self.candidates
        self.candidates = select_candidates(
            self.facilities,
            self.beds,
            self.acceptance,
            self.details,
            self.triage.condition,
            self.now(),
        )
        return self.candidates

    def next_calls(self):
        if self.lookup_error:
            return []
        if self.assessment.blocked or self.assessment.greeting:
            return []
        if not self.location:
            if self.location_query and not self.geo_attempted:
                return [{"name": "geocode", "args": {"query": self.location_query}}]
            return []
        if not self.listed:
            return [
                {
                    "name": "list_nearby_ers",
                    "args": {
                        "lat": self.location["lat"],
                        "lon": self.location["lon"],
                        "radius_km": self.radius,
                    },
                }
            ]
        if not self.beds_checked:
            calls = [
                {
                    "name": "get_er_bed_status",
                    "args": {"regions": self.regions(), "hpids": sorted(self.facilities)},
                }
            ]
            new_regions = [
                r
                for r in self.regions()
                if (r["sido"], r["sigungu"]) not in self.severe_region_cache
            ]
            if self.triage.condition and new_regions:
                calls.append(
                    {
                        "name": "get_severe_acceptance",
                        "args": {"regions": new_regions, "condition": self.triage.condition},
                    }
                )
            return calls
        positive_beds = any(
            isinstance(b.get("er_beds_available"), int) and b["er_beds_available"] > 0
            for b in self.beds.values()
        )
        if self.triage.condition and not self.severe_checked and positive_beds:
            return [
                {
                    "name": "get_severe_acceptance",
                    "args": {
                        "regions": [
                            r
                            for r in self.regions()
                            if (r["sido"], r["sigungu"]) not in self.severe_region_cache
                        ],
                        "condition": self.triage.condition,
                    },
                }
            ]
        candidates = self._candidates()
        if not candidates and self.radius_index < len(self.radii) - 1:
            return [
                {
                    "name": "list_nearby_ers",
                    "args": {
                        "lat": self.location["lat"],
                        "lon": self.location["lon"],
                        "radius_km": self.radii[self.radius_index + 1],
                    },
                }
            ]
        return [
            {"name": "get_er_detail", "args": {"hpid": h.hpid}}
            for h in candidates
            if h.hpid not in self.details
        ]

    def _require(self, name, arguments):
        if not any(c["name"] == name and c["args"] == arguments for c in self.next_calls()):
            raise ToolOrderError("도구 순서 또는 인자가 현재 검색 상태와 일치하지 않습니다.")
