"""Search operations validate their order and retain evidence outside model control."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, RLock
from typing import Any
from zoneinfo import ZoneInfo

from er_finder.guardrails.input_guard import InputAssessment, assess_input
from er_finder.guardrails.pii import mask_pii
from er_finder.messages import (
    BEDS_FAILED_SKIP_SEVERE,
    BEDS_NOT_CONFIRMED,
    BEDS_REQUIRED_FIRST,
    GEOCODE_NOT_ALLOWED,
    LOCATION_QUERY_INVALID,
    LOOKUP_FAILED,
)
from er_finder.models import ERSearchReply, HospitalCandidate, Radius
from er_finder.search.distance import distance_km
from er_finder.search.errors import ToolOrderError
from er_finder.search.keywords import is_followup, mentions_new_symptom, requests_specialty
from er_finder.search.location import extract_location
from er_finder.search.planner import CallPlanner
from er_finder.search.radius import radius_sequence
from er_finder.search.reply import ReplyBuilder

__all__ = ["SearchSession", "ToolOrderError", "extract_location"]


class SearchSession:
    """Search state plus the ordered, validated provider operations that fill it.

    Next-call planning and reply composition live in CallPlanner and ReplyBuilder;
    this class keeps their public entry points so callers see one object.
    """

    def __init__(
        self,
        provider: Any,
        transport: str = "car",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.transport = transport
        self.now = now or (lambda: datetime.now(ZoneInfo("Asia/Seoul")))
        self._lock = RLock()
        self.text = ""
        self.symptom_text = ""
        self.location_query: str | None = None
        self.assessment = assess_input("")
        self.triage = self.assessment.triage
        self.location: dict[str, Any] | None = None
        self._planner = CallPlanner(self)
        self._replies = ReplyBuilder(self, self._planner)
        self._reset_lookup("")

    def _reset_lookup(self, text: str, *, force_refresh: bool = False) -> None:
        """Discard everything fetched for the previous request; symptom context is kept."""
        self.radii = radius_sequence(self.transport, text)
        self.radius_index = 0
        self.geo_attempted = False
        self.listed = self.beds_checked = self.severe_checked = False
        self.facilities: dict[str, dict[str, Any]] = {}
        self.beds: dict[str, dict[str, Any]] = {}
        self.acceptance: dict[str, dict[str, Any]] = {}
        self.details: dict[str, dict[str, Any]] = {}
        self.candidates: list[HospitalCandidate] = []
        self.audit: list[dict[str, Any]] = []
        self.evidence_corrections = 0
        self.lookup_error: str | None = None
        self.force_refresh = force_refresh
        self.specialty_requested = requests_specialty(text)
        self.data_timestamp = self.now().isoformat()
        self.severe_region_cache: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
        self.beds_ready = Event()
        self.allow_severe_batch = False
        self.planned_calls: list[dict[str, Any]] | None = None

    @property
    def radius(self) -> Radius:
        return self.radii[self.radius_index]

    def next_calls(self) -> list[dict[str, Any]]:
        return self._planner.next_calls()

    def regions(self) -> list[dict[str, str]]:
        return self._planner.regions()

    def make_reply(self, *, refresh: bool = True) -> ERSearchReply:
        return self._replies.make_reply(refresh=refresh)

    def check_evidence(self, proposal: Any) -> ERSearchReply:
        return self._replies.check_evidence(proposal)

    def _require(self, name: str, arguments: dict[str, Any]) -> None:
        self._planner._require(name, arguments)

    def begin(
        self,
        text: str,
        home_address: str | None = None,
        *,
        force_refresh: bool = False,
        assessment: InputAssessment | None = None,
    ) -> None:
        had_context = bool(self.symptom_text)
        self.text = mask_pii(text)
        self.assessment = assessment or assess_input(self.text)
        # A follow-up that only changes search options retains the clinical retrieval context.
        followup = is_followup(self.text)
        self.location_query = extract_location(self.text)
        location_only = bool(self.location_query) and not mentions_new_symptom(self.text)
        preserve = (
            had_context
            and (location_only or followup)
            and self.assessment.triage.severity == "standard"
        )
        if not preserve:
            self.triage = self.assessment.triage
            self.symptom_text = self.text
        if self.location_query:
            self.location = None
        elif not self.location and home_address:
            self.location_query = home_address
        self._reset_lookup(self.text, force_refresh=force_refresh)

    def fail_lookup(self) -> None:
        with self._lock:
            self.lookup_error = LOOKUP_FAILED
            self.candidates = []
            self.beds_ready.set()

    def status(self) -> dict[str, Any]:
        # next_calls() already refreshes candidates when it reaches that stage; reuse that pass.
        self._planner.refreshed = False
        calls = self.next_calls()
        return {
            "next_calls": calls,
            "draft": self.make_reply(refresh=not self._planner.refreshed).model_dump(),
            "demo": bool(getattr(self.provider, "is_demo", False)),
        }

    def geocode(self, query: str) -> dict[str, Any]:
        if self.assessment.blocked or self.assessment.greeting or self.geo_attempted or self.listed:
            raise ToolOrderError(GEOCODE_NOT_ALLOWED)
        query = mask_pii(query).strip()
        if not query or len(query) > 150:
            raise ToolOrderError(LOCATION_QUERY_INVALID)
        self.geo_attempted = True
        self.audit.append({"name": "geocode"})
        result = self.provider.geocode(query)
        if result.get("found"):
            try:
                distance_km(float(result["lat"]), float(result["lon"]), 0, 0)
                self.location = {**result, "lat": float(result["lat"]), "lon": float(result["lon"])}
            except (ValueError, TypeError, KeyError):
                self.location = None
        else:
            self.location = None
        return {"location": self.location, **self.status()}

    def list_nearby_ers(self, lat: float, lon: float, radius_km: int = 5) -> dict[str, Any]:
        self._require("list_nearby_ers", dict(lat=lat, lon=lon, radius_km=radius_km))
        if radius_km != self.radius:
            self.radius_index += 1
        self.audit.append({"name": "list_nearby_ers", "radius_km": radius_km})
        rows = self.provider.list_nearby_ers(lat, lon, radius_km)
        self.facilities = {}
        for row in rows:
            try:
                d = distance_km(lat, lon, float(row["lat"]), float(row["lon"]))
                if d <= radius_km and row.get("hpid"):
                    self.facilities[row["hpid"]] = {**row, "distance_km": d}
            except (ValueError, TypeError, KeyError):
                continue
        self.listed = True
        self.beds_checked = False
        self.beds_ready = Event()
        self.allow_severe_batch = False
        self.severe_checked = all(
            (r["sido"], r["sigungu"]) in self.severe_region_cache for r in self.regions()
        )
        self.beds, self.details = {}, {}
        self._reuse_acceptance()
        self.candidates = []
        return {
            "facilities": list(self.facilities.values()),
            "reused_acceptance": list(self.acceptance.values()),
            **self.status(),
        }

    def _reuse_acceptance(self) -> None:
        self.acceptance = {}
        for hpid, row in self.facilities.items():
            prior = self.severe_region_cache.get((row.get("sido"), row.get("sigungu")), {})
            if hpid in prior:
                self.acceptance[hpid] = prior[hpid]

    def get_er_bed_status(self, regions: list[dict[str, str]], hpids: list[str]) -> dict[str, Any]:
        self._require("get_er_bed_status", dict(regions=regions, hpids=hpids))
        self.audit.append({"name": "get_er_bed_status", "region_count": len(regions)})

        ids_by_region: dict[tuple[Any, Any], list[str]] = {}
        for h in hpids:
            fac = self.facilities[h]
            ids_by_region.setdefault((fac.get("sido"), fac.get("sigungu")), []).append(h)

        def fetch(region: dict[str, str]) -> list[dict[str, Any]]:
            ids = ids_by_region.get((region["sido"], region["sigungu"]), [])
            id_set = set(ids)
            rows = self.provider.get_er_bed_status(
                region["sido"], region["sigungu"], ids, force_refresh=self.force_refresh
            )
            return [r for r in rows if r.get("hpid") in id_set]

        try:
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(regions)))) as pool:
                for rows in pool.map(fetch, regions):
                    for row in rows:
                        self.beds[row["hpid"]] = row
            for hpid in hpids:
                self.beds.setdefault(
                    hpid,
                    {
                        "hpid": hpid,
                        "er_beds_available": None,
                        "beds_updated_at": None,
                        "is_cached": False,
                    },
                )
            self.beds_checked = True
            self.data_timestamp = self.now().isoformat()
            return {"beds": list(self.beds.values()), **self.status()}
        finally:
            self.beds_ready.set()

    def get_severe_acceptance(
        self, regions: list[dict[str, str]], condition: str
    ) -> dict[str, Any]:
        if not self.beds_checked:
            if not self.allow_severe_batch:
                raise ToolOrderError(BEDS_REQUIRED_FIRST)
            if not self.beds_ready.wait(timeout=35):
                raise ToolOrderError(BEDS_NOT_CONFIRMED)
            if not self.beds_checked:
                raise ToolOrderError(BEDS_FAILED_SKIP_SEVERE)
        if self.triage.condition == condition and not any(
            isinstance(b.get("er_beds_available"), int) and b["er_beds_available"] > 0
            for b in self.beds.values()
        ):
            return {"acceptance": [], **self.status()}
        self._require("get_severe_acceptance", dict(regions=regions, condition=condition))
        self.audit.append({"name": "get_severe_acceptance", "region_count": len(regions)})

        def fetch(
            region: dict[str, str],
        ) -> tuple[tuple[str, str], dict[str, dict[str, Any]]]:
            rows = self.provider.get_severe_acceptance(region["sido"], region["sigungu"], condition)
            return (region["sido"], region["sigungu"]), {
                r["hpid"]: r for r in rows if r.get("hpid")
            }

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(regions)))) as pool:
            for key, rows in pool.map(fetch, regions):
                self.severe_region_cache[key] = rows
        self._reuse_acceptance()
        self.severe_checked = True
        return {"acceptance": list(self.acceptance.values()), **self.status()}

    def get_er_detail(self, hpid: str) -> dict[str, Any]:
        # Model tool calls for the top three may execute concurrently.
        with self._lock:
            self._require("get_er_detail", dict(hpid=hpid))
        result = self.provider.get_er_detail(hpid)
        with self._lock:
            self.details[hpid] = result or {}
            self.audit.append({"name": "get_er_detail", "hpid": hpid})
            return {"hpid": hpid, "detail": self.details[hpid], **self.status()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_location": self.location,
            "triage": self.triage.model_dump(),
            "search_radius_km": self.radius,
            "candidates": [h.model_dump() for h in self.candidates],
        }

    def clear(self) -> None:
        provider, transport, now = self.provider, self.transport, self.now
        SearchSession.__init__(self, provider, transport=transport, now=now)
