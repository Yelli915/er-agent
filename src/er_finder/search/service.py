"""Search operations validate their order and retain evidence outside model control."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, RLock
from typing import Any, ParamSpec, TypeVar
from zoneinfo import ZoneInfo

from er_finder.errors import FailureMetadata, FailureStage, OperationError
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
from er_finder.search.candidates import select_candidates
from er_finder.search.contracts import PlannedCall, Region, SearchProvider
from er_finder.search.distance import distance_km
from er_finder.search.errors import ToolOrderError
from er_finder.search.keywords import is_followup, mentions_new_symptom, requests_specialty
from er_finder.search.location import extract_location, needs_location_clarification
from er_finder.search.planner import CallPlanner
from er_finder.search.radius import radius_sequence
from er_finder.search.reply import ReplyBuilder

P = ParamSpec("P")
T = TypeVar("T")


__all__ = ["SearchSession", "ToolOrderError", "extract_location"]


class SearchSession:
    """Search state plus the ordered, validated provider operations that fill it.

    Next-call planning and reply composition live in CallPlanner and ReplyBuilder;
    this class keeps their public entry points so callers see one object.
    """

    def __init__(
        self,
        provider: SearchProvider,
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
        self._replies = ReplyBuilder(self)
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
        self.failure: FailureMetadata | None = None
        self.force_refresh = force_refresh
        self.specialty_requested = requests_specialty(text)
        self.data_timestamp = self.now().isoformat()
        self.severe_region_cache: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
        self.beds_ready = Event()
        self.allow_severe_batch = False

    @property
    def lookup_error(self) -> str | None:
        """Compatibility view of lookup failures; failure metadata owns the state."""
        return (
            LOOKUP_FAILED if self.failure and self.failure["failure_category"] == "lookup" else None
        )

    @property
    def radius(self) -> Radius:
        return self.radii[self.radius_index]

    def _refresh_candidates(self) -> None:
        """Update derived candidates only when lookup evidence changes."""
        if self.failure or self.assessment.blocked or self.assessment.greeting:
            self.candidates = []
            return
        self.candidates = select_candidates(
            self.facilities,
            self.beds,
            self.acceptance,
            self.details,
            self.triage.condition,
            self.now(),
        )

    def next_calls(self) -> list[PlannedCall]:
        return self._planner.next_calls()

    def regions(self) -> list[Region]:
        return self._planner.regions()

    def make_reply(self) -> ERSearchReply:
        return self._replies.make_reply()

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
        workplace_unknown = needs_location_clarification(self.text)
        new_symptom = mentions_new_symptom(self.text)
        self.location_query = self.assessment.location_query or extract_location(self.text)
        explicit_location = bool(self.assessment.location_query) or (
            bool(self.location_query) and not workplace_unknown
        )
        unresolved = not explicit_location and (
            workplace_unknown or self.assessment.needs_location
        )
        self.assessment.needs_location = unresolved
        if unresolved:
            self.location_query = None
            self.location = None
        preserve = (
            had_context
            and (bool(self.location_query) or followup or unresolved)
            and not new_symptom
            and self.assessment.triage.severity == "standard"
        )
        if not preserve:
            self.triage = self.assessment.triage
            self.symptom_text = self.text
        if self.location_query:
            self.location = None
        elif not self.location and home_address and not unresolved:
            self.location_query = home_address
        self._reset_lookup(self.text, force_refresh=force_refresh)

    def fail_run(self, failure: FailureMetadata) -> None:
        """Keep the first sanitized failure; never retain exceptions or traceback text."""
        with self._lock:
            if self.failure is None:
                self.failure = failure.copy()  # detach from the caller
            self.candidates = []
            self.beds_ready.set()

    def _call_provider(
        self, stage: FailureStage, call: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        try:
            return call(*args, **kwargs)
        except Exception as error:
            raise OperationError("lookup", stage) from error

    def status(self, *, next_calls: list[PlannedCall] | None = None) -> dict[str, Any]:
        return {
            "next_calls": self.next_calls() if next_calls is None else next_calls,
            "draft": self.make_reply().model_dump(),
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
        result = self._call_provider("geocode", self.provider.geocode, query)
        if result.get("found"):
            try:
                lat, lon = float(result["lat"]), float(result["lon"])
                distance_km(lat, lon, 0, 0)
                self.location = {**result, "lat": lat, "lon": lon}
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
        rows = self._call_provider(
            "list_nearby_ers", self.provider.list_nearby_ers, lat, lon, radius_km
        )
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

    def get_er_bed_status(self, regions: list[Region], hpids: list[str]) -> dict[str, Any]:
        self._require("get_er_bed_status", dict(regions=regions, hpids=hpids))
        self.audit.append({"name": "get_er_bed_status", "region_count": len(regions)})

        ids_by_region: dict[tuple[Any, Any], list[str]] = {}
        for h in hpids:
            fac = self.facilities[h]
            ids_by_region.setdefault((fac.get("sido"), fac.get("sigungu")), []).append(h)

        def fetch(region: Region) -> list[dict[str, Any]]:
            ids = ids_by_region.get((region["sido"], region["sigungu"]), [])
            id_set = set(ids)
            rows = self._call_provider(
                "get_er_bed_status",
                self.provider.get_er_bed_status,
                region["sido"],
                region["sigungu"],
                ids,
                force_refresh=self.force_refresh,
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
            self._refresh_candidates()
            return {"beds": list(self.beds.values()), **self.status()}
        finally:
            self.beds_ready.set()

    def get_severe_acceptance(self, regions: list[Region], condition: str) -> dict[str, Any]:
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
            region: Region,
        ) -> tuple[tuple[str, str], dict[str, dict[str, Any]]]:
            rows = self._call_provider(
                "get_severe_acceptance",
                self.provider.get_severe_acceptance,
                region["sido"],
                region["sigungu"],
                condition,
            )
            return (region["sido"], region["sigungu"]), {
                r["hpid"]: r for r in rows if r.get("hpid")
            }

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(regions)))) as pool:
            for key, rows in pool.map(fetch, regions):
                self.severe_region_cache[key] = rows
        self._reuse_acceptance()
        self.severe_checked = True
        self._refresh_candidates()
        return {"acceptance": list(self.acceptance.values()), **self.status()}

    def get_er_detail(self, hpid: str) -> dict[str, Any]:
        # Model tool calls for the top three may execute concurrently.
        with self._lock:
            self._require("get_er_detail", dict(hpid=hpid))
        result = self._call_provider("get_er_detail", self.provider.get_er_detail, hpid)
        with self._lock:
            self.details[hpid] = result or {}
            self._refresh_candidates()
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
