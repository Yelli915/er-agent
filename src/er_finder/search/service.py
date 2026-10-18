"""Search operations validate their order and retain evidence outside model control."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, RLock
from zoneinfo import ZoneInfo

from er_finder.guardrails.input_guard import assess_input
from er_finder.guardrails.pii import mask_pii
from er_finder.memory.state import ERFinderState
from er_finder.search.distance import distance_km
from er_finder.search.errors import ToolOrderError
from er_finder.search.location import extract_location
from er_finder.search.planner import CallPlanner
from er_finder.search.radius import radius_sequence
from er_finder.search.reply import ReplyBuilder

__all__ = ["SearchSession", "ToolOrderError", "extract_location"]


class SearchSession(CallPlanner, ReplyBuilder, ERFinderState):
    def __init__(self, provider, transport="car", now=None):
        super().__init__()
        self.provider = provider
        self.transport = transport
        self.now = now or (lambda: datetime.now(ZoneInfo("Asia/Seoul")))
        self._lock = RLock()
        self.text = ""
        self.symptom_text = ""
        self.location_query = None
        self.assessment = assess_input("")
        self.triage = self.assessment.triage
        self.facilities, self.beds, self.acceptance, self.details = {}, {}, {}, {}
        self.audit = []
        self.radii = radius_sequence(transport)
        self.radius_index = 0
        self.listed = self.beds_checked = self.severe_checked = False
        self.geo_attempted = False
        self.evidence_corrections = 0
        self.lookup_error = None
        self.force_refresh = False
        self.specialty_requested = False
        self.data_timestamp = self.now().isoformat()
        self.severe_region_cache = {}
        self.beds_ready = Event()
        self.allow_severe_batch = False
        self.planned_calls = None

    @property
    def location(self):
        """Keep the existing search API backed by memory's current_location field."""
        return self.current_location

    @location.setter
    def location(self, value):
        self.current_location = value

    @property
    def radius(self):
        return self.radii[self.radius_index]

    def begin(self, text, home_address=None, *, force_refresh=False, assessment=None):
        had_context = bool(self.symptom_text)
        self.text = mask_pii(text)
        self.assessment = assessment or assess_input(self.text)
        # A follow-up that only changes search options retains the clinical retrieval context.
        followup = any(w in self.text for w in ("다시", "같은 위치", "반경", "지난번"))
        self.location_query = extract_location(self.text)
        # Full addresses may include province names, road numbers and Korean endings.
        # A location answer with no new symptom must not erase the prior assessment.
        new_symptoms = re.search(
            r"아프|통증|불편|답답|식은땀|부었|붓|찢|다쳤|넘어|구토|토하|설사|두통|어지|기침|발열|고열|출혈|화상|의식|경련|숨|마비",
            self.text,
        )
        location_only = bool(self.location_query) and not new_symptoms
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
        self.radii = radius_sequence(self.transport, self.text)
        self.radius_index = 0
        self.geo_attempted = False
        self.listed = self.beds_checked = self.severe_checked = False
        self.facilities, self.beds, self.acceptance, self.details = {}, {}, {}, {}
        self.candidates, self.audit = [], []
        self.evidence_corrections = 0
        self.lookup_error = None
        self.force_refresh = force_refresh
        self.specialty_requested = any(
            w in self.text for w in ("소아", "봉합", "정형외과", "산부인과")
        )
        self.data_timestamp = self.now().isoformat()
        self.severe_region_cache = {}
        self.beds_ready = Event()
        self.allow_severe_batch = False
        self.planned_calls = None

    def fail_lookup(self):
        with self._lock:
            self.lookup_error = "외부 정보 조회에 실패하여 현재 후보를 확인할 수 없습니다."
            self.candidates = []
            self.beds_ready.set()

    def status(self):
        return {
            "next_calls": self.next_calls(),
            "draft": self.make_reply().model_dump(),
            "demo": bool(getattr(self.provider, "is_demo", False)),
        }

    def geocode(self, query):
        if self.assessment.blocked or self.assessment.greeting or self.geo_attempted or self.listed:
            raise ToolOrderError("현재 단계에서는 위치 검색을 할 수 없습니다.")
        query = mask_pii(query).strip()
        if not query or len(query) > 150:
            raise ToolOrderError("구체적인 위치를 150자 이내로 입력하세요.")
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

    def list_nearby_ers(self, lat, lon, radius_km=5):
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

    def _reuse_acceptance(self):
        self.acceptance = {}
        for hpid, row in self.facilities.items():
            prior = self.severe_region_cache.get((row.get("sido"), row.get("sigungu")), {})
            if hpid in prior:
                self.acceptance[hpid] = prior[hpid]

    def get_er_bed_status(self, regions, hpids):
        self._require("get_er_bed_status", dict(regions=regions, hpids=hpids))
        self.audit.append({"name": "get_er_bed_status", "region_count": len(regions)})

        def fetch(region):
            ids = [
                h
                for h in hpids
                if (self.facilities[h].get("sido"), self.facilities[h].get("sigungu"))
                == (region["sido"], region["sigungu"])
            ]
            rows = self.provider.get_er_bed_status(
                region["sido"], region["sigungu"], ids, force_refresh=self.force_refresh
            )
            return [r for r in rows if r.get("hpid") in ids]

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

    def get_severe_acceptance(self, regions, condition):
        if not self.beds_checked:
            if not self.allow_severe_batch:
                raise ToolOrderError("먼저 병상 조회를 완료해야 합니다.")
            if not self.beds_ready.wait(timeout=35):
                raise ToolOrderError("병상 조회 완료를 확인하지 못했습니다.")
            if not self.beds_checked:
                raise ToolOrderError("병상 조회가 실패하여 중증 조회를 진행하지 않습니다.")
        if self.triage.condition == condition and not any(
            isinstance(b.get("er_beds_available"), int) and b["er_beds_available"] > 0
            for b in self.beds.values()
        ):
            return {"acceptance": [], **self.status()}
        self._require("get_severe_acceptance", dict(regions=regions, condition=condition))
        self.audit.append({"name": "get_severe_acceptance", "region_count": len(regions)})

        def fetch(region):
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

    def get_er_detail(self, hpid):
        # Model tool calls for the top three may execute concurrently.
        with self._lock:
            self._require("get_er_detail", dict(hpid=hpid))
        result = self.provider.get_er_detail(hpid)
        with self._lock:
            self.details[hpid] = result or {}
            self.audit.append({"name": "get_er_detail", "hpid": hpid})
            return {"hpid": hpid, "detail": self.details[hpid], **self.status()}

    def snapshot(self):
        return {
            "current_location": self.location,
            "triage": self.triage.model_dump(),
            "search_radius_km": self.radius,
            "candidates": [h.model_dump() for h in self.candidates],
        }

    def clear(self):
        provider, transport, now = self.provider, self.transport, self.now
        self.__init__(provider, transport=transport, now=now)
