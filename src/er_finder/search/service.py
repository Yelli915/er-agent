"""Search operations validate their order and retain evidence outside model control."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, RLock
from zoneinfo import ZoneInfo

from er_finder.memory.state import ERFinderState
from er_finder.models import DISCLAIMER, ERSearchReply
from er_finder.safety import assess_input, mask_pii, sanitize_prose
from er_finder.search.candidates import select_candidates
from er_finder.search.coordinates import extract_coordinates
from er_finder.search.distance import distance_km
from er_finder.search.radius import radius_sequence


class ToolOrderError(ValueError):
    pass


def extract_location(text: str) -> str | None:
    coordinates = extract_coordinates(text)
    if coordinates is not None:
        return coordinates
    # Offline extraction; live model may additionally extract a precise query from the input.
    if "우리 동네" in text or "우리동네" in text:
        return "우리 동네"
    known = (
        "강남구 역삼동",
        "역삼동",
        "강남구",
        "강남역",
        "경포대",
        "강릉",
        "영월군 상동읍",
        "상동읍",
        "영월군",
        "마포구 합정동",
        "합정동",
        "마포구",
    )
    for place in known:
        if place in text:
            return place
    # Strip Korean particles following administrative names rather than swallowing symptoms.
    tokens = re.findall(
        r"[가-힣0-9]+?(?:특별자치도|특별시|광역시|시|군|구|읍|면|동|역|대로|로)(?=\s|[,.!?]|(?:에서|이에요|인데|입니다|이야)|$)",
        text,
    )
    tokens = [t for t in tokens if t not in {"다시", "혹시", "응급시", "응급실로", "병원으로"}]
    if tokens:
        return " ".join(tokens)
    m = re.search(r"(?:^|[,.!?]\s*)([가-힣A-Za-z0-9 ]{2,35}?)\s*(?:근처|주변|앞에서)", text)
    return m[1].strip() if m else None


class SearchSession(ERFinderState):
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

    def make_reply(self):
        self._candidates()
        reason = None
        action = "후보 병원에 전화해 현재 수용 가능 여부를 확인하세요."
        if self.assessment.blocked:
            self.candidates = []
            reason = "서비스 범위 또는 입력 안전 정책에 따라 검색하지 않았습니다."
            action = "응급실 안내를 위해 현재 위치와 증상을 알려주세요."
        elif self.assessment.greeting:
            self.candidates = []
            reason = "응급실 찾기를 시작하려면 위치와 증상이 필요합니다."
            action = "현재 위치와 증상을 알려주세요."
        elif self.lookup_error:
            self.candidates = []
            reason = self.lookup_error
            action = "연결 상태를 확인하고 다시 조회하세요. 위급한 경우 즉시 119에 연락하세요."
        elif not self.location:
            reason = "현재 위치를 확인할 수 없습니다."
            action = "현재 위치를 주소·동 이름·건물명·역 이름으로 알려주세요."
        elif not self.candidates:
            if self.triage.condition == "중증외상" and any(
                b.get("er_beds_available", 0)
                and self.acceptance.get(hpid, {}).get("acceptable") is None
                for hpid, b in self.beds.items()
            ):
                reason = (
                    "중증외상은 현재 응급의료 API에서 병원별 수용 가능 여부를 "
                    "정확히 확인할 수 없어 후보 병원을 확정하지 않았습니다."
                )
                action = "119에 연락해 현재 이송 가능한 응급의료기관 안내를 받으세요."
            else:
                reason = (
                    f"반경 {self.radius}km 내 가용 병상과 필요한 수용 조건을 "
                    "확인한 후보가 없습니다."
                )
                if self.triage.condition and any(
                    b.get("er_beds_available", 0)
                    and self.acceptance.get(hpid, {}).get("acceptable") is None
                    for hpid, b in self.beds.items()
                ):
                    reason += " 중증 수용 여부 확인 불가인 병원이 포함되어 있습니다."
                action = "119에 연락해 수용 가능한 응급실 안내를 받으세요."
        if self.specialty_requested and self.candidates:
            action = "소아·봉합 등 필요한 진료의 현재 가능 여부는 병원에 전화해 확인하세요."
        if self.triage.severity == "critical":
            action = "119에 즉시 신고하세요. 검색 결과를 기다리지 마세요."
        return ERSearchReply(
            severity=self.triage.severity,
            call_119_first=self.triage.severity == "critical",
            search_radius_km=self.radius,
            hospitals=self.candidates,
            no_candidate_reason=reason,
            data_timestamp=self.data_timestamp,
            next_action=action,
            disclaimer=DISCLAIMER,
        )

    def check_evidence(self, proposal):
        canonical = self.make_reply().model_dump()
        if hasattr(proposal, "model_dump"):
            proposal = proposal.model_dump()
        if not isinstance(proposal, dict):
            return ERSearchReply.model_validate(canonical)
        rows = proposal.get("hospitals", [])
        if not isinstance(rows, list):
            rows = []
            self.evidence_corrections += 1
        proposed = {}
        for hospital in rows:
            if not isinstance(hospital, dict) or not isinstance(hospital.get("hpid"), str):
                self.evidence_corrections += 1
                continue
            proposed[hospital["hpid"]] = hospital
        self.evidence_corrections += sum(hpid not in self.facilities for hpid in proposed)
        for hospital in canonical["hospitals"]:
            candidate = proposed.get(hospital["hpid"])
            if candidate is not None:
                self.evidence_corrections += sum(
                    candidate.get(field) != value for field, value in hospital.items()
                )
        prose = proposal.get("next_action", "")
        if isinstance(prose, str) and sanitize_prose(prose) != prose:
            self.evidence_corrections += 1
        # A model proposal cannot change verified source data, the selected radius,
        # emergency instructions or provenance. Keep the source-derived reply.
        return ERSearchReply.model_validate(canonical)

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
