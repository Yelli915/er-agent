"""Agent-only adapters over the shared memory contracts."""

from __future__ import annotations

import re
from threading import RLock

from langchain.agents import AgentState

from er_finder.guardrails.pii import mask_pii
from er_finder.memory.store import ERFinderStore
from er_finder.memory.visit_plan import PendingVisitPlan, VisitPlanService, VisitPlanStatus
from er_finder.models import ERSearchReply, HospitalCandidate


class ERGraphState(AgentState[ERSearchReply], total=False):
    current_location: dict | None
    triage: dict
    search_radius_km: int
    candidates: list[dict]


def selected_index(text: str) -> int | None:
    match = re.fullmatch(
        r"\s*([1-9]\d*)\s*(?:번)?\s*(?:으로\s*갈게요|으로|이요|요|선택|갈게요)?[.!]?\s*",
        text,
    )
    return int(match[1]) - 1 if match else None


class VisitSelection:
    """Keep the selected evidence and one explicit approval outside model control."""

    def __init__(self, profiles: ERFinderStore, session):
        self._service = VisitPlanService(profiles)
        self._session = session
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._plan: PendingVisitPlan | None = None
            self._search_timestamp: str | None = None
            self.approved = False
            self.decided = False
            self.saved = False
            self.result = None

    @property
    def pending(self) -> dict | None:
        if self._plan is None:
            return None
        return {
            "hpid": self._plan.hpid,
            "name": self._plan.name,
            "symptom_summary": self._plan.symptom_summary,
        }

    def prepare(self, hospital: HospitalCandidate, symptom_summary: str) -> None:
        with self._lock:
            self.reset()
            if self._session.next_calls():
                raise ValueError("병원 정보 조회를 완료한 뒤 후보를 선택하세요.")
            candidate = next((h for h in self._session.candidates if h.hpid == hospital.hpid), None)
            if candidate is None or candidate.model_dump() != hospital.model_dump():
                raise ValueError("현재 검색 결과에 있는 후보만 선택할 수 있습니다.")
            self._plan = PendingVisitPlan(
                hpid=hospital.hpid,
                name=hospital.name,
                symptom_summary=mask_pii(symptom_summary).strip()[:250],
            )
            self._search_timestamp = self._session.data_timestamp

    def validate(self, arguments: dict) -> None:
        with self._lock:
            if self._plan is None or self._plan.status is not VisitPlanStatus.PENDING or self.saved:
                raise ValueError("저장 가능한 방문 계획이 없습니다.")
            if arguments != self.pending:
                raise ValueError("선택한 병원 및 증상 요약과 일치하지 않습니다.")
            if self._search_timestamp != self._session.data_timestamp:
                raise ValueError("검색 정보가 바뀌었습니다. 현재 후보를 다시 선택하세요.")
            if not any(
                h.hpid == self._plan.hpid and h.name == self._plan.name
                for h in self._session.candidates
            ):
                raise ValueError("선택한 병원이 현재 후보에 없습니다.")

    def authorize(self, approved: bool) -> None:
        with self._lock:
            self.validate(self.pending)
            if self.decided:
                raise ValueError("이미 결정된 방문 계획입니다.")
            self.approved, self.decided = approved, True
            if not approved:
                self.result = self._service.decide(self._plan, approved=False)

    def save(self, hpid: str, name: str, symptom_summary: str) -> dict:
        with self._lock:
            self.validate(dict(hpid=hpid, name=name, symptom_summary=symptom_summary))
            if not self.decided or not self.approved:
                raise ValueError("사용자의 명시적인 저장 승인이 필요합니다.")
            self.result = self._service.decide(self._plan, approved=True)
            self.saved = self.result.saved
            return self.result.as_dict()
