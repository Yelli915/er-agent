"""Agent-only adapters over the shared memory contracts."""

from __future__ import annotations

import re
from threading import RLock
from typing import Any

from langchain.agents import AgentState

from er_finder.guardrails.pii import mask_pii
from er_finder.memory.store import ERFinderStore
from er_finder.memory.visit_plan import (
    PendingVisitPlan,
    VisitPlanResult,
    VisitPlanService,
    VisitPlanStatus,
)
from er_finder.messages import (
    VISIT_ALREADY_DECIDED,
    VISIT_APPROVAL_REQUIRED,
    VISIT_FINISH_LOOKUP_FIRST,
    VISIT_MISMATCH,
    VISIT_NO_SAVABLE_PLAN,
    VISIT_NOT_A_CANDIDATE,
    VISIT_ONLY_CURRENT_CANDIDATES,
    VISIT_SEARCH_CHANGED,
)
from er_finder.models import ERSearchReply, HospitalCandidate


class ERGraphState(AgentState[ERSearchReply], total=False):
    current_location: dict[str, Any] | None
    triage: dict[str, Any]
    search_radius_km: int
    candidates: list[dict[str, Any]]


def selected_index(text: str) -> int | None:
    match = re.fullmatch(
        r"\s*([1-9]\d*)\s*(?:번)?\s*(?:으로\s*갈게요|으로|이요|요|선택|갈게요)?[.!]?\s*",
        text,
    )
    return int(match[1]) - 1 if match else None


class VisitSelection:
    """Keep the selected evidence and one explicit approval outside model control."""

    def __init__(self, profiles: ERFinderStore, session: Any) -> None:
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
            self.result: VisitPlanResult | None = None

    @property
    def pending(self) -> dict[str, Any] | None:
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
                raise ValueError(VISIT_FINISH_LOOKUP_FIRST)
            candidate = next((h for h in self._session.candidates if h.hpid == hospital.hpid), None)
            if candidate is None or candidate.model_dump() != hospital.model_dump():
                raise ValueError(VISIT_ONLY_CURRENT_CANDIDATES)
            self._plan = PendingVisitPlan(
                hpid=hospital.hpid,
                name=hospital.name,
                symptom_summary=mask_pii(symptom_summary).strip()[:250],
            )
            self._search_timestamp = self._session.data_timestamp

    def validate(self, arguments: dict[str, Any]) -> None:
        with self._lock:
            if self._plan is None or self._plan.status is not VisitPlanStatus.PENDING or self.saved:
                raise ValueError(VISIT_NO_SAVABLE_PLAN)
            if arguments != self.pending:
                raise ValueError(VISIT_MISMATCH)
            if self._search_timestamp != self._session.data_timestamp:
                raise ValueError(VISIT_SEARCH_CHANGED)
            if not any(
                h.hpid == self._plan.hpid and h.name == self._plan.name
                for h in self._session.candidates
            ):
                raise ValueError(VISIT_NOT_A_CANDIDATE)

    def authorize(self, approved: bool) -> None:
        with self._lock:
            pending = self.pending
            if pending is None:
                raise ValueError(VISIT_NO_SAVABLE_PLAN)
            self.validate(pending)
            if self.decided:
                raise ValueError(VISIT_ALREADY_DECIDED)
            self.approved, self.decided = approved, True
            if not approved and self._plan is not None:
                self.result = self._service.decide(self._plan, approved=False)

    def save(self, hpid: str, name: str, symptom_summary: str) -> dict[str, Any]:
        with self._lock:
            self.validate(dict(hpid=hpid, name=name, symptom_summary=symptom_summary))
            if not self.decided or not self.approved:
                raise ValueError(VISIT_APPROVAL_REQUIRED)
            if self._plan is None:
                raise ValueError(VISIT_NO_SAVABLE_PLAN)
            result = self._service.decide(self._plan, approved=True)
            self.result = result
            self.saved = result.saved
            return result.as_dict()
