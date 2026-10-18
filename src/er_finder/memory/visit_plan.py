"""save_visit_plan 관련 로직 — HITL 승인 이후에만 Store에 쓴다.

승인 요청을 사용자에게 띄우는 인터럽트 자체는 미들웨어 쪽 책임이고,
여기서는 "승인됐으면 저장, 거절됐으면 저장 안 함"이라는 결정 이후의
흐름만 구현한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from er_finder.memory.store import ERFinderStore, VisitRecord


class VisitPlanStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PendingVisitPlan:
    """승인 대기 중인 방문 계획 1건.

    save_visit_plan 도구가 호출되면 바로 Store에 쓰지 않고 먼저 이 객체를
    만들어 HITL 인터럽트에 실어 보낸다. 승인/거절 결과는 decide()로 반영한다.
    """

    hpid: str
    name: str
    symptom_summary: str
    status: VisitPlanStatus = VisitPlanStatus.PENDING

    def __post_init__(self) -> None:
        if not self.hpid or not self.name:
            raise ValueError("hpid와 name은 필수입니다.")


@dataclass
class VisitPlanResult:
    """save_visit_plan 도구가 최종적으로 돌려주는 값.

    reason은 도구 반환값에는 포함하지 않고 로그/디버깅용으로만 쓴다.
    """

    saved: bool
    visit_id: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"saved": self.saved, "visit_id": self.visit_id}


class VisitPlanService:
    """PendingVisitPlan의 승인/거절 이후 처리를 담당한다."""

    def __init__(self, store: ERFinderStore) -> None:
        self._store = store

    def decide(self, plan: PendingVisitPlan, *, approved: bool) -> VisitPlanResult:
        """승인/거절 결과를 받아 최종 저장 여부를 확정한다.

        이미 결정된 plan에 다시 decide()를 호출하면 RuntimeError를 던진다.
        HITL 인터럽트는 한 번만 응답을 받는 게 정상이므로, 두 번째 호출은
        버그로 보고 조용히 넘어가지 않는다.
        """
        if plan.status is not VisitPlanStatus.PENDING:
            raise RuntimeError(
                f"이미 결정된 방문 계획입니다 (status={plan.status.value}). "
                "재결정은 지원하지 않습니다."
            )

        if not approved:
            plan.status = VisitPlanStatus.REJECTED
            return VisitPlanResult(saved=False, reason="user_rejected")

        plan.status = VisitPlanStatus.APPROVED
        try:
            record: VisitRecord = self._store.add_visit(
                hpid=plan.hpid,
                name=plan.name,
                symptom_summary=plan.symptom_summary,
            )
        except Exception:
            return VisitPlanResult(saved=False, reason="store_write_failed")

        return VisitPlanResult(saved=True, visit_id=record["visit_id"])
