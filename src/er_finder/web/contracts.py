"""Validated display contract for the real agent boundary."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from er_finder.memory.store import ERFinderStore
from er_finder.memory.visit_plan import PendingVisitPlan

DISCLAIMER = "응급실 상황은 수시로 변하므로 방문 전 전화 확인을 권장합니다"


class UIHospital(BaseModel):
    hpid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    distance_km: float = Field(ge=0, allow_inf_nan=False)
    er_beds_available: int | None
    beds_updated_at: str | None
    accepts_condition: Literal["yes", "no", "unknown"] = "unknown"
    er_tel: str | None = None
    address: str
    is_cached: bool = False
    is_stale: bool = False


class UIReply(BaseModel):
    severity: Literal["critical", "urgent", "standard"]
    call_119_first: bool
    search_radius_km: int = Field(gt=0, le=30)
    hospitals: list[UIHospital] = Field(default_factory=list, max_length=3)
    no_candidate_reason: str | None = None
    data_timestamp: str
    next_action: str
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def consistent_display(self) -> "UIReply":
        if self.call_119_first != (self.severity == "critical"):
            raise ValueError("critical 응답은 119 우선 안내와 일치해야 합니다")
        if not self.hospitals and not self.no_candidate_reason:
            raise ValueError("후보가 없으면 화면에 표시할 이유가 필요합니다")
        return self


@dataclass
class WebResult:
    reply: UIReply | None = None
    pending_approval: PendingVisitPlan | None = None
    note: str | None = None


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str


class Backend(Protocol):
    """Runner adapters implement this port; construction stays outside UI."""

    user_id: str
    profile: ERFinderStore

    def status(self) -> Any: ...
    def connect(self, *, transport: str = "car") -> Any: ...
    def chat(self, text: str, *, transport: str, force_refresh: bool = False) -> WebResult: ...
    def select_hospital(self, hpid: str) -> WebResult: ...
    def approve(self, approved: bool) -> WebResult: ...
    def reset(self) -> None: ...
    def forget(self) -> None: ...
