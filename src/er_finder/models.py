from datetime import datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DISCLAIMER: Final = "응급실 상황은 수시로 변하므로 방문 전 전화 확인을 권장합니다"
EMERGENCY = "119에 즉시 신고하세요. 검색 결과를 기다리지 마세요."
Severity = Literal["critical", "urgent", "standard"]
Condition = Literal["심근경색", "뇌출혈", "뇌졸중", "중증외상", "화상", "분만"]
Radius = Literal[3, 5, 10, 20, 30]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TriageAssessment(StrictModel):
    severity: Severity
    condition: Condition | None = None
    injection: bool
    confidence: float = Field(ge=0, le=1)
    confidence_escalated: bool = False

    @model_validator(mode="after")
    def conservative(self) -> Self:
        if self.confidence < 0.6 and not self.confidence_escalated:
            escalated: dict[Severity, Severity] = {
                "standard": "urgent",
                "urgent": "critical",
                "critical": "critical",
            }
            self.severity = escalated[self.severity]
            self.confidence_escalated = True
        if self.severity == "standard" and self.condition is not None:
            raise ValueError("standard cannot have a severe condition")
        return self


class HospitalCandidate(StrictModel):
    hpid: str = Field(min_length=1)
    name: str
    distance_km: float = Field(ge=0, allow_inf_nan=False)
    er_beds_available: int | None
    beds_updated_at: str | None
    accepts_condition: Literal["yes", "no", "unknown"]
    er_tel: str | None
    address: str
    is_cached: bool
    is_stale: bool

    @property
    def stale(self) -> bool:
        return self.is_cached or self.is_stale

    @field_validator("distance_km")
    @classmethod
    def one_decimal(cls, value: float) -> float:
        return round(value, 1)


class ERSearchReply(StrictModel):
    severity: Severity
    call_119_first: bool
    search_radius_km: Radius
    hospitals: list[HospitalCandidate] = Field(max_length=3)
    no_candidate_reason: str | None = None
    data_timestamp: str
    next_action: str = Field(min_length=1, max_length=80)
    disclaimer: Literal["응급실 상황은 수시로 변하므로 방문 전 전화 확인을 권장합니다"]

    @field_validator("data_timestamp")
    @classmethod
    def iso_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def invariants(self) -> Self:
        if self.call_119_first != (self.severity == "critical"):
            raise ValueError("call_119_first must equal severity == critical")
        if not self.hospitals and not self.no_candidate_reason:
            raise ValueError("empty candidates require a reason")
        if len({h.hpid for h in self.hospitals}) != len(self.hospitals):
            raise ValueError("duplicate hospitals")
        distances = [h.distance_km for h in self.hospitals]
        if distances != sorted(distances):
            raise ValueError("hospitals must be sorted by distance")
        if any(h.distance_km > self.search_radius_km for h in self.hospitals):
            raise ValueError("hospital outside search radius")
        return self
