"""Reply composition and evidence checking from verified search state."""

from typing import TYPE_CHECKING, Any

from er_finder.guardrails.evidence import sanitize_prose
from er_finder.messages import (
    ACTION_BLOCKED,
    ACTION_CALL_HOSPITAL,
    ACTION_CRITICAL,
    ACTION_GREETING,
    ACTION_LOOKUP_FAILED,
    ACTION_NO_CANDIDATE_CALL_119,
    ACTION_NO_LOCATION,
    ACTION_SPECIALTY,
    ACTION_TRAUMA_CALL_119,
    REASON_BLOCKED,
    REASON_GREETING,
    REASON_NO_LOCATION,
    REASON_SEVERE_UNCONFIRMED_SUFFIX,
    REASON_TRAUMA_UNCONFIRMED,
    no_candidate_in_radius,
)
from er_finder.models import DISCLAIMER, ERSearchReply

if TYPE_CHECKING:
    from er_finder.search.planner import CallPlanner
    from er_finder.search.service import SearchSession


class ReplyBuilder:
    """Builds the source-derived reply from a SearchSession and its call planner."""

    def __init__(self, session: "SearchSession", planner: "CallPlanner") -> None:
        self.session = session
        self.planner = planner

    def make_reply(self) -> ERSearchReply:
        s = self.session
        self.planner.refresh_candidates()
        reason: str | None = None
        action = ACTION_CALL_HOSPITAL
        if s.assessment.blocked:
            s.candidates = []
            reason = REASON_BLOCKED
            action = ACTION_BLOCKED
        elif s.assessment.greeting:
            s.candidates = []
            reason = REASON_GREETING
            action = ACTION_GREETING
        elif s.lookup_error:
            s.candidates = []
            reason = s.lookup_error
            action = ACTION_LOOKUP_FAILED
        elif not s.location:
            reason = REASON_NO_LOCATION
            action = ACTION_NO_LOCATION
        elif not s.candidates:
            if s.triage.condition == "중증외상" and any(
                b.get("er_beds_available", 0)
                and s.acceptance.get(hpid, {}).get("acceptable") is None
                for hpid, b in s.beds.items()
            ):
                reason = REASON_TRAUMA_UNCONFIRMED
                action = ACTION_TRAUMA_CALL_119
            else:
                reason = no_candidate_in_radius(s.radius)
                if s.triage.condition and any(
                    b.get("er_beds_available", 0)
                    and s.acceptance.get(hpid, {}).get("acceptable") is None
                    for hpid, b in s.beds.items()
                ):
                    reason += REASON_SEVERE_UNCONFIRMED_SUFFIX
                action = ACTION_NO_CANDIDATE_CALL_119
        if s.specialty_requested and s.candidates:
            action = ACTION_SPECIALTY
        if s.triage.severity == "critical":
            action = ACTION_CRITICAL
        return ERSearchReply(
            severity=s.triage.severity,
            call_119_first=s.triage.severity == "critical",
            search_radius_km=s.radius,
            hospitals=s.candidates,
            no_candidate_reason=reason,
            data_timestamp=s.data_timestamp,
            next_action=action,
            disclaimer=DISCLAIMER,
        )

    def check_evidence(self, proposal: Any) -> ERSearchReply:
        s = self.session
        canonical = self.make_reply().model_dump()
        if hasattr(proposal, "model_dump"):
            proposal = proposal.model_dump()
        if not isinstance(proposal, dict):
            return ERSearchReply.model_validate(canonical)
        rows = proposal.get("hospitals", [])
        if not isinstance(rows, list):
            rows = []
            s.evidence_corrections += 1
        proposed: dict[str, dict[str, Any]] = {}
        for hospital in rows:
            if not isinstance(hospital, dict) or not isinstance(hospital.get("hpid"), str):
                s.evidence_corrections += 1
                continue
            proposed[hospital["hpid"]] = hospital
        s.evidence_corrections += sum(hpid not in s.facilities for hpid in proposed)
        for hospital in canonical["hospitals"]:
            candidate = proposed.get(hospital["hpid"])
            if candidate is not None:
                s.evidence_corrections += sum(
                    candidate.get(field) != value for field, value in hospital.items()
                )
        prose = proposal.get("next_action", "")
        if isinstance(prose, str) and sanitize_prose(prose) != prose:
            s.evidence_corrections += 1
        # A model proposal cannot change verified source data, the selected radius,
        # emergency instructions or provenance. Keep the source-derived reply.
        return ERSearchReply.model_validate(canonical)
