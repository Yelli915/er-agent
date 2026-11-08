"""Reply composition and evidence checking from verified search state."""

from typing import TYPE_CHECKING, Any

from er_finder.freshness import is_stale
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
    FAILURE_MESSAGES,
    REASON_BLOCKED,
    REASON_GREETING,
    REASON_NO_LOCATION,
    REASON_SEVERE_UNCONFIRMED_SUFFIX,
    REASON_TRAUMA_UNCONFIRMED,
    no_candidate_in_radius,
)
from er_finder.models import DISCLAIMER, ERSearchReply

if TYPE_CHECKING:
    from er_finder.search.service import SearchSession


class ReplyBuilder:
    """Builds a detached reply without changing the verified search state."""

    def __init__(self, session: "SearchSession") -> None:
        self.session = session

    def make_reply(self) -> ERSearchReply:
        s = self.session
        candidates = s.candidates
        reason: str | None = None
        action = ACTION_CALL_HOSPITAL
        if s.assessment.blocked:
            candidates = []
            reason = REASON_BLOCKED
            action = ACTION_BLOCKED
        elif s.assessment.greeting:
            candidates = []
            reason = REASON_GREETING
            action = ACTION_GREETING
        elif s.failure:
            candidates = []
            reason = FAILURE_MESSAGES[s.failure["failure_category"]]
            action = ACTION_LOOKUP_FAILED
        elif not s.location:
            reason = REASON_NO_LOCATION
            action = ACTION_NO_LOCATION
        elif not candidates:
            unconfirmed = any(
                b.get("er_beds_available", 0)
                and s.acceptance.get(hpid, {}).get("acceptable") is None
                for hpid, b in s.beds.items()
            )
            if s.triage.condition == "중증외상" and unconfirmed:
                reason = REASON_TRAUMA_UNCONFIRMED
                action = ACTION_TRAUMA_CALL_119
            else:
                reason = no_candidate_in_radius(s.radius)
                if s.triage.condition and unconfirmed:
                    reason += REASON_SEVERE_UNCONFIRMED_SUFFIX
                action = ACTION_NO_CANDIDATE_CALL_119
        if s.specialty_requested and candidates:
            action = ACTION_SPECIALTY
        if s.triage.severity == "critical":
            action = ACTION_CRITICAL
        now = s.now()
        return ERSearchReply(
            severity=s.triage.severity,
            call_119_first=s.triage.severity == "critical",
            search_radius_km=s.radius,
            hospitals=[
                candidate.model_copy(
                    deep=True,
                    update={
                        "is_stale": is_stale(candidate.beds_updated_at, now)
                        or bool(s.beds.get(candidate.hpid, {}).get("is_stale", False))
                    },
                )
                for candidate in candidates
            ],
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
