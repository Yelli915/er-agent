"""Reply composition and evidence checking from verified search state."""

from er_finder.guardrails.evidence import sanitize_prose
from er_finder.models import DISCLAIMER, ERSearchReply


class ReplyBuilder:
    """Mixin for SearchSession; builds the source-derived reply."""

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
