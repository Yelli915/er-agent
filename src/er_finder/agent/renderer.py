"""Format the runner reply without importing the domain model or making API calls."""


def render_reply(reply, *, note=None, pending=None) -> str:
    lines = []
    if reply.call_119_first:
        lines.append("119에 즉시 신고하세요. 검색 결과를 기다리지 마세요.")
    if note:
        lines.append(str(note))
    for index, hospital in enumerate(reply.hospitals, 1):
        beds = (
            "확인 불가" if hospital.er_beds_available is None else f"{hospital.er_beds_available}개"
        )
        lines.append(f"{index}. {hospital.name} · {hospital.distance_km:.1f} km · 가용 병상 {beds}")
        lines.append(f"주소: {hospital.address or '확인 불가'}")
        lines.append(f"병원 대표전화: {hospital.er_tel or '확인 불가'}")
    if not reply.hospitals:
        lines.append(reply.no_candidate_reason or "확인된 후보가 없습니다.")
    if pending:
        lines.append("방문 계획 저장을 승인하거나 거절해 주세요.")
    lines.extend([reply.next_action, reply.disclaimer])
    return "\n".join(lines)
