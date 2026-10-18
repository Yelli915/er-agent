import heapq
from datetime import UTC, datetime
from typing import Any

from er_finder.guardrails.evidence import safe_data
from er_finder.models import HospitalCandidate


def is_stale(timestamp: str | None, now: datetime) -> bool:
    if not timestamp:
        return True
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            return True
        age = (now.astimezone(UTC) - parsed.astimezone(UTC)).total_seconds()
        return age > 900 or age < -60
    except (TypeError, ValueError):
        return True


def select_candidates(
    facilities: dict[str, dict[str, Any]],
    beds: dict[str, dict[str, Any]],
    acceptance: dict[str, dict[str, Any]],
    details: dict[str, dict[str, Any]],
    condition: str | None,
    now: datetime,
) -> list[HospitalCandidate]:
    # Rank on the full-precision distance first so only the winners are built (O(n log 3)).
    eligible: list[tuple[float, str, int]] = []
    for hpid, facility in facilities.items():
        bed = beds.get(hpid, {})
        count = bed.get("er_beds_available")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            continue
        acceptable = acceptance.get(hpid, {}).get("acceptable")
        if condition and acceptable is not True:
            continue
        eligible.append((facility["distance_km"], hpid, count))

    selected: list[HospitalCandidate] = []
    for _, hpid, count in heapq.nsmallest(3, eligible):
        facility = facilities[hpid]
        bed = beds.get(hpid, {})
        detail = details.get(hpid, {})
        tel = detail.get("er_tel") or bed.get("er_tel") or facility.get("er_tel")
        safe_tel = safe_data(tel) if tel else None
        if safe_tel == "확인 불가":
            safe_tel = None
        selected.append(
            HospitalCandidate(
                hpid=hpid,
                name=safe_data(facility.get("name")),
                distance_km=facility["distance_km"],
                er_beds_available=count,
                beds_updated_at=bed.get("beds_updated_at"),
                accepts_condition="yes" if condition else "unknown",
                er_tel=safe_tel,
                address=safe_data(detail.get("address") or facility.get("address")),
                is_cached=bool(bed.get("is_cached", False)),
                is_stale=is_stale(bed.get("beds_updated_at"), now)
                or bool(bed.get("is_stale", False)),
            )
        )
    return selected
