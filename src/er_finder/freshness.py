"""Pure timestamp freshness checks shared by API and search results."""

from datetime import UTC, datetime


def is_stale(timestamp: str | None, now: datetime, *, max_age_seconds: float = 900) -> bool:
    if not timestamp:
        return True
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            return True
        age = (now.astimezone(UTC) - parsed.astimezone(UTC)).total_seconds()
        return age > max_age_seconds or age < -60
    except (TypeError, ValueError):
        return True
