from datetime import UTC, datetime, timedelta

from er_finder.freshness import is_stale


def test_freshness_honors_the_configured_window_and_timezone():
    now = datetime(2026, 11, 8, 0, 0, tzinfo=UTC)
    timestamp = (now - timedelta(minutes=3)).isoformat()
    assert not is_stale(timestamp, now)
    assert is_stale(timestamp, now, max_age_seconds=120)
    assert not is_stale("2026-11-08T08:59:00+09:00", now, max_age_seconds=120)
    assert not is_stale((now + timedelta(seconds=60)).isoformat(), now)
    assert is_stale((now + timedelta(seconds=61)).isoformat(), now)
