"""Turn LangGraph stream updates into short user-facing progress lines."""

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[str], None]

_TOOL_DONE = {
    "geocode": "위치를 확인했습니다",
    "list_nearby_ers": "주변 응급실 목록을 확인했습니다",
    "get_er_bed_status": "병상 현황을 확인했습니다",
    "get_severe_acceptance": "중증 수용 여부를 확인했습니다",
    "get_er_detail": "병원 상세 정보를 확인했습니다",
}
COMPOSING = "답변을 정리하고 있습니다"


def progress_lines(chunk: Any) -> list[str]:
    """Labels for one `stream_mode="updates"` chunk; unknown shapes yield nothing."""
    if not isinstance(chunk, dict):
        return []
    update = chunk.get("tools")
    messages = update.get("messages") if isinstance(update, dict) else None
    if not isinstance(messages, list):
        return []
    done = [_TOOL_DONE[n] for m in messages if (n := getattr(m, "name", None)) in _TOOL_DONE]
    # Detail lookups are the last search step; the reply is composed right after.
    if any(getattr(m, "name", None) == "get_er_detail" for m in messages):
        done = [_TOOL_DONE["get_er_detail"], COMPOSING]
    return list(dict.fromkeys(done))
