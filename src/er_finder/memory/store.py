"""Long-term memory (Store) — 세션이 끝나도 남아 다음 대화까지 이어지는 값.

home_address, recent_visits 두 항목을 담당하며, 실제 저장소는 LangGraph의
InMemoryStore를 그대로 사용한다.

Store에는 "동의한 기본 주소"와 "방문 기록"만 저장한다. current_location 같은
세션 중 위치 정보는 State(state.py)에만 있어야 하고 여기 들어오면 안 된다.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import TypedDict

from langgraph.store.base import BaseStore

_NAMESPACE_PROFILE = "profile"
_NAMESPACE_VISITS = "visits"
_HOME_ADDRESS_KEY = "home_address"


class VisitRecord(TypedDict):
    visit_id: str
    hpid: str
    name: str
    symptom_summary: str
    saved_at: str  # ISO 8601


class ConsentRequiredError(Exception):
    """동의 없이 home_address를 저장하려고 할 때 발생한다.

    @dataclass로 만들면 e.args가 비어버리므로 일반 클래스로 둔다.
    """

    def __init__(
        self, message: str = "사용자 동의 없이는 home_address를 저장할 수 없습니다."
    ) -> None:
        self.message = message
        super().__init__(message)


class ERFinderStore:
    """user_id별 home_address / recent_visits를 관리하는 얇은 래퍼.

    InMemoryStore를 여기저기서 직접 호출하지 않고 이 클래스를 거치게 해서
    namespace 규칙과 "동의 없으면 저장 금지" 규칙을 한 곳에서 관리한다.
    """

    def __init__(self, store: BaseStore, user_id: str) -> None:
        self._store = store
        self._user_id = user_id

    # ---------- home_address ----------

    def get_home_address(self) -> str | None:
        item = self._store.get(self._profile_namespace(), _HOME_ADDRESS_KEY)
        if item is None:
            return None
        return item.value.get("address")

    def set_home_address(self, address: str, *, consent: bool) -> None:
        """기본 주소를 저장한다. consent=False면 ConsentRequiredError를 던진다."""
        if not consent:
            raise ConsentRequiredError()
        if not address or not address.strip():
            raise ValueError("address는 빈 값일 수 없습니다.")
        self._store.put(
            self._profile_namespace(),
            _HOME_ADDRESS_KEY,
            {"address": address, "saved_at": _now_iso()},
        )

    def clear_home_address(self) -> None:
        """사용자가 동의를 철회했을 때 기본 주소를 삭제한다."""
        self._store.delete(self._profile_namespace(), _HOME_ADDRESS_KEY)

    # ---------- recent_visits ----------

    def add_visit(self, *, hpid: str, name: str, symptom_summary: str) -> VisitRecord:
        """방문 계획을 저장한다. HITL 승인이 끝난 뒤에만 호출돼야 한다."""
        if not hpid or not hpid.strip():
            raise ValueError("hpid는 빈 값일 수 없습니다.")
        if not name or not name.strip():
            raise ValueError("name은 빈 값일 수 없습니다.")
        visit_id = str(uuid.uuid4())
        record: VisitRecord = {
            "visit_id": visit_id,
            "hpid": hpid,
            "name": name,
            "symptom_summary": symptom_summary,
            "saved_at": _now_iso(),
        }
        self._store.put(self._visits_namespace(), visit_id, dict(record))
        return record

    def get_recent_visits(self, limit: int = 5) -> list[VisitRecord]:
        """최근 방문 기록을 최신순으로 최대 limit개 반환한다.

        InMemoryStore.search()는 저장 순서를 보장하지 않으므로, 전체를
        가져와 saved_at 기준으로 정렬한 뒤 limit을 적용한다.
        """
        if limit < 0:
            raise ValueError("limit은 0 이상이어야 합니다.")
        if limit == 0:
            return []
        visits = []
        page_size = 100
        offset = 0
        while True:
            items = self._store.search(self._visits_namespace(), limit=page_size, offset=offset)
            visits.extend(item.value for item in items)
            if len(items) < page_size:
                break
            offset += len(items)
        visits.sort(key=lambda v: v.get("saved_at", ""), reverse=True)
        return visits[:limit]

    # ---------- namespace 규칙 ----------

    def _profile_namespace(self) -> tuple[str, ...]:
        return ("er_finder", self._user_id, _NAMESPACE_PROFILE)

    def _visits_namespace(self) -> tuple[str, ...]:
        return ("er_finder", self._user_id, _NAMESPACE_VISITS)


_clock_lock = threading.Lock()
_last_stamp = datetime.min.replace(tzinfo=UTC)


def _now_iso() -> str:
    """Strictly increasing timestamps, so records saved within one clock tick keep their order."""
    global _last_stamp
    with _clock_lock:
        now = datetime.now(UTC)
        if now <= _last_stamp:
            now = _last_stamp + timedelta(microseconds=1)
        _last_stamp = now
    return now.isoformat()
