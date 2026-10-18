"""Runtime Context — 대화 호출 시 한 번 정해지고 끝까지 바뀌지 않는 값.

user_id, transport 두 값을 담는다. create_agent의 context_schema로
등록되어 미들웨어·도구가 runtime.context로 읽기 전용 접근한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

TransportMode = Literal["walk", "car", "transit"]
_VALID_TRANSPORT_MODES = get_args(TransportMode)


@dataclass(frozen=True)
class ERFinderContext:
    """앱이 에이전트를 호출할 때 1회 전달하는 고정값. frozen=True로 불변을 강제한다."""

    user_id: str
    """사용자 식별자. Store 키 및 세션 체크포인터(thread_id)로 쓴다."""

    transport: TransportMode = "car"
    """이동수단. 기본값 car. 반경 확대 시작값·순서를 결정하는 데 쓴다.
    - walk: 3 → 6 → 9 → 12km
    - car / transit: 5 → 10 → 20 → 30km
    """

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id는 빈 값일 수 없습니다 (Store/세션 키로 사용됨)")
        if self.transport not in _VALID_TRANSPORT_MODES:
            raise ValueError(
                f"transport는 {_VALID_TRANSPORT_MODES} 중 하나여야 합니다: {self.transport!r}"
            )
