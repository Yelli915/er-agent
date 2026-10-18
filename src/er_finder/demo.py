"""Keyless demo mode (ER_DEMO=1): the real agent graph, guardrails and memory run
unchanged; only the model and the medical data source are replaced by canned ones.

Hospitals here are invented samples near Gangnam station and are labelled as such."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from er_finder.search.coordinates import parse_coordinates

DEMO_NOTICE = "데모 모드: 병원·병상 정보는 가상의 샘플이며 모델 응답은 미리 정해진 흐름입니다."

_SIDO, _SIGUNGU = "서울특별시", "강남구"
_CENTER = (37.4979, 127.0276)
# Identifier, name, latitude offset, available beds, myocardial-infarction acceptance, phone.
_SAMPLES = (
    ("DEMO-1", "샘플 강남응급의료센터", 0.004, 3, True, "02-0000-0001"),
    ("DEMO-2", "샘플 역삼대학병원", 0.008, 2, True, "02-0000-0002"),
    ("DEMO-3", "샘플 서초종합병원", 0.011, 0, True, "02-0000-0003"),
    ("DEMO-4", "샘플 논현의료원", 0.015, 4, False, "02-0000-0004"),
    ("DEMO-5", "샘플 삼성중앙병원", 0.022, 1, True, "02-0000-0005"),
)


def is_demo() -> bool:
    return os.environ.get("ER_DEMO", "").strip().lower() in {"1", "true", "yes"}


class DemoProvider:
    """Same port as LiveProvider, served from the sample table above."""

    def geocode(self, query: str) -> dict[str, Any]:
        coordinates = parse_coordinates(query)
        if coordinates is None and any(word in query for word in ("강남", "역삼")):
            coordinates = _CENTER
        if coordinates is None:
            return {"found": False}
        return dict(
            found=True, lat=coordinates[0], lon=coordinates[1], sido=_SIDO, sigungu=_SIGUNGU
        )

    def list_nearby_ers(self, lat: float, lon: float, radius_km: int = 5) -> list[dict[str, Any]]:
        return [
            dict(
                hpid=hpid,
                name=name,
                lat=_CENTER[0] + offset,
                lon=_CENTER[1],
                sido=_SIDO,
                sigungu=_SIGUNGU,
                address=f"{_SIDO} {_SIGUNGU} 샘플로 {number}",
                er_tel=tel,
            )
            for number, (hpid, name, offset, _, _, tel) in enumerate(_SAMPLES, start=1)
        ]

    def get_er_bed_status(
        self, sido: str, sigungu: str, hpids: list[str], *, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        return [
            dict(hpid=hpid, er_beds_available=beds, beds_updated_at=now, is_cached=False)
            for hpid, _, _, beds, _, _ in _SAMPLES
            if hpid in hpids
        ]

    def get_severe_acceptance(
        self, sido: str, sigungu: str, condition: str
    ) -> list[dict[str, Any]]:
        return [
            dict(hpid=hpid, acceptable=accepted if condition == "심근경색" else None)
            for hpid, _, _, _, accepted, _ in _SAMPLES
        ]

    def get_er_detail(self, hpid: str) -> dict[str, Any]:
        tel = next((item[5] for item in _SAMPLES if item[0] == hpid), None)
        return {"er_tel": tel}

    def clear_cache(self) -> None:
        return None

    def close(self) -> None:
        return None


class ScriptedModel(BaseChatModel):
    """Follows the server-planned `next_calls`, then returns the drafted reply verbatim."""

    @property
    def _llm_type(self) -> str:
        return "scripted-demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        system = next(message.content for message in messages if message.type == "system")
        data = json.loads(system[system.index("{") :])
        if data.get("pending_visit"):
            calls = [{"name": "save_visit_plan", "args": data["pending_visit"]}]
        elif data["next_calls"]:
            calls = data["next_calls"]
        else:
            calls = [{"name": "ERSearchReply", "args": data["draft"]}]
        message = AIMessage(
            content="",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            tool_calls=[{**call, "id": str(uuid4()), "type": "tool_call"} for call in calls],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def demo_runner_factory(**kwargs: Any) -> Any:
    from er_finder.agent.factory import ManagedRunner
    from er_finder.agent.runner import ERFinder

    provider = DemoProvider()
    return ManagedRunner(ERFinder(provider=provider, model=ScriptedModel(), **kwargs), provider)
