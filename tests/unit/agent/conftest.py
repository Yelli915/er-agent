"""Real agent/guardrail/memory integration; only the model and API provider are offline."""

import json
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from tests.unit.search.provider_fixture import OfflineSearchProvider


class OfflineModel(BaseChatModel):
    strategy: str = "next_calls"
    _human_inputs: list[str] = PrivateAttr(default_factory=list)
    _requests: list[dict] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-agent-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._human_inputs = [message.content for message in messages if message.type == "human"]
        if self.strategy == "failure":
            error = RuntimeError("private-key-must-not-be-shown")
            error.status_code = 400
            error.code = "invalid_function_parameters"
            error.param = "tools[0].function.parameters"
            raise error
        system = next(message.content for message in messages if message.type == "system")
        data = json.loads(system[system.index("{") :])
        if self.strategy == "fail_after_beds" and data["draft"]["hospitals"]:
            raise RuntimeError("private model error")
        if self.strategy == "fail_after_save" and data["visit_saved"]:
            raise RuntimeError("private model error")
        self._requests.append(data)
        if data.get("pending_visit"):
            calls = [{"name": "save_visit_plan", "args": data["pending_visit"]}]
        elif data["next_calls"] and self.strategy != "premature":
            calls = data["next_calls"]
        else:
            calls = [{"name": "ERSearchReply", "args": data["draft"]}]
        message = AIMessage(
            content="",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            tool_calls=[{**call, "id": str(uuid4()), "type": "tool_call"} for call in calls],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class OfflineProvider(OfflineSearchProvider):
    def __init__(self):
        self.closed = False
        self.queries = []

    def geocode(self, query):
        self.queries.append(query)
        return super().geocode(query)

    def get_er_bed_status(self, sido, sigungu, hpids, *, force_refresh=False):
        rows = super().get_er_bed_status(sido, sigungu, hpids, force_refresh=force_refresh)
        return [
            {**row, "beds_updated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
            for row in rows
        ]

    def close(self):
        self.closed = True


@pytest.fixture(params=[True, False], ids=["deterministic", "final-model"])
def make_finder(request):
    from er_finder.agent.runner import ERFinder

    created = []

    def make(**kwargs):
        finder = ERFinder(
            provider=kwargs.pop("provider", OfflineProvider()),
            model=kwargs.pop("model", OfflineModel()),
            deterministic_reply=kwargs.pop("deterministic_reply", request.param),
            **kwargs,
        )
        created.append(finder)
        return finder

    yield make
    for finder in created:
        finder.close()
