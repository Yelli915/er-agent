"""Server-owned tool scheduling and model-call accounting for the agent graph."""

from typing import Any
from uuid import uuid4

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse, hook_config
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

from er_finder.errors import OperationError
from er_finder.search.contracts import PlannedCall
from er_finder.search.service import SearchSession


class ExecutionState(AgentState[Any], total=False):
    planned_calls: list[PlannedCall]


def _as_tool_calls(calls: list[PlannedCall]) -> list[dict[str, Any]]:
    return [{**c, "id": str(uuid4()), "type": "tool_call"} for c in calls]


def _remove_tool_results(messages: list[Any], call_ids: set[str | None]) -> list[RemoveMessage]:
    return [
        RemoveMessage(id=message.id)
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id in call_ids and message.id
    ]


class ModelCounter(AgentMiddleware):
    def __init__(self) -> None:
        self.calls = 0

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        self.calls += 1
        try:
            return handler(request)
        except OperationError:
            raise
        except Exception as error:
            raise OperationError("model", "model") from error


class ServerPlannedCalls(AgentMiddleware):
    """Plan each model step once and enforce it before human approval.

    The graph owns the plan: prompt generation, model bypass and tool-call replacement
    all read the same step snapshot. Disabling bypass retains server-owned ordering.
    """

    state_schema = ExecutionState

    def __init__(
        self, session: SearchSession, visit: Any, *, enabled: bool = True,
        deterministic_reply: bool = True,
    ) -> None:
        self.session, self.visit = session, visit
        self.enabled = enabled
        self.deterministic_reply = deterministic_reply

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        calls = [] if self.session.failure or self.visit.pending else self.session.next_calls()
        if self.enabled and self.deterministic_reply and not calls and (
            not self.visit.pending or self.visit.decided
        ):
            return {
                "planned_calls": [], "jump_to": "end",
                "structured_response": self.session.make_reply(),
            }
        return {"planned_calls": calls}

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        calls = request.state["planned_calls"]
        if (
            self.enabled and self.deterministic_reply
            and self.visit.pending and not self.visit.decided
        ):
            calls = [{"name": "save_visit_plan", "args": dict(self.visit.pending)}]
        if not self.enabled or not calls:
            return handler(request)
        return ModelResponse(result=[AIMessage(content="", tool_calls=_as_tool_calls(calls))])

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.session.failure:
            return None
        last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if last is None:
            return None
        if self.visit.pending and not self.visit.decided:
            self.visit.validate(self.visit.pending)
            next_calls: list[PlannedCall] = [
                {"name": "save_visit_plan", "args": dict(self.visit.pending)}
            ]
        else:
            next_calls = [] if self.visit.pending else state["planned_calls"]
        if next_calls:
            # Server-owned calls enforce both order and arguments, including one selected
            # save request. Drop acknowledgements of replaced structured-output calls.
            old_ids = {c["id"] for c in last.tool_calls}
            removals = _remove_tool_results(state["messages"], old_ids)
            proposed = [{"name": c["name"], "args": c["args"]} for c in last.tool_calls]
            calls = last.tool_calls if proposed == next_calls else _as_tool_calls(next_calls)
            tool_names = {c["name"] for c in calls}
            self.session.allow_severe_batch = {
                "get_er_bed_status", "get_severe_acceptance"
            } <= tool_names
            if self.session.allow_severe_batch:
                self.session.beds_ready.clear()
            return {
                "messages": [
                    *removals,
                    last.model_copy(update={"tool_calls": calls, "content": ""}),
                ],
                "structured_response": None,
            }
        rejected_ids = {c["id"] for c in last.tool_calls if c["name"] == "save_visit_plan"}
        if rejected_ids:
            # HITL examines AI tool calls even if a ToolMessage already answers them.
            # Remove rejected calls and any matching results before HITL can interrupt.
            removals = _remove_tool_results(state["messages"], rejected_ids)
            accepted = [c for c in last.tool_calls if c["id"] not in rejected_ids]
            return {
                "messages": [
                    *removals,
                    last.model_copy(
                        update={
                            "tool_calls": accepted,
                            "content": "사용자 선택이 없어 저장 요청을 차단했습니다.",
                        }
                    ),
                ]
            }
        return None
