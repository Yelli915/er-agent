"""The model and tool hooks share one server-owned plan per step."""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from er_finder.agent.execution import ServerPlannedCalls
from er_finder.search.service import SearchSession
from tests.unit.search.provider_fixture import OfflineSearchProvider


@pytest.mark.parametrize("enabled", [True, False])
def test_model_and_tool_hooks_share_one_plan(enabled, monkeypatch):
    session = SearchSession(OfflineSearchProvider())
    visit = SimpleNamespace(pending=None, decided=False)
    execution = ServerPlannedCalls(session, visit, enabled=enabled)
    calls = [{"name": "geocode", "args": {"query": "test-location"}}]
    plans = []

    def plan():
        plans.append(True)
        assert len(plans) == 1, "a model step must not plan again"
        return calls

    monkeypatch.setattr(session, "next_calls", plan)
    state = execution.before_model({}, None)
    # Prompt generation must use even an empty supplied plan instead of replanning.
    assert session.status(next_calls=state["planned_calls"])["next_calls"] == calls
    model_calls = []

    def model(request):
        model_calls.append(True)
        return ModelResponse(result=[AIMessage(content="premature answer")])

    response = execution.wrap_model_call(SimpleNamespace(state=state), model)
    message = response.result[0]
    updated = execution.after_model({**state, "messages": [message]}, None)
    tool_calls = updated["messages"][-1].tool_calls
    assert [{"name": c["name"], "args": c["args"]} for c in tool_calls] == calls
    assert len(plans) == 1
    assert len(model_calls) == int(not enabled)
    if enabled:
        assert tool_calls == message.tool_calls


def test_next_step_replaces_the_previous_plan_including_an_empty_plan(monkeypatch):
    session = SearchSession(OfflineSearchProvider())
    execution = ServerPlannedCalls(session, SimpleNamespace(pending=None, decided=False))
    calls = [{"name": "geocode", "args": {"query": "previous-location"}}]
    monkeypatch.setattr(session, "next_calls", lambda: calls)
    state = execution.before_model({}, None)
    calls = []
    state.update(execution.before_model(state, None))
    assert state["planned_calls"] == []

    def unexpected_plan():
        raise AssertionError("an empty plan is already a completed planning decision")

    monkeypatch.setattr(session, "next_calls", unexpected_plan)
    assert session.status(next_calls=state["planned_calls"])["next_calls"] == []
    message = AIMessage(content="final answer")
    response = execution.wrap_model_call(
        SimpleNamespace(state=state), lambda request: ModelResponse(result=[message])
    )
    assert response.result == [message]
    assert execution.after_model({**state, "messages": [message]}, None) is None
