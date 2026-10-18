"""Assemble the LangGraph agent. Middleware order is behavior: keep it in one place."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy

from er_finder.agent.adapters import ERGraphState
from er_finder.agent.tools import TOOLS
from er_finder.guardrails.middleware import (
    EmergencyInputGuard,
    EvidenceCheckMiddleware,
    ProfileDynamicPrompt,
    ServerPlannedCalls,
    ToolSafetyMiddleware,
)
from er_finder.guardrails.pii import PII_PATTERN
from er_finder.memory.context import ERFinderContext
from er_finder.models import ERSearchReply
from er_finder.prompts import SYSTEM_PROMPT

MODEL_CALL_LIMIT = 10
TOOL_MAX_RETRIES = 2
TOOL_RETRY_DELAY = 1.0


def build_graph(
    model: Any,
    *,
    session: Any,
    profiles: Any,
    visit: Any,
    counter: Any,
    store: Any,
    checkpointer: Any,
    server_planned_calls: bool = True,
) -> Any:
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ERSearchReply, handle_errors=True),
        state_schema=ERGraphState,
        context_schema=ERFinderContext,
        checkpointer=checkpointer,
        store=store,
        middleware=[
            EmergencyInputGuard(session),
            ProfileDynamicPrompt(session, profiles, visit),
            PIIMiddleware("korean_personal_data", strategy="redact", detector=PII_PATTERN),
            ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT, exit_behavior="end"),
            # Outside the counter so skipped turns are not reported as model calls.
            *([ServerPlannedCalls(session, visit)] if server_planned_calls else []),
            counter,
            ToolRetryMiddleware(max_retries=TOOL_MAX_RETRIES, initial_delay=TOOL_RETRY_DELAY),
            HumanInTheLoopMiddleware(
                interrupt_on={"save_visit_plan": {"allowed_decisions": ["approve", "reject"]}},
                description_prefix="이 방문 계획을 저장할까요?",
            ),
            # after_model hooks run in reverse list order: validate before HITL.
            ToolSafetyMiddleware(visit, session),
            EvidenceCheckMiddleware(session),
        ],
    )
