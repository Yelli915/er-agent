"""Run the real LangGraph agent using the shared memory and search implementations."""

import re
from dataclasses import dataclass
from time import perf_counter

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import merge_configs
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from er_finder.agent.adapters import ERGraphState, VisitSelection, selected_index
from er_finder.agent.prompts import SYSTEM_PROMPT
from er_finder.agent.tools import TOOLS
from er_finder.agent.usage import ClassifierUsage, RunUsage
from er_finder.cli.renderer import render_reply
from er_finder.guardrails.middleware import (
    EmergencyInputGuard,
    EvidenceCheckMiddleware,
    ModelCounter,
    ProfileDynamicPrompt,
    ServerPlannedCalls,
    ToolSafetyMiddleware,
)
from er_finder.guardrails.pii import PII_PATTERN
from er_finder.guardrails.triage import InputClassifier
from er_finder.memory.context import ERFinderContext
from er_finder.memory.session import SessionManager
from er_finder.memory.store import ERFinderStore
from er_finder.models import EMERGENCY, ERSearchReply
from er_finder.safety import assess_input, mask_pii, safe_data
from er_finder.search.service import SearchSession


@dataclass
class RunDiagnostics:
    """Numeric metrics. Token fields cover the agent graph; classifier_* the classifier."""

    model_calls: int
    session_model_calls: int
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    tokens_complete: bool
    succeeded: bool
    token_scope: str = "agent_graph"
    classifier_calls: int = 0
    classifier_elapsed_seconds: float = 0.0
    classifier_input_tokens: int | None = None
    classifier_output_tokens: int | None = None
    classifier_tokens_complete: bool = True
    error_type: str | None = None
    status_code: int | None = None
    error_code: str | None = None
    error_param: str | None = None


def _error_metadata(error: Exception) -> dict:
    """Allow only exception class and fixed API diagnostic identifiers."""
    allowed_codes = {
        "invalid_api_key",
        "insufficient_quota",
        "rate_limit_exceeded",
        "model_not_found",
        "invalid_request_error",
        "invalid_function_parameters",
        "invalid_json_schema",
        "context_length_exceeded",
        "unsupported_parameter",
        "unsupported_value",
        "permission_denied",
        "billing_hard_limit_reached",
    }
    status = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    param = getattr(error, "param", None)
    param_pattern = (
        r"(?:tools(?:\[\d+\])?(?:\.function(?:\.(?:name|description|parameters))?)?"
        r"|response_format(?:\.json_schema(?:\.(?:name|schema|strict))?)?"
        r"|model|max_tokens|max_completion_tokens|temperature|messages)"
    )
    return {
        "error_type": type(error).__name__,
        "status_code": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code if isinstance(code, str) and code in allowed_codes else None,
        "error_param": (
            param if isinstance(param, str) and re.fullmatch(param_pattern, param) else None
        ),
    }


@dataclass
class ChatResult:
    reply: ERSearchReply
    text: str
    pending_approval: dict | None = None
    note: str | None = None
    diagnostics: RunDiagnostics | None = None


class ERFinder:
    def __init__(
        self,
        provider,
        model=None,
        classifier=None,
        user_id="local-user",
        transport="car",
        store=None,
        checkpointer=None,
        on_emergency=None,
        server_planned_calls=True,
    ):
        self.context = ERFinderContext(user_id, transport)
        self.session = SearchSession(provider, transport)
        self.store = store if store is not None else InMemoryStore()
        self.profiles = ERFinderStore(self.store, user_id)
        self.visit = VisitSelection(self.profiles, self.session)
        self.checkpointer = (
            checkpointer
            if checkpointer is not None
            else InMemorySaver(
                serde=JsonPlusSerializer(
                    # Pydantic dumps nested candidates as data; only the reply is revived.
                    allowed_msgpack_modules=[("er_finder.models", "ERSearchReply")],
                )
            )
        )
        self.session_manager = SessionManager(self.checkpointer)
        self.classifier = classifier if classifier is not None else InputClassifier()
        self.on_emergency = on_emergency
        self.counter = ModelCounter()
        self.graph = create_agent(
            model,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(ERSearchReply, handle_errors=True),
            state_schema=ERGraphState,
            context_schema=ERFinderContext,
            checkpointer=self.checkpointer,
            store=self.store,
            middleware=[
                EmergencyInputGuard(self.session),
                ProfileDynamicPrompt(self.session, self.profiles, self.visit),
                PIIMiddleware("korean_personal_data", strategy="redact", detector=PII_PATTERN),
                ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
                # Outside the counter so skipped turns are not reported as model calls.
                *([ServerPlannedCalls(self.session, self.visit)] if server_planned_calls else []),
                self.counter,
                ToolRetryMiddleware(max_retries=2, initial_delay=1.0),
                HumanInTheLoopMiddleware(
                    interrupt_on={"save_visit_plan": {"allowed_decisions": ["approve", "reject"]}},
                    description_prefix="이 방문 계획을 저장할까요?",
                ),
                # after_model hooks run in reverse list order: validate before HITL.
                ToolSafetyMiddleware(self.visit, self.session),
                EvidenceCheckMiddleware(self.session),
            ],
        )
        self.run_config = self.session_manager.build_config(user_id)
        self.run_config["configurable"].update(session=self.session, visit=self.visit)
        self.last_reply = None
        self.pending = False
        self.mode = "search"
        self.last_diagnostics = None
        self.last_error_type = None
        self._session_model_calls = 0
        self._classifier_usage = ClassifierUsage()

    def chat(self, text: str, *, force_refresh: bool = False) -> ChatResult:
        text = mask_pii(text).strip()
        if not text:
            raise ValueError("위치와 증상을 입력하세요.")
        rules = assess_input(text)
        alerted = rules.triage.severity == "critical"
        if alerted and self.on_emergency:
            self.on_emergency(EMERGENCY)
        selected = selected_index(text)
        if selected is not None and self.last_reply and not rules.blocked:
            assessment = rules
            self._classifier_usage = ClassifierUsage()  # no classifier call this turn
        else:
            assessment = self.classifier.assess(text)
            self._classifier_usage = (
                getattr(self.classifier, "last_usage", None) or ClassifierUsage()
            )
        if not alerted and assessment.triage.severity == "critical" and self.on_emergency:
            self.on_emergency(EMERGENCY)
        self.counter.calls = 0
        if self.pending:
            # A new turn cancels the paused action rather than implicitly resuming it.
            self.session_manager.reset_session(self.context.user_id)
            self.pending = False
            self.visit.reset()
        history_note = None
        if "지난번" in text or "최근 방문" in text:
            visits = self.profiles.get_recent_visits(limit=1)
            history_note = (
                f"최근 저장한 방문 계획: {safe_data(visits[0]['name'])}. "
                "현재 수용 여부는 새로 확인해야 합니다."
                if visits
                else "저장된 방문 기록이 없습니다."
            )
        if selected is not None and self.last_reply and not assessment.blocked:
            if selected >= len(self.last_reply.hospitals):
                raise ValueError("안내된 후보 번호를 선택하세요.")
            self.mode = "visit"
            self.visit.prepare(self.last_reply.hospitals[selected], self.session.symptom_text)
        else:
            self.mode = "search"
            self.visit.reset()
            self.session.begin(
                text,
                self.profiles.get_home_address(),
                force_refresh=force_refresh,
                assessment=assessment,
            )
        return self._run_graph({"messages": [HumanMessage(content=text)]}, history_note)

    def approve(self, approved: bool) -> ChatResult:
        if not self.pending or not self.graph.get_state(self.run_config).interrupts:
            raise ValueError("승인 대기 중인 방문 계획이 없습니다.")
        self.visit.authorize(approved)
        self.pending = False
        self._classifier_usage = ClassifierUsage()
        self.counter.calls = 0
        decision = (
            {"type": "approve"}
            if approved
            else {
                "type": "reject",
                "message": "사용자가 저장을 거절했습니다.",
            }
        )
        return self._run_graph(Command(resume={"decisions": [decision]}))

    def _run_graph(self, inputs, history_note=None) -> ChatResult:
        started = perf_counter()
        usage = RunUsage()
        succeeded = False
        error_metadata = {}
        self.last_error_type = None
        try:
            output = self.graph.invoke(
                inputs,
                config=merge_configs(self.run_config, {"callbacks": [usage]}),
                context=self.context,
            )
            self.pending = bool(output.get("__interrupt__"))
            reply = self.session.check_evidence(output.get("structured_response"))
            note = self.session.assessment.reason or history_note
            if self.visit.decided and not self.pending:
                if self.visit.saved:
                    note = "방문 계획을 저장했습니다. 병원 예약이나 연락은 실행하지 않았습니다."
                elif not self.visit.approved:
                    note = "저장을 거절했습니다. 방문 계획을 저장하지 않았습니다."
                else:
                    note = "방문 계획을 저장하지 못했습니다. 다시 선택해 주세요."
            succeeded = not bool(self.session.lookup_error)
        except Exception as error:
            error_metadata = _error_metadata(error)
            self.last_error_type = error_metadata["error_type"]
            saved = self.visit.saved
            self.pending = False
            self.visit.reset()
            self.session_manager.reset_session(self.context.user_id)
            self.session.fail_lookup()
            reply = self.session.check_evidence(None)
            note = (
                "방문 계획은 저장되었으나 안내 응답을 완료하지 못했습니다. 저장을 반복하지 마세요."
                if saved
                else "요청 처리에 실패해 병원 후보를 표시하지 않습니다. 다시 조회해 주세요."
            )
        self._session_model_calls += self.counter.calls
        self.last_diagnostics = RunDiagnostics(
            model_calls=self.counter.calls,
            session_model_calls=self._session_model_calls,
            elapsed_seconds=max(0.0, perf_counter() - started),
            **{
                key: value if usage.known_responses or not self.counter.calls else None
                for key, value in usage.tokens.items()
            },
            tokens_complete=usage.known_responses == usage.responses == self.counter.calls,
            succeeded=succeeded,
            classifier_calls=self._classifier_usage.calls,
            classifier_elapsed_seconds=self._classifier_usage.elapsed_seconds,
            classifier_input_tokens=self._classifier_usage.input_tokens,
            classifier_output_tokens=self._classifier_usage.output_tokens,
            classifier_tokens_complete=self._classifier_usage.tokens_complete,
            **error_metadata,
        )
        self.last_reply = reply
        pending = self.visit.pending if self.pending else None
        return ChatResult(
            reply=reply,
            text=render_reply(reply, note=note, pending=pending),
            pending_approval=pending,
            note=note,
            diagnostics=self.last_diagnostics,
        )

    def end_session(self):
        self.session_manager.end_session(self.context.user_id)
        self.session.clear()
        self.visit.reset()
        self.pending = False
        self.mode = "search"
        self.last_reply = None
        self.last_diagnostics = None
        self.last_error_type = None
        self._session_model_calls = 0

    def close(self):
        try:
            self.end_session()
        finally:
            self.session.provider.close()
