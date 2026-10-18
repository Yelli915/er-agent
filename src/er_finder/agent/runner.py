"""Run the real LangGraph agent using the shared memory and search implementations."""

from dataclasses import dataclass
from time import perf_counter

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import merge_configs
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from er_finder.agent.adapters import VisitSelection, selected_index
from er_finder.agent.diagnostics import RunDiagnostics, error_metadata
from er_finder.agent.graph import build_graph
from er_finder.agent.renderer import render_reply
from er_finder.guardrails.evidence import safe_data
from er_finder.guardrails.input_guard import assess_input
from er_finder.guardrails.middleware import ModelCounter
from er_finder.guardrails.pii import mask_pii
from er_finder.guardrails.triage import InputClassifier
from er_finder.memory.context import ERFinderContext
from er_finder.memory.session import SessionManager
from er_finder.memory.store import ERFinderStore
from er_finder.messages import (
    EMPTY_INPUT,
    NO_PENDING_APPROVAL,
    NO_VISIT_HISTORY,
    PICK_LISTED_CANDIDATE,
    REQUEST_FAILED,
    USER_REJECTED_SAVE,
    VISIT_REJECTED,
    VISIT_SAVE_FAILED,
    VISIT_SAVED,
    VISIT_SAVED_REPLY_FAILED,
)
from er_finder.models import EMERGENCY, ERSearchReply
from er_finder.search.service import SearchSession
from er_finder.usage import ClassifierUsage, RunUsage

__all__ = ["ChatResult", "ERFinder", "RunDiagnostics"]

RECENT_VISIT_MARKERS = ("지난번", "최근 방문")


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
        self.graph = build_graph(
            model,
            session=self.session,
            profiles=self.profiles,
            visit=self.visit,
            counter=self.counter,
            store=self.store,
            checkpointer=self.checkpointer,
            server_planned_calls=server_planned_calls,
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
            raise ValueError(EMPTY_INPUT)
        assessment, selected = self._assess(text)
        self.counter.calls = 0
        if self.pending:
            # A new turn cancels the paused action rather than implicitly resuming it.
            self.session_manager.reset_session(self.context.user_id)
            self.pending = False
            self.visit.reset()
        history_note = self._history_note(text)
        if selected is not None:
            if selected >= len(self.last_reply.hospitals):
                raise ValueError(PICK_LISTED_CANDIDATE)
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

    def _assess(self, text: str):
        """Return (assessment, chosen candidate index or None); alert once on critical input."""
        rules = assess_input(text)
        alerted = rules.triage.severity == "critical"
        if alerted:
            self._alert_emergency()
        selected = selected_index(text)
        if selected is not None and self.last_reply and not rules.blocked:
            # Choosing a listed candidate needs no classifier call.
            self._classifier_usage = ClassifierUsage()
            return rules, selected
        assessment = self.classifier.assess(text)
        self._classifier_usage = getattr(self.classifier, "last_usage", None) or ClassifierUsage()
        if not alerted and assessment.triage.severity == "critical":
            self._alert_emergency()
        if selected is not None and self.last_reply and not assessment.blocked:
            return assessment, selected
        return assessment, None

    def _alert_emergency(self) -> None:
        if self.on_emergency:
            self.on_emergency(EMERGENCY)

    def _history_note(self, text: str) -> str | None:
        if not any(marker in text for marker in RECENT_VISIT_MARKERS):
            return None
        visits = self.profiles.get_recent_visits(limit=1)
        if not visits:
            return NO_VISIT_HISTORY
        return (
            f"최근 저장한 방문 계획: {safe_data(visits[0]['name'])}. "
            "현재 수용 여부는 새로 확인해야 합니다."
        )

    def approve(self, approved: bool) -> ChatResult:
        if not self.pending or not self.graph.get_state(self.run_config).interrupts:
            raise ValueError(NO_PENDING_APPROVAL)
        self.visit.authorize(approved)
        self.pending = False
        self._classifier_usage = ClassifierUsage()
        self.counter.calls = 0
        decision = (
            {"type": "approve"} if approved else {"type": "reject", "message": USER_REJECTED_SAVE}
        )
        return self._run_graph(Command(resume={"decisions": [decision]}))

    def _run_graph(self, inputs, history_note=None) -> ChatResult:
        started = perf_counter()
        usage = RunUsage()
        succeeded = False
        error_info = None
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
                    note = VISIT_SAVED
                elif not self.visit.approved:
                    note = VISIT_REJECTED
                else:
                    note = VISIT_SAVE_FAILED
            succeeded = not bool(self.session.lookup_error)
        except Exception as error:
            error_info = error_metadata(error)
            self.last_error_type = error_info["error_type"]
            saved = self.visit.saved
            self.pending = False
            self.visit.reset()
            self.session_manager.reset_session(self.context.user_id)
            self.session.fail_lookup()
            reply = self.session.check_evidence(None)
            note = VISIT_SAVED_REPLY_FAILED if saved else REQUEST_FAILED
        self._session_model_calls += self.counter.calls
        self.last_diagnostics = RunDiagnostics.from_run(
            usage=usage,
            classifier_usage=self._classifier_usage,
            model_calls=self.counter.calls,
            session_model_calls=self._session_model_calls,
            elapsed_seconds=perf_counter() - started,
            succeeded=succeeded,
            error=error_info,
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
