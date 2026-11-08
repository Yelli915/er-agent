"""Run the real LangGraph agent using the shared memory and search implementations."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import merge_configs
from langgraph.types import Command

from er_finder.agent.adapters import selected_index
from er_finder.agent.assembly import build_runner_dependencies
from er_finder.agent.diagnostics import RunDiagnostics
from er_finder.agent.progress import ProgressCallback, progress_lines
from er_finder.agent.results import ChatResult, RunResultProcessor
from er_finder.errors import ApprovalStateError
from er_finder.guardrails.contracts import Classifier
from er_finder.guardrails.evidence import safe_data
from er_finder.guardrails.input_guard import InputAssessment, assess_input
from er_finder.guardrails.pii import mask_pii
from er_finder.memory.context import TransportMode
from er_finder.messages import (
    EMPTY_INPUT,
    NO_PENDING_APPROVAL,
    NO_VISIT_HISTORY,
    PICK_LISTED_CANDIDATE,
    USER_REJECTED_SAVE,
)
from er_finder.models import EMERGENCY, ERSearchReply
from er_finder.search.contracts import SearchProvider
from er_finder.usage import ClassifierUsage

__all__ = ["ChatResult", "ERFinder", "RunDiagnostics"]

RECENT_VISIT_MARKERS = ("지난번", "최근 방문")


class ERFinder:
    def __init__(
        self,
        provider: SearchProvider,
        model: Any = None,
        classifier: Classifier | None = None,
        user_id: str = "local-user",
        transport: TransportMode = "car",
        store: Any = None,
        checkpointer: Any = None,
        on_emergency: Callable[[str], None] | None = None,
        server_planned_calls: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.on_progress = on_progress
        dependencies = build_runner_dependencies(
            provider,
            model,
            classifier=classifier,
            user_id=user_id,
            transport=transport,
            store=store,
            checkpointer=checkpointer,
            server_planned_calls=server_planned_calls,
        )
        self.context = dependencies.context
        self.session = dependencies.session
        self.store = dependencies.store
        self.profiles = dependencies.profiles
        self.visit = dependencies.visit
        self.checkpointer = dependencies.checkpointer
        self.session_manager = dependencies.session_manager
        self.classifier = dependencies.classifier
        self.counter = dependencies.counter
        self.graph = dependencies.graph
        self.run_config = dependencies.run_config
        self.on_emergency = on_emergency
        self._results = RunResultProcessor(
            self.session,
            self.visit,
            self.session_manager,
            user_id,
            self.counter,
        )
        self.last_reply: ERSearchReply | None = None
        self.pending = False
        self.last_diagnostics: RunDiagnostics | None = None
        self.last_error_type: str | None = None
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
            if self.last_reply is None or selected >= len(self.last_reply.hospitals):
                raise ApprovalStateError(PICK_LISTED_CANDIDATE)
            self.visit.prepare(self.last_reply.hospitals[selected], self.session.symptom_text)
        else:
            self.visit.reset()
            self.session.begin(
                text,
                self.profiles.get_home_address(),
                force_refresh=force_refresh,
                assessment=assessment,
            )
        return self._run_graph({"messages": [HumanMessage(content=text)]}, history_note)

    def _assess(self, text: str) -> tuple[InputAssessment, int | None]:
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
            raise ApprovalStateError(NO_PENDING_APPROVAL)
        self.visit.authorize(approved)
        self.pending = False
        self._classifier_usage = ClassifierUsage()
        self.counter.calls = 0
        decision = (
            {"type": "approve"} if approved else {"type": "reject", "message": USER_REJECTED_SAVE}
        )
        return self._run_graph(Command(resume={"decisions": [decision]}))

    def _run_graph(self, inputs: Any, history_note: str | None = None) -> ChatResult:
        self.last_error_type = None
        outcome = self._results.run(
            lambda usage: self._execute(
                inputs, merge_configs(self.run_config, {"callbacks": [usage]})
            ),
            classifier_usage=self._classifier_usage,
            history_note=history_note,
        )
        self.pending = outcome.pending
        self.last_reply = outcome.result.reply
        self.last_diagnostics = outcome.result.diagnostics
        self.last_error_type = self.last_diagnostics.error_type if self.last_diagnostics else None
        return outcome.result

    def _execute(self, inputs: Any, config: Any) -> dict[str, Any]:
        """Run the graph; with a progress listener, stream updates and read the final state."""
        if self.on_progress is None:
            output: dict[str, Any] = self.graph.invoke(inputs, config=config, context=self.context)
            return output
        for chunk in self.graph.stream(
            inputs, config=config, context=self.context, stream_mode="updates"
        ):
            self._notify(chunk)
        state = self.graph.get_state(self.run_config)
        return {**state.values, "__interrupt__": state.interrupts}

    def _notify(self, chunk: Any) -> None:
        for line in progress_lines(chunk):
            try:
                if self.on_progress is not None:
                    self.on_progress(line)
            except Exception:
                # A failing display must not abort a search that may already have written.
                self.on_progress = None

    def end_session(self) -> None:
        self.session_manager.end_session(self.context.user_id)
        self.session.clear()
        self.visit.reset()
        self.pending = False
        self.last_reply = None
        self.last_diagnostics = None
        self.last_error_type = None
        self._results.reset()

    def close(self) -> None:
        try:
            self.end_session()
        finally:
            self.session.provider.close()
