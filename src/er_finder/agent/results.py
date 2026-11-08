"""Finalize graph outcomes, recover failures, and account for each completed turn."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from er_finder.agent.adapters import VisitSelection
from er_finder.agent.diagnostics import RunDiagnostics
from er_finder.agent.execution import ModelCounter
from er_finder.agent.renderer import render_reply
from er_finder.errors import FailureStage, error_metadata
from er_finder.memory.session import SessionManager
from er_finder.messages import (
    FAILURE_MESSAGES,
    VISIT_REJECTED,
    VISIT_SAVE_FAILED,
    VISIT_SAVED,
    VISIT_SAVED_REPLY_FAILED,
)
from er_finder.models import ERSearchReply
from er_finder.search.service import SearchSession
from er_finder.usage import ClassifierUsage, RunUsage


@dataclass
class ChatResult:
    reply: ERSearchReply
    text: str
    pending_approval: dict[str, Any] | None = None
    note: str | None = None
    diagnostics: RunDiagnostics | None = None


@dataclass(frozen=True)
class RunOutcome:
    result: ChatResult
    pending: bool


class RunResultProcessor:
    def __init__(
        self,
        session: SearchSession,
        visit: VisitSelection,
        session_manager: SessionManager,
        user_id: str,
        counter: ModelCounter,
    ) -> None:
        self.session, self.visit = session, visit
        self.session_manager, self.user_id = session_manager, user_id
        self.counter = counter
        self.reset()

    def reset(self) -> None:
        self._session_model_calls = 0

    def run(
        self,
        execute: Callable[[RunUsage], dict[str, Any]],
        *,
        classifier_usage: ClassifierUsage,
        history_note: str | None = None,
    ) -> RunOutcome:
        started = perf_counter()
        usage = RunUsage()
        succeeded = False
        error_info = None
        stage: FailureStage = "graph"
        try:
            output = execute(usage)
            stage = "output"
            pending = bool(output.get("__interrupt__"))
            finalized = output.get("structured_response")
            # Completed graphs already passed the output guard. Interrupts may not
            # run after_agent hooks and must finalize from current evidence here.
            reply = (
                finalized if not pending and isinstance(finalized, ERSearchReply)
                else self.session.check_evidence(finalized)
            )
            note = self.session.assessment.reason or history_note
            if self.visit.decided and not pending:
                if self.visit.saved:
                    note = VISIT_SAVED
                elif not self.visit.approved:
                    note = VISIT_REJECTED
                else:
                    note = VISIT_SAVE_FAILED
            error_info = self.session.failure
            succeeded = error_info is None
        except Exception as error:
            error_info = self.session.failure or error_metadata(error, stage=stage)
            saved = self.visit.saved
            pending = False
            self.visit.reset()
            self.session_manager.reset_session(self.user_id)
            self.session.fail_run(error_info)
            reply = self.session.check_evidence(None)
            note = VISIT_SAVED_REPLY_FAILED if saved else None
        if error_info is not None and note != VISIT_SAVED_REPLY_FAILED:
            note = FAILURE_MESSAGES[error_info["failure_category"]]
        self._session_model_calls += self.counter.calls
        diagnostics = RunDiagnostics.from_run(
            usage=usage,
            classifier_usage=classifier_usage,
            model_calls=self.counter.calls,
            session_model_calls=self._session_model_calls,
            elapsed_seconds=perf_counter() - started,
            succeeded=succeeded,
            error=error_info,
        )
        pending_approval = self.visit.pending if pending else None
        result = ChatResult(
            reply=reply,
            text=render_reply(reply, note=note, pending=pending_approval),
            pending_approval=pending_approval,
            note=note,
            diagnostics=diagnostics,
        )
        return RunOutcome(result, pending)
