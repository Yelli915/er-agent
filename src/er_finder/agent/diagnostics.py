"""Per-turn run metrics and the sanitized error identifiers reported with them."""

from dataclasses import dataclass

from er_finder.errors import FailureCategory, FailureMetadata, FailureStage
from er_finder.usage import ClassifierUsage, RunUsage


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
    failure_category: FailureCategory | None = None
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    error_code: str | None = None
    error_param: str | None = None

    @classmethod
    def from_run(
        cls,
        *,
        usage: RunUsage,
        classifier_usage: ClassifierUsage,
        model_calls: int,
        session_model_calls: int,
        elapsed_seconds: float,
        succeeded: bool,
        error: FailureMetadata | None = None,
    ) -> "RunDiagnostics":
        known = usage.known_responses or not model_calls
        return cls(
            model_calls=model_calls,
            session_model_calls=session_model_calls,
            elapsed_seconds=max(0.0, elapsed_seconds),
            input_tokens=usage.tokens["input_tokens"] if known else None,
            output_tokens=usage.tokens["output_tokens"] if known else None,
            total_tokens=usage.tokens["total_tokens"] if known else None,
            tokens_complete=usage.known_responses == usage.responses == model_calls,
            succeeded=succeeded,
            classifier_calls=classifier_usage.calls,
            classifier_elapsed_seconds=classifier_usage.elapsed_seconds,
            classifier_input_tokens=classifier_usage.input_tokens,
            classifier_output_tokens=classifier_usage.output_tokens,
            classifier_tokens_complete=classifier_usage.tokens_complete,
            failure_category=error["failure_category"] if error else None,
            failure_stage=error["failure_stage"] if error else None,
            failure_code=error["failure_code"] if error else None,
            error_type=error["error_type"] if error else None,
            status_code=error["status_code"] if error else None,
            error_code=error["error_code"] if error else None,
            error_param=error["error_param"] if error else None,
        )
