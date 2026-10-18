"""Per-turn run metrics and the sanitized error identifiers reported with them."""

import re
from dataclasses import dataclass

from er_finder.usage import ClassifierUsage, RunUsage

ALLOWED_ERROR_CODES = frozenset(
    {
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
)
_ERROR_PARAM = re.compile(
    r"(?:tools(?:\[\d+\])?(?:\.function(?:\.(?:name|description|parameters))?)?"
    r"|response_format(?:\.json_schema(?:\.(?:name|schema|strict))?)?"
    r"|model|max_tokens|max_completion_tokens|temperature|messages)"
)


def error_metadata(error: Exception) -> dict:
    """Allow only exception class and fixed API diagnostic identifiers."""
    status = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    param = getattr(error, "param", None)
    return {
        "error_type": type(error).__name__,
        "status_code": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code if isinstance(code, str) and code in ALLOWED_ERROR_CODES else None,
        "error_param": (
            param if isinstance(param, str) and _ERROR_PARAM.fullmatch(param) else None
        ),
    }


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
        error: dict | None = None,
    ) -> "RunDiagnostics":
        known = usage.known_responses or not model_calls
        return cls(
            model_calls=model_calls,
            session_model_calls=session_model_calls,
            elapsed_seconds=max(0.0, elapsed_seconds),
            **{key: value if known else None for key, value in usage.tokens.items()},
            tokens_complete=usage.known_responses == usage.responses == model_calls,
            succeeded=succeeded,
            classifier_calls=classifier_usage.calls,
            classifier_elapsed_seconds=classifier_usage.elapsed_seconds,
            classifier_input_tokens=classifier_usage.input_tokens,
            classifier_output_tokens=classifier_usage.output_tokens,
            classifier_tokens_complete=classifier_usage.tokens_complete,
            **(error or {}),
        )
