"""Typed operation failures and sanitized diagnostic identifiers."""

import re
from typing import Literal, TypedDict

FailureCategory = Literal["lookup", "model", "approval", "internal"]
FailureStage = Literal[
    "geocode",
    "list_nearby_ers",
    "get_er_bed_status",
    "get_severe_acceptance",
    "get_er_detail",
    "model",
    "approval",
    "save_visit_plan",
    "tools",
    "graph",
    "output",
]
FAILURE_CODES: dict[FailureCategory, str] = {
    "lookup": "lookup_failed",
    "model": "model_failed",
    "approval": "approval_failed",
    "internal": "internal_failed",
}


class OperationError(RuntimeError):
    def __init__(self, category: FailureCategory, stage: FailureStage) -> None:
        self.category, self.stage = category, stage
        super().__init__(FAILURE_CODES[category])


class ApprovalStateError(OperationError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__("approval", "approval")
        self.args = (message,)


class FailureMetadata(TypedDict):
    failure_category: FailureCategory
    failure_stage: FailureStage
    failure_code: str
    error_type: str | None
    status_code: int | None
    error_code: str | None
    error_param: str | None


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


def error_metadata(
    error: Exception | None,
    *,
    category: FailureCategory = "internal",
    stage: FailureStage = "graph",
) -> FailureMetadata:
    """Allow only exception class and fixed API diagnostic identifiers."""
    if isinstance(error, OperationError):
        category, stage = error.category, error.stage
        if isinstance(error.__cause__, Exception):
            error = error.__cause__
    status = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    param = getattr(error, "param", None)
    return {
        "failure_category": category,
        "failure_stage": stage,
        "failure_code": FAILURE_CODES[category],
        "error_type": type(error).__name__ if error is not None else None,
        "status_code": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code if isinstance(code, str) and code in ALLOWED_ERROR_CODES else None,
        "error_param": (
            param if isinstance(param, str) and _ERROR_PARAM.fullmatch(param) else None
        ),
    }
