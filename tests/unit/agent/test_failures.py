"""Failures keep their origin while removing unsafe candidates and replayable writes."""

import pytest

from er_finder.errors import ApprovalStateError, OperationError, error_metadata

SECRET = "TEST_ONLY_PRIVATE_REQUEST_AND_KEY"


@pytest.mark.parametrize(
    "stage",
    [
        "geocode",
        "list_nearby_ers",
        "get_er_bed_status",
        "get_severe_acceptance",
        "get_er_detail",
    ],
)
def test_provider_errors_keep_the_lookup_stage_even_for_value_errors(
    make_finder, monkeypatch, stage
):
    finder = make_finder()
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(True)
        raise ValueError(SECRET)

    monkeypatch.setattr(finder.session.provider, stage, fail)
    result = finder.chat("가슴이 답답하고 식은땀이 나요. 강남구 역삼동")
    assert result.diagnostics.failure_category == "lookup"
    assert result.diagnostics.failure_stage == stage
    assert result.diagnostics.failure_code == "lookup_failed"
    assert result.diagnostics.error_type == "ValueError"
    assert not result.diagnostics.succeeded
    assert not result.reply.hospitals and not result.pending_approval
    assert finder.session.lookup_error is not None
    assert not finder.session.next_calls()
    assert 1 <= len(attempts) <= (3 if stage == "get_er_detail" else 1)
    assert SECRET not in repr(result) + repr(finder.session.failure)


def test_graph_errors_do_not_claim_the_medical_api_failed(make_finder, monkeypatch):
    finder = make_finder()

    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(finder.graph, "invoke", fail)
    result = finder.chat("강남구 역삼동 발목이 아파요")
    assert result.diagnostics.failure_category == "internal"
    assert result.diagnostics.failure_stage == "graph"
    assert not result.diagnostics.succeeded and not result.reply.hospitals
    assert finder.session.lookup_error is None
    assert not finder.session.next_calls()
    assert SECRET not in repr(result)


@pytest.mark.parametrize("write_before_error", [False, True])
def test_store_failures_are_approval_failures_without_automatic_replay(
    make_finder,
    monkeypatch,
    write_before_error,
):
    finder = make_finder()
    finder.chat("강남구 역삼동 발목이 아파요")
    finder.chat("1\ubc88")
    original = finder.profiles.add_visit
    attempts = []

    def fail(**kwargs):
        attempts.append(True)
        if write_before_error:
            original(**kwargs)
        raise OSError(SECRET)

    monkeypatch.setattr(finder.profiles, "add_visit", fail)
    result = finder.approve(True)
    assert result.diagnostics.failure_category == "approval"
    assert result.diagnostics.failure_stage == "save_visit_plan"
    assert result.diagnostics.error_type == "OSError"
    assert not result.diagnostics.succeeded and not result.pending_approval
    assert finder.session.lookup_error is None
    assert result.reply.hospitals == []
    with pytest.raises(ApprovalStateError):
        finder.approve(True)
    assert len(attempts) == 1
    assert len(finder.profiles.get_recent_visits()) == int(write_before_error)
    assert SECRET not in repr(result) + repr(finder.session.failure)


def test_missing_approval_is_typed_and_remains_a_value_error(make_finder):
    with pytest.raises(ValueError) as caught:
        make_finder().approve(True)
    assert isinstance(caught.value, ApprovalStateError)
    failure = error_metadata(caught.value)
    assert failure["failure_category"] == "approval"
    assert failure["failure_stage"] == "approval"


def test_operation_metadata_keeps_only_allowed_api_identifiers():
    original = RuntimeError(SECRET)
    original.status_code = True
    original.code = SECRET
    original.param = SECRET
    try:
        raise OperationError("model", "model") from original
    except OperationError as error:
        metadata = error_metadata(error)
        assert SECRET not in str(error) + repr(metadata)
    assert metadata["failure_code"] == "model_failed"
    assert metadata["error_type"] == "RuntimeError"
    assert metadata["status_code"] is None
    assert metadata["error_code"] is None
    assert metadata["error_param"] is None
