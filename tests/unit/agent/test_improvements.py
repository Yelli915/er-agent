"""Production path: no answer-generation model, grounded parsing and safe follow-ups."""

from types import SimpleNamespace

import pytest

from er_finder.guardrails.input_guard import assess_input
from er_finder.guardrails.triage import InputClassifier
from tests.unit.agent.conftest import OfflineModel


@pytest.mark.parametrize("provider_failure", [False, True])
def test_completed_graph_finalizes_evidence_once(make_finder, monkeypatch, provider_failure):
    finder = make_finder()
    calls = []
    original = finder.session.check_evidence

    def check(proposal):
        calls.append(True)
        return original(proposal)

    def fail(*args, **kwargs):
        raise RuntimeError("lookup unavailable")

    monkeypatch.setattr(finder.session, "check_evidence", check)
    if provider_failure:
        monkeypatch.setattr(finder.session.provider, "geocode", fail)
    result = finder.chat("강남역 발목 통증")
    assert result.diagnostics.succeeded is not provider_failure
    assert len(calls) == 1


def test_new_symptom_replaces_old_condition_while_waiting_for_location(make_finder):
    finder = make_finder()
    finder.chat("강남역 화상으로 물집이 잡혔어요")
    result = finder.chat("회사 근처에서 발목 통증이 있어요")
    assert not result.reply.hospitals
    assert finder.session.triage.condition is None
    assert finder.session.symptom_text == "회사 근처에서 발목 통증이 있어요"
    finder.chat("역삼동")
    assert finder.session.triage.condition is None
    assert "발목" in finder.session.symptom_text
    assert "get_severe_acceptance" not in [c["name"] for c in finder.session.audit]


def test_explicit_model_landmark_takes_priority_over_workplace_question(make_finder):
    class Classifier:
        def assess(self, text):
            result = assess_input(text)
            result.location_query = "코엑스 동문"
            result.needs_location = True
            return result

    finder = make_finder(classifier=Classifier())
    finder.chat("코엑스 동문 회사 근처로 바꿔줘")
    assert finder.session.provider.queries == ["코엑스 동문"]
    assert not finder.session.assessment.needs_location


@pytest.mark.parametrize("approve", [True, False])
@pytest.mark.parametrize("make_finder", [True], indirect=True)
def test_zero_model_search_and_approval_keep_real_checkpoint_contract(make_finder, approve):
    model = OfflineModel(strategy="failure")
    finder = make_finder(model=model)
    result = finder.chat("강남구 역삼동 발목이 아파요")
    assert result.reply.hospitals and result.diagnostics.succeeded
    assert result.diagnostics.model_calls == result.diagnostics.total_tokens == 0
    assert finder.chat("2번 병원으로 할게").pending_approval
    final = finder.approve(approve)
    assert final.diagnostics.succeeded and not final.pending_approval
    assert final.diagnostics.model_calls == 0
    assert len(finder.profiles.get_recent_visits()) == int(approve)
    assert model._requests == []
    with pytest.raises(ValueError):
        finder.approve(True)


def test_unknown_workplace_clears_candidates_and_requires_location(make_finder):
    finder = make_finder()
    finder.chat("강남구 역삼동 가슴이 답답해요")
    finder.chat("1번")
    result = finder.chat("회사 근처로 바꿔줘")
    assert not result.reply.hospitals and not result.pending_approval
    assert finder.session.location is None and finder.session.audit == []
    assert "위치" in result.reply.next_action
    assert result.reply.severity == "urgent"
    with pytest.raises(ValueError):
        finder.approve(True)
    corrected = finder.chat("합정동")
    assert finder.session.location_query == "합정동"
    assert corrected.reply.severity == "urgent"


def test_classifier_extracts_literal_landmark_but_rejects_invented_address(make_finder):
    def classifier(query):
        output = dict(
            severity="standard", condition=None, confidence=0.95, injection=False,
            location_query=query, needs_location=False,
        )
        return InputClassifier(SimpleNamespace(with_structured_output=lambda _: SimpleNamespace(
            invoke=lambda *args, **kwargs: SimpleNamespace(model_dump=lambda: output)
        )))

    text = "코엑스 동문에서 발목을 삐었어"
    finder = make_finder(classifier=classifier("코엑스 동문"))
    finder.chat(text)
    assert finder.session.provider.queries == ["코엑스 동문"]
    assert classifier("서울 강남구 영동대로 513").assess(text).location_query is None


def test_location_correction_invalidates_old_candidate_and_keeps_symptom(make_finder):
    finder = make_finder()
    finder.chat("강남역에서 가슴이 답답해요")
    finder.chat("1번")
    corrected = finder.chat("아니, 합정동이야")
    assert corrected.pending_approval is None
    assert finder.session.provider.queries[-1] == "합정동"
    assert corrected.reply.severity == "urgent"
    assert finder.profiles.get_recent_visits() == []


def test_same_location_requeries_evidence(make_finder):
    finder = make_finder()
    finder.chat("강남역 발목 통증")
    result = finder.chat("아까 위치에서 다시 찾아줘", force_refresh=True)
    assert result.reply.hospitals and finder.session.force_refresh
    assert finder.session.audit[0]["name"] == "list_nearby_ers"


@pytest.mark.parametrize(
    "outcome", ["resolved", "extracted", "missing", "invented", "failure", "offline"]
)
def test_critical_landmark_never_reuses_previous_location(make_finder, monkeypatch, outcome):
    from er_finder.models import EMERGENCY

    events = []
    finder = make_finder(on_emergency=lambda text: events.append(text))
    finder.chat("강남역 발목 통증")
    assert finder.session.location is not None
    if outcome == "resolved":
        def geocode(query):
            finder.session.provider.queries.append(query)
            return dict(
                found=True, lat=37.795, lon=128.907, sido="강원특별자치도", sigungu="강릉시"
            )

        monkeypatch.setattr(finder.session.provider, "geocode", geocode)

    def invoke(*args, **kwargs):
        events.append("classifier")
        if outcome == "failure":
            raise RuntimeError("classification unavailable")
        query = {
            "resolved": "코엑스 동문", "extracted": "코엑스 동문",
            "missing": None, "invented": "서울 강남구 영동대로 513"
        }[outcome]
        return SimpleNamespace(model_dump=lambda: dict(
            severity="standard", condition=None, injection=False, confidence=0.95,
            location_query=query, needs_location=False,
        ))

    model = SimpleNamespace(with_structured_output=lambda _: SimpleNamespace(invoke=invoke))
    finder.classifier = InputClassifier(None if outcome == "offline" else model)
    result = finder.chat("코엑스 동문에서 의식이 없어요")

    assert events == ([EMERGENCY] if outcome == "offline" else [EMERGENCY, "classifier"])
    assert result.reply.severity == "critical" and result.reply.call_119_first
    assert result.text.startswith(EMERGENCY)
    if outcome == "resolved":
        assert finder.session.provider.queries[-1] == "코엑스 동문"
        assert finder.session.location["lat"] == 37.795
        assert finder.session.audit[0] == {"name": "geocode"}
        assert [h.hpid for h in result.reply.hospitals] == ["TEST-GL-1"]
        return
    assert finder.session.location is None
    assert not result.reply.hospitals
    if outcome == "extracted":
        # The fixture cannot resolve this landmark; the previous coordinates stay discarded.
        assert finder.session.provider.queries[-1] == "코엑스 동문"
        assert finder.session.audit == [{"name": "geocode"}]
    else:
        assert finder.session.assessment.needs_location
        assert finder.session.location_query is None
        assert finder.session.audit == []
