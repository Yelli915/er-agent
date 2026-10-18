"""One real end-to-end turn: model tool schemas, classifier, E-Gen and the call budget."""

import pytest

from er_finder.evaluation.benchmark import LOCATION

pytestmark = pytest.mark.live


@pytest.fixture
def finder(egen, gemini_key):
    from er_finder.agent.factory import build_models
    from er_finder.agent.runner import ERFinder
    from er_finder.config import Settings
    from er_finder.web.provider import LiveProvider

    settings = Settings()
    model, classifier = build_models(settings)
    runner = ERFinder(provider=LiveProvider(kakao_key=""), model=model, classifier=classifier)
    yield runner
    runner.close()


def test_classifier_reports_usage_and_a_valid_label(finder):
    classifier = finder.classifier
    result = classifier.assess("발목을 삐어서 부었어요")
    assert result.triage.severity in {"standard", "urgent", "critical"}
    assert classifier.last_usage.calls == 1
    assert classifier.last_usage.tokens_complete and classifier.last_usage.input_tokens > 0


def test_standard_search_succeeds_within_the_call_budget(finder):
    result = finder.chat(f"{LOCATION} 근처에서 발목이 아파요")
    d = result.diagnostics
    assert d.succeeded, f"조회 실패: {d.error_type} {d.error_code}"
    assert result.reply.hospitals
    # Server-planned steps skip the model; a regression to one call per step shows up here.
    assert d.model_calls <= 2, f"메인 모델 호출이 {d.model_calls}회입니다."
    assert d.tokens_complete and d.classifier_calls == 1
