"""Exercise the live UI port offline; injected runners are explicitly test doubles."""

import importlib
from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from er_finder import config
from er_finder.memory.store import ERFinderStore
from er_finder.web.contracts import UIReply


@pytest.fixture
def live(monkeypatch):
    assert importlib.util.find_spec("er_finder.web.live") is not None, "live adapter is missing"
    module = importlib.import_module("er_finder.web.live")
    monkeypatch.setattr(config, "load_dotenv", lambda **kwargs: False)
    for name in ("GEMINI_API_KEY", "KAKAO_REST_API_KEY", "EGEN_SERVICE_KEY"):
        monkeypatch.setenv(name, "offline-test-value")
    return module


def reply():
    return UIReply(
        severity="standard",
        call_119_first=False,
        search_radius_km=5,
        hospitals=[
            dict(
                hpid="H1",
                name="테스트 병원",
                distance_km=1.2,
                er_beds_available=2,
                beds_updated_at="2026-09-11T09:00:00+09:00",
                address="테스트 주소",
            )
        ],
        data_timestamp="2026-09-11T09:00:00+09:00",
        next_action="방문 전 전화로 확인하세요.",
    )


class OfflineRunner:
    def __init__(self, *, user_id, transport, store):
        self.profile = ERFinderStore(store, user_id)
        self.transport = transport
        self.inputs = []
        self.approvals = []
        self.resets = 0
        self.failure = None
        self.pending = False
        self.malformed_pending = False
        self.repeat_interrupt = False

    def chat(self, text, *, force_refresh=False):
        self.inputs.append((text, force_refresh))
        if self.failure:
            raise self.failure
        self.pending = text == "1번"
        pending = None
        if self.pending:
            pending = dict(hpid="H1", name="테스트 병원", symptom_summary="테스트 증상")
            if self.malformed_pending:
                pending["hpid"] = "OTHER"
        return SimpleNamespace(reply=reply(), pending_approval=pending, note=None)

    def approve(self, approved):
        self.approvals.append(approved)
        if self.failure:
            raise self.failure
        if self.repeat_interrupt:
            return SimpleNamespace(
                reply=reply(),
                pending_approval={
                    "hpid": "H1",
                    "name": "테스트 병원",
                    "symptom_summary": "테스트 증상",
                },
                note=None,
            )
        assert self.pending
        self.pending = False
        if approved:
            self.profile.add_visit(hpid="H1", name="테스트 병원", symptom_summary="테스트 증상")
        return SimpleNamespace(reply=reply(), pending_approval=None, note="저장 처리 완료")

    def end_session(self):
        self.resets += 1
        self.pending = False


def connected(live, *, store=None):
    runners = []

    def factory(**kwargs):
        runner = OfflineRunner(**kwargs)
        runners.append(runner)
        return runner

    backend = live.RunnerBackend(runner_factory=factory, store=store)
    assert backend.connect().connected
    return backend, runners


def test_status_checks_keys_without_importing_or_constructing_runner(live, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("status must not construct the agent")

    monkeypatch.setenv("GEMINI_API_KEY", " ")
    backend = live.RunnerBackend(runner_factory=forbidden)
    status = backend.status()
    assert not status.connected
    assert status.checks == {
        "GEMINI_API_KEY": False,
        "KAKAO_REST_API_KEY": True,
        "EGEN_SERVICE_KEY": True,
    }
    assert "offline-test-value" not in repr(status)
    assert backend.connect().category == "keys"


def test_dotenv_respects_existing_environment(live, tmp_path, monkeypatch):
    from dotenv import load_dotenv

    monkeypatch.setattr(config, "load_dotenv", load_dotenv)
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=file-value\nKAKAO_REST_API_KEY=file-kakao\n", encoding="utf-8")
    monkeypatch.delenv("KAKAO_REST_API_KEY")
    live.load_environment(path)
    import os

    assert os.environ["GEMINI_API_KEY"] == "offline-test-value"
    assert os.environ["KAKAO_REST_API_KEY"] == "file-kakao"


def test_factory_failures_are_safe_and_do_not_return_preview_data(live):
    def broken(**kwargs):
        raise ImportError("offline-test-value /private/secret/module.py")

    backend = live.RunnerBackend(runner_factory=broken)
    status = backend.connect()
    assert not status.connected and status.category == "module"
    result = backend.chat("강남역", transport="car")
    assert result.reply is None and result.pending_approval is None
    assert "offline-test-value" not in status.message + result.note
    assert "/private/secret" not in status.message + result.note


def test_default_factory_connects_actual_modules_without_network_or_kakao(live, monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "")
    backend = live.RunnerBackend()
    status = backend.connect()
    assert status.connected and status.category == "ready"
    assert not status.checks["KAKAO_REST_API_KEY"]
    assert backend.select_hospital("FORGED").reply is None
    backend._invalidate("not_connected")
    assert not backend.status().connected


def test_injected_factory_must_supply_the_runner_contract(live):
    backend = live.RunnerBackend(runner_factory=lambda **kwargs: object())
    assert backend.connect().category == "contract"
    assert backend.chat("강남역", transport="car").reply is None


def test_chat_converts_pydantic_reply_and_refresh_reaches_runner(live):
    backend, runners = connected(live)
    result = backend.chat("강남역", transport="car", force_refresh=True)
    assert isinstance(result.reply, UIReply)
    assert result.reply.hospitals[0].hpid == "H1"
    assert runners[0].inputs == [("강남역", True)]
    assert backend.profile.get_recent_visits() == []


def test_selection_and_approval_write_real_profile_only_once(live):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    selected = backend.select_hospital("H1")
    assert selected.pending_approval.hpid == "H1"
    assert runners[0].inputs[-1][0] == "1번"
    assert backend.profile.get_recent_visits() == []
    assert backend.approve(True).pending_approval is None
    backend.approve(True)
    assert len(backend.profile.get_recent_visits()) == 1
    assert runners[0].approvals == [True]


def test_rejection_and_unknown_candidate_never_save(live):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    assert backend.select_hospital("OTHER").pending_approval is None
    assert len(runners[0].inputs) == 1
    backend.select_hospital("H1")
    backend.approve(False)
    assert backend.profile.get_recent_visits() == []


def test_pending_hospital_must_match_the_selected_candidate(live):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    runners[0].malformed_pending = True
    result = backend.select_hospital("H1")
    assert result.reply is None and result.pending_approval is None
    backend.approve(True)
    assert runners[0].approvals == []
    assert backend.profile.get_recent_visits() == []


@pytest.mark.parametrize("operation", ["chat", "approve"])
def test_failure_clears_previous_candidates_and_prevents_approval_replay(live, operation):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    backend.select_hospital("H1")
    runners[0].failure = RuntimeError("offline-test-value raw traceback")
    result = backend.chat("다시", transport="car") if operation == "chat" else backend.approve(True)
    assert result.reply is None and result.pending_approval is None
    assert "offline-test-value" not in result.note and "traceback" not in result.note
    count = len(runners[0].approvals)
    backend.approve(True)
    assert len(runners[0].approvals) == count
    assert backend.select_hospital("H1").pending_approval is None


def test_repeated_interrupt_does_not_offer_the_same_approval_again(live):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    backend.select_hospital("H1")
    runners[0].repeat_interrupt = True
    result = backend.approve(True)
    assert result.pending_approval is None and result.reply is None
    backend.approve(True)
    assert runners[0].approvals == [True]


def test_transport_change_reconstructs_runner_and_cancels_old_pending(live):
    backend, runners = connected(live)
    backend.chat("강남역", transport="car")
    backend.select_hospital("H1")
    result = backend.chat("강남역", transport="walk")
    assert result.pending_approval is None
    assert len(runners) == 2 and runners[1].transport == "walk"
    assert runners[0].resets >= 1
    backend.approve(True)
    assert backend.profile.get_recent_visits() == []


def test_reset_clears_graph_but_preserves_consented_profile(live):
    backend, runners = connected(live)
    backend.profile.set_home_address("서울", consent=True)
    backend.chat("강남역", transport="car")
    backend.select_hospital("H1")
    backend.reset()
    backend.approve(True)
    assert runners[0].resets >= 1
    assert backend.profile.get_home_address() == "서울"
    assert backend.profile.get_recent_visits() == []
    assert backend.select_hospital("H1").pending_approval is None


def test_forget_deletes_all_records_from_original_store_without_touching_other_users(live):
    store = InMemoryStore()
    backend, _ = connected(live, store=store)
    original_profile = backend.profile
    backend.profile.set_home_address("서울", consent=True)
    for index in range(25):
        backend.profile.add_visit(hpid=f"H{index}", name="테스트", symptom_summary="테스트")
    other = ERFinderStore(store, "other-user")
    other.set_home_address("부산", consent=True)
    backend.forget()
    assert original_profile.get_home_address() is None
    assert original_profile.get_recent_visits(limit=100) == []
    assert store.search(("er_finder", backend.user_id, "visits"), limit=100) == []
    assert other.get_home_address() == "부산"


def test_separate_backends_have_separate_profiles(live):
    one, _ = connected(live)
    two, _ = connected(live)
    one.profile.set_home_address("서울", consent=True)
    assert one.user_id != two.user_id
    assert two.profile.get_home_address() is None


def test_runner_cannot_offer_save_without_validated_selection(live):
    backend, _ = connected(live)
    backend.chat("강남역", transport="car")
    result = backend.chat("1번", transport="car")
    assert result.pending_approval is None and result.reply is None
    backend.approve(True)
    assert backend.profile.get_recent_visits() == []


def test_forget_reports_failed_graph_cleanup_even_when_profile_is_deleted(live, monkeypatch):
    backend, runners = connected(live)
    backend.profile.set_home_address("서울", consent=True)

    def broken_cleanup():
        raise RuntimeError("offline-test-value")

    monkeypatch.setattr(runners[0], "end_session", broken_cleanup)
    with pytest.raises(live.ConnectionUnavailable) as caught:
        backend.forget()
    assert "offline-test-value" not in str(caught.value)
    assert backend.profile.get_home_address() is None
    assert not backend.status().connected


def test_default_factory_constructs_configured_models_classifier_and_actual_runner(
    live, monkeypatch
):
    models, providers, arguments = [], [], []

    class Model:
        def __init__(self, **kwargs):
            models.append(kwargs)

    class Provider:
        def __init__(self, **kwargs):
            self.closed = False
            providers.append(self)

        def clear_cache(self):
            return None

        def close(self):
            self.closed = True

    def runner(**kwargs):
        arguments.append(kwargs)
        return OfflineRunner(**{key: kwargs[key] for key in ("user_id", "transport", "store")})

    factory = importlib.import_module("er_finder.agent.factory")
    modules = {
        "er_finder.agent.runner": SimpleNamespace(ERFinder=runner),
        "er_finder.agent.factory": factory,
        "er_finder.guardrails.triage": SimpleNamespace(InputClassifier=lambda model: model),
        "langchain_google_genai": SimpleNamespace(ChatGoogleGenerativeAI=Model),
        "er_finder.web.provider": SimpleNamespace(LiveProvider=Provider),
    }
    monkeypatch.setattr(factory.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setenv("ER_MAIN_MODEL", "configured-main")
    monkeypatch.setenv("ER_MAIN_THINKING_BUDGET", "0")
    monkeypatch.setenv("ER_MAIN_MAX_OUTPUT_TOKENS", "1234")
    monkeypatch.setenv("ER_CLASSIFIER_MODEL", "configured-classifier")
    monkeypatch.setenv("ER_CLASSIFIER_TEMPERATURE", "0")
    monkeypatch.setenv("ER_CLASSIFIER_MAX_TOKENS", "120")
    backend = live.RunnerBackend()
    assert backend.connect(transport="walk").connected
    assert [model["model"] for model in models] == ["configured-main", "configured-classifier"]
    assert models[0]["max_output_tokens"] == 1234 and "temperature" not in models[0]
    assert models[0]["thinking_budget"] == 0
    assert models[1]["max_output_tokens"] == 120 and models[1]["temperature"] == 0
    assert arguments[0]["provider"] is providers[0]
    assert arguments[0]["transport"] == "walk"
    backend.profile.set_home_address("서울", consent=True)
    assert ERFinderStore(arguments[0]["store"], backend.user_id).get_home_address() == "서울"
    assert backend.connect(transport="car").connected
    assert providers[0].closed


def test_default_factory_closes_resources_when_runner_construction_fails(live, monkeypatch):
    models, providers = [], []

    def model(**kwargs):
        models.append(kwargs)
        return object()

    class Provider:
        def __init__(self, **kwargs):
            self.closed = False
            providers.append(self)

        def close(self):
            self.closed = True

    def broken_runner(**kwargs):
        raise AttributeError("offline-test-value")

    factory = importlib.import_module("er_finder.agent.factory")
    modules = {
        "er_finder.agent.runner": SimpleNamespace(ERFinder=broken_runner),
        "er_finder.agent.factory": factory,
        "er_finder.guardrails.triage": SimpleNamespace(InputClassifier=lambda model: model),
        "langchain_google_genai": SimpleNamespace(ChatGoogleGenerativeAI=model),
        "er_finder.web.provider": SimpleNamespace(LiveProvider=Provider),
    }
    monkeypatch.setattr(factory.importlib, "import_module", lambda name: modules[name])
    backend = live.RunnerBackend()
    assert backend.connect().category == "contract"
    assert providers and providers[0].closed
