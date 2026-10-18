"""Demo mode runs the real graph with canned data, without any API key."""

import pytest

from er_finder.demo import DEMO_NOTICE, is_demo
from er_finder.web.live import RunnerBackend


@pytest.fixture
def backend(monkeypatch):
    for key in ("GEMINI_API_KEY", "KAKAO_REST_API_KEY", "EGEN_SERVICE_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("ER_DEMO", "1")
    backend = RunnerBackend()
    yield backend
    backend._invalidate("not_connected")


def test_demo_flag(monkeypatch):
    monkeypatch.delenv("ER_DEMO", raising=False)
    assert not is_demo()
    monkeypatch.setenv("ER_DEMO", "1")
    assert is_demo()
    assert "샘플" in DEMO_NOTICE


def test_connects_without_keys_and_returns_sample_candidates(backend):
    assert backend.connect().connected
    result = backend.chat("37.4979, 127.0276 근처에서 발목이 아파요", transport="car")
    assert result.reply is not None
    assert result.reply.hospitals
    assert all("샘플" in hospital.name for hospital in result.reply.hospitals)


def test_emergency_input_is_short_circuited_before_the_model(backend):
    backend.connect()
    result = backend.chat(
        "37.4979, 127.0276 근처에서 의식이 없고 숨을 쉬지 않아요", transport="car"
    )
    assert result.reply is not None
    assert result.reply.call_119_first
    assert result.reply.severity == "critical"


def test_without_demo_flag_keys_are_still_required(monkeypatch):
    monkeypatch.delenv("ER_DEMO", raising=False)
    for key in ("GEMINI_API_KEY", "EGEN_SERVICE_KEY"):
        monkeypatch.setenv(key, "")
    assert not RunnerBackend().connect().connected
