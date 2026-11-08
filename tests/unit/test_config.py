import pytest
from pydantic import ValidationError

from er_finder.config import Settings


def test_model_settings_keep_network_timeouts_separate(monkeypatch):
    monkeypatch.setenv("ER_MAIN_MODEL", "configured-model")
    monkeypatch.setenv("ER_MODEL_TIMEOUT", "60")
    settings = Settings()
    assert settings.main_model == "configured-model"
    assert settings.model_timeout == 60


@pytest.mark.parametrize(
    "name,value", [("ER_MODEL_TIMEOUT", "0"), ("ER_MAIN_MAX_OUTPUT_TOKENS", "-1")]
)
def test_invalid_runtime_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()
