"""Validated runtime options. Credentials remain in environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ER_", extra="ignore")

    main_model: str = Field(default="gpt-5-mini", min_length=1)
    main_reasoning_effort: str = "low"
    main_max_output_tokens: int = Field(default=4000, ge=256, le=16000)
    classifier_model: str = Field(default="gpt-4o-mini", min_length=1)
    classifier_temperature: float = Field(default=0, ge=0, le=2)
    classifier_max_tokens: int = Field(default=200, ge=64, le=2000)
    request_timeout: float = Field(default=5, gt=0, le=60)
    model_timeout: float = Field(default=60, gt=0, le=180)


REQUIRED_KEYS = ("OPENAI_API_KEY", "KAKAO_REST_API_KEY", "EGEN_SERVICE_KEY")
AGENT_REQUIRED_KEYS = ("OPENAI_API_KEY", "EGEN_SERVICE_KEY")


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Read the project .env without replacing values supplied by the launching process."""
    path = dotenv_path if dotenv_path is not None else Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=path, override=False)


def credential_status() -> dict[str, bool]:
    load_environment()
    return {name: bool(os.environ.get(name, "").strip()) for name in REQUIRED_KEYS}
