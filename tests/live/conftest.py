"""Live contract tests call real paid/quota-limited APIs; they never run by accident."""

import os

import pytest

from er_finder.web.live import load_environment


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ER_RUN_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="실제 API 호출은 ER_RUN_LIVE=1 일 때만 실행합니다.")
    for item in items:
        if "tests/live" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def egen():
    """The real client module with the project .env applied (its key is read at import)."""
    from urllib.parse import unquote

    from er_finder.medical_api import client

    load_environment()
    key = os.environ.get("EGEN_SERVICE_KEY", "").strip()
    if not key:
        pytest.fail("EGEN_SERVICE_KEY 가 설정되지 않았습니다. 조용히 건너뛰지 않습니다.")
    client.SERVICE_KEY = unquote(key)
    return client


@pytest.fixture(scope="session")
def gemini_key():
    load_environment()
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        pytest.fail("GEMINI_API_KEY 가 설정되지 않았습니다. 조용히 건너뛰지 않습니다.")
