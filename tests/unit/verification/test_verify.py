"""The verification report must never promote absent evidence to a pass."""

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def verify():
    path = ROOT / "scripts" / "verify.py"
    assert path.exists(), "the offline verification runner has not been implemented"
    spec = importlib.util.spec_from_file_location("verification_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_junit_counts_setup_errors_as_errors_not_passes(verify, tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites><testsuite tests="5" failures="1" errors="1" skipped="1">'
        '<testcase name="ok1"/><testcase name="ok2"/>'
        '<testcase name="bad"><failure message="assertion"/></testcase>'
        '<testcase name="setup"><error message="fixture failed"/></testcase>'
        '<testcase name="skip"><skipped/></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert verify.read_junit(report) == {
        "total": 5,
        "passed": 2,
        "failed": 1,
        "errors": 1,
        "skipped": 1,
    }


@pytest.mark.parametrize("contents", ["", "<not-junit/>", "<testsuites>"])
def test_invalid_junit_is_not_treated_as_zero_failure_success(verify, tmp_path, contents):
    report = tmp_path / "junit.xml"
    report.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        verify.read_junit(report)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        '"""Future implementation."""',
        "import typing\npass\n",
        "class Backend:\n    pass\n",
        "def run():\n    pass\n",
        "def run():\n    raise NotImplementedError\n",
    ],
)
def test_backend_preflight_reports_empty_and_stub_modules(verify, tmp_path, contents):
    target = tmp_path / "backend.py"
    target.write_text(contents, encoding="utf-8")
    assert verify.find_blockers(tmp_path, backend_files=("backend.py",), fixture_files=())


def test_backend_preflight_also_reports_missing_and_invalid_fixtures(verify, tmp_path):
    (tmp_path / "response.json").write_text("{", encoding="utf-8")
    blockers = verify.find_blockers(
        tmp_path, backend_files=("missing.py",), fixture_files=("response.json",)
    )
    assert {item["path"] for item in blockers} == {"missing.py", "response.json"}


def test_nonempty_implementation_is_only_a_preflight_result(verify, tmp_path):
    (tmp_path / "backend.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    assert verify.find_blockers(tmp_path, backend_files=("backend.py",), fixture_files=()) == []


def test_preflight_allows_empty_exception_class_in_implemented_module(verify, tmp_path):
    (tmp_path / "backend.py").write_text(
        "class ToolOrderError(ValueError):\n    pass\n\n"
        "def validate(order):\n    if not order:\n        raise ToolOrderError('missing')\n"
        "    return order\n",
        encoding="utf-8",
    )
    assert verify.find_blockers(tmp_path, backend_files=("backend.py",), fixture_files=()) == []


@pytest.mark.parametrize("returncode,expected_status", [(0, "PASS"), (1, "FAIL"), (5, "NO_TESTS")])
@pytest.mark.parametrize("scope", ["all", "web"])
def test_report_preserves_pytest_exit_and_does_not_claim_integration(
    verify, tmp_path, monkeypatch, capsys, returncode, expected_status, scope
):
    output = tmp_path / "results"

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert command[command.index("-m", 3) + 1] == "not live"
        assert "scripts.verify" in command
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "ER_DEMO_MODE" not in kwargs["env"]
        targets = [part for part in command if part.startswith("tests/")]
        assert targets == (
            ["tests/unit/web", "tests/integration", "tests/unit/verification"]
            if scope == "web"
            else []
        )
        (output / "junit.xml").write_text(
            '<testsuite><testcase name="memory_check"/></testsuite>', encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, returncode, "pytest output", "")

    monkeypatch.setattr(verify.subprocess, "run", run)
    code = verify.run_verification(
        tmp_path, output, timeout=5, require_integration=False, scope=scope
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["local_tests"]["status"] == expected_status
    assert report["local_tests"]["exit_code"] == returncode
    assert report["integration"]["status"] == "BLOCKED"
    assert report["integration"]["blockers"]
    assert report["provenance"]["clinical_validation"] is False
    assert report["provenance"]["live_api_tests"] == "EXCLUDED"
    assert report["test_scope"] == scope
    assert ("web UI" in report["provenance"]["scope"]) is (scope == "web")
    if expected_status == "FAIL":
        assert "pytest output" in capsys.readouterr().out
    assert (code == 0) is (returncode == 0)


def test_timeout_replaces_stale_junit_and_preserves_partial_log(verify, tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    (output / "junit.xml").write_text('<testsuite><testcase name="old"/></testsuite>')

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 1, "", "not a git repository")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial output")

    monkeypatch.setattr(verify.subprocess, "run", run)
    assert verify.run_verification(tmp_path, output, timeout=0.01, require_integration=False) != 0
    report = json.loads((output / "report.json").read_text())
    assert report["local_tests"]["status"] == "TIMEOUT"
    assert report["local_tests"]["counts"] is None
    assert "partial output" in (output / "pytest.log").read_text()
    assert not (output / "junit.xml").exists()


def test_require_integration_fails_even_when_local_tests_pass(verify, tmp_path, monkeypatch):
    output = tmp_path / "results"

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        (output / "junit.xml").write_text('<testsuite><testcase name="ok"/></testsuite>')
        return subprocess.CompletedProcess(command, 0, "1 passed", "")

    monkeypatch.setattr(verify.subprocess, "run", run)
    assert verify.run_verification(tmp_path, output, timeout=5, require_integration=True) == 2


def test_offline_guard_blocks_dns_and_connection_without_contacting_network(verify):
    cleanups = []

    class Config:
        add_cleanup = cleanups.append

    original = socket.getaddrinfo
    verify.pytest_configure(Config())
    try:
        with pytest.raises(RuntimeError, match="network"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as connection:
            with pytest.raises(RuntimeError, match="network"):
                connection.connect(("127.0.0.1", 9))
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
    assert socket.getaddrinfo is original


def test_offline_guard_allows_only_in_process_loopback_pairs(verify):
    cleanups = []

    class Config:
        add_cleanup = cleanups.append

    verify.pytest_configure(Config())
    try:
        a, b = socket.socketpair()  # Windows builds this from a loopback listener
        a.close()
        b.close()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.socket() as client:
                client.connect(("127.0.0.1", port))
            with socket.socket() as client:
                with pytest.raises(RuntimeError, match="network"):
                    client.connect(("127.0.0.1", port + 1 if port < 65535 else port - 1))
                with pytest.raises(RuntimeError, match="network"):
                    client.connect(("203.0.113.1", port))
        with pytest.raises(RuntimeError, match="network"):
            socket.socket().connect(("127.0.0.1", port))  # listener closed: no longer trusted
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


def test_offline_guard_blocks_legacy_dns_and_udp_without_contacting_network(verify):
    cleanups = []

    class Config:
        add_cleanup = cleanups.append

    verify.pytest_configure(Config())
    try:
        with pytest.raises(RuntimeError, match="network"):
            socket.gethostbyname("localhost")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            with pytest.raises(RuntimeError, match="network"):
                connection.sendto(b"offline check", ("127.0.0.1", 9))
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


def test_runner_executes_a_small_offline_suite_and_excludes_live_tests(verify, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify.py").write_text(
        (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = live: contacts external services\n", encoding="utf-8"
    )
    (tmp_path / "test_sample.py").write_text(
        "import pytest, socket\n"
        "def test_offline_guard():\n"
        "    with pytest.raises(RuntimeError, match='network'):\n"
        "        socket.create_connection(('127.0.0.1', 9))\n"
        "@pytest.mark.live\n"
        "def test_live_is_excluded():\n"
        "    raise AssertionError('live test must not execute')\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"
    assert verify.run_verification(tmp_path, output, timeout=15, require_integration=False) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["local_tests"]["counts"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert "1 deselected" in (output / "pytest.log").read_text()


def test_runner_does_not_pass_a_real_suite_when_every_test_is_skipped(verify, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify.py").write_text(
        (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = live: contacts external services\n", encoding="utf-8"
    )
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='implementation is unavailable')\n"
        "def test_not_executed():\n"
        "    raise AssertionError('this test must be skipped')\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"
    code = verify.run_verification(tmp_path, output, timeout=15, require_integration=False)
    report = json.loads((output / "report.json").read_text())
    assert report["local_tests"]["counts"] == {
        "total": 1,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 1,
    }
    assert report["local_tests"]["exit_code"] == 0
    assert report["local_tests"]["status"] == "NO_TESTS"
    assert code != 0


def test_benchmark_refuses_to_call_live_apis_without_opt_in():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert "P90" in result.stderr


def test_collection_error_still_reports_independent_test_results(verify, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify.py").write_text(
        (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "test_good.py").write_text("def test_available():\n    assert 2 + 2 == 4\n")
    (tmp_path / "test_broken.py").write_text("raise ImportError('missing team interface')\n")
    output = tmp_path / "results"
    assert verify.run_verification(tmp_path, output, timeout=15, require_integration=False) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["local_tests"]["status"] == "FAIL"
    assert report["local_tests"]["counts"] == {
        "total": 2,
        "passed": 1,
        "failed": 0,
        "errors": 1,
        "skipped": 0,
    }
