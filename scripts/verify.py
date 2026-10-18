"""Run offline checks and record evidence separately from backend readiness.

The same module is loaded by pytest as a plugin to deny Python socket network
operations before test collection. This is a test guard, not an OS sandbox.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import subprocess
import sys
import time
import weakref
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

BACKEND_FILES = (
    "src/er_finder/config.py",
    "src/er_finder/models.py",
    "src/er_finder/agent/runner.py",
    "src/er_finder/agent/tools.py",
    "src/er_finder/agent/prompts.py",
    "src/er_finder/medical_api/client.py",
    "src/er_finder/medical_api/parser.py",
    "src/er_finder/medical_api/cache.py",
    "src/er_finder/http_retry.py",
    "src/er_finder/search/geocoder.py",
    "src/er_finder/search/distance.py",
    "src/er_finder/search/candidates.py",
    "src/er_finder/search/radius.py",
    "src/er_finder/search/service.py",
    "src/er_finder/safety.py",
    "src/er_finder/cli/renderer.py",
    "src/er_finder/guardrails/triage.py",
    "src/er_finder/guardrails/input_guard.py",
    "src/er_finder/guardrails/pii.py",
    "src/er_finder/guardrails/evidence.py",
    "src/er_finder/guardrails/middleware.py",
)
FIXTURE_FILES = (
    "tests/fixtures/egen/nearby_gangnam.xml",
    "tests/fixtures/egen/bed_status_gangnam.xml",
    "tests/fixtures/egen/severe_gangnam.xml",
    "tests/fixtures/egen/detail_A1100057.xml",
    "tests/fixtures/kakao/sample_response.json",
    "tests/fixtures/scenarios/sample_scenario.json",
)


_LOOPBACK_HOSTS = {"127.0.0.1", "::1"}
_local_listen_ports: set[int] = set()


def _deny_network(*args, **kwargs):
    raise RuntimeError("Offline verification blocks network access; use local fixtures.")


def _is_in_process_listener(address) -> bool:
    """True for loopback targets served by a socket this process is listening on."""
    return (
        isinstance(address, tuple)
        and len(address) >= 2
        and address[0] in _LOOPBACK_HOSTS
        and address[1] in _local_listen_ports
    )


def _guard_connect(original):
    """Allow only in-process loopback pairs (asyncio self-pipe on Windows), never remote hosts."""

    def guarded(self, address, *args, **kwargs):
        if _is_in_process_listener(address):
            return original(self, address, *args, **kwargs)
        return _deny_network()

    return guarded


def _record_listen(original):
    def listen(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            port = self.getsockname()[1]
            _local_listen_ports.add(port)
            # A listener dropped without close() must not keep its port trusted.
            weakref.finalize(self, _local_listen_ports.discard, port)
        except OSError:
            pass
        return result

    return listen


def _forget_on_close(original):
    """Stop trusting a port once its listening socket closes (accepted peers share the port)."""

    def close(self):
        try:
            port = self.getsockname()[1]
            try:
                self.getpeername()
            except OSError:  # unconnected, i.e. the listener itself
                _local_listen_ports.discard(port)
        except OSError:
            pass  # unbound or already closed
        return original(self)

    return close


def pytest_configure(config):
    """Prevent accidental Python DNS/TCP/UDP calls, including during collection."""
    _install_listen_recorder(config)
    for owner, name in (
        (socket, "getaddrinfo"),
        (socket, "gethostbyname"),
        (socket, "gethostbyname_ex"),
        (socket, "gethostbyaddr"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket.socket, "sendto"),
        (socket.socket, "sendmsg"),
    ):
        if not hasattr(owner, name):
            continue
        original = getattr(owner, name)
        guard = _guard_connect(original) if name in {"connect", "connect_ex"} else _deny_network
        setattr(owner, name, guard)
        config.add_cleanup(lambda o=owner, n=name, value=original: setattr(o, n, value))


def _install_listen_recorder(config) -> None:
    original = socket.socket.listen
    socket.socket.listen = _record_listen(original)
    config.add_cleanup(lambda: setattr(socket.socket, "listen", original))
    original_close = socket.socket.close
    socket.socket.close = _forget_on_close(original_close)
    config.add_cleanup(lambda: setattr(socket.socket, "close", original_close))
    config.add_cleanup(_local_listen_ports.clear)


def read_junit(path: Path) -> dict[str, int]:
    """Count testcase outcomes, including setup/collection errors, without recounting suites."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"JUnit report unavailable or malformed: {exc}") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit report has an unsupported root element")
    counts = dict.fromkeys(("total", "passed", "failed", "errors", "skipped"), 0)
    for case in root.iter("testcase"):
        counts["total"] += 1
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_stub(node: ast.AST) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return node.value.value is Ellipsis
    if isinstance(node, ast.Raise):
        exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return isinstance(exception, ast.Name) and exception.id == "NotImplementedError"
    return False


def find_blockers(
    root: Path,
    backend_files: tuple[str, ...] = BACKEND_FILES,
    fixture_files: tuple[str, ...] = FIXTURE_FILES,
) -> list[dict[str, str]]:
    """Static preflight only: populated files alone do not establish integration correctness."""
    blockers = []
    for relative in (*backend_files, *fixture_files):
        try:
            contents = (root / relative).read_text(encoding="utf-8")
            if not contents.strip():
                raise ValueError("empty file")
            if relative in backend_files:
                tree = ast.parse(contents)
                meaningful = [
                    node
                    for node in tree.body
                    if not _is_docstring(node)
                    and not isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                if not meaningful:
                    raise ValueError("module has no implementation")
                if all(_is_stub(node) for node in meaningful):
                    raise ValueError("module has only placeholder statements")
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.ClassDef)
                        and not node.bases
                        and not node.decorator_list
                    ):
                        body = [
                            statement for statement in node.body if not _is_docstring(statement)
                        ]
                        if not body or all(_is_stub(statement) for statement in body):
                            raise ValueError(f"class {node.name} has no implementation")
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = [
                            statement for statement in node.body if not _is_docstring(statement)
                        ]
                        if not body or all(_is_stub(statement) for statement in body):
                            raise ValueError(f"function {node.name} has no implementation")
            elif relative.endswith(".json"):
                if not json.loads(contents):
                    raise ValueError("fixture contains no records")
            else:
                ET.fromstring(contents)
        except (OSError, ValueError, SyntaxError, ET.ParseError) as exc:
            blockers.append({"path": relative, "reason": str(exc)})
    return blockers


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def run_verification(
    root: Path,
    output: Path,
    *,
    timeout: float,
    require_integration: bool,
    scope: str = "all",
) -> int:
    targets = {
        "all": (),
        "web": ("tests/unit/web", "tests/integration", "tests/unit/verification"),
    }[scope]
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "junit.xml"
    junit.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "scripts.verify",
        "-m",
        "not live",
        "--continue-on-collection-errors",
        f"--junitxml={junit}",
        *targets,
    ]
    # Do not forward API credentials, tracing flags, or ambient pytest options.
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "TMPDIR",
            "SYSTEMROOT",
            "LANG",
            "LC_ALL",
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "TEMP",
            "TMP",  # Windows home/temp
        )
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(root), str(root / "src"))),
            # Locale code pages (cp949) break reading UTF-8 test sources and decoding output.
            "PYTHONUTF8": "1",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    returncode, counts, error = None, None, None
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        returncode = result.returncode
        log = result.stdout + result.stderr
        try:
            counts = read_junit(junit)
        except ValueError as exc:
            error = str(exc)
        if returncode == 5:
            status = "NO_TESTS"
        elif returncode != 0:
            status = "FAIL"
        elif counts is None:
            status = "ERROR"
        elif counts["errors"] or counts["failed"]:
            status = "FAIL"
        elif counts["passed"] == 0:
            status = "NO_TESTS"
        else:
            status = "PASS"
    except subprocess.TimeoutExpired as exc:
        status, error = "TIMEOUT", f"pytest exceeded {timeout} seconds"
        log = _text(exc.stdout) + _text(exc.stderr)
    except OSError as exc:
        status, error, log = "ERROR", str(exc), str(exc)
    elapsed = time.perf_counter() - start
    blockers = find_blockers(root)
    dirty = _git_value(root, "status", "--porcelain")
    report = {
        "started_at": started_at,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "working_tree_dirty": None if dirty is None else bool(dirty),
        "python": sys.version.split()[0],
        "test_scope": scope,
        "provenance": {
            "scope": (
                "web UI and adapter contracts, verification tooling"
                if scope == "web"
                else "all offline module tests, API and runner UI contracts, verification tooling"
            ),
            "live_api_tests": "EXCLUDED",
            "network_policy": "Python socket DNS/TCP/UDP calls blocked in pytest",
            "clinical_validation": False,
            "performance_validation": "NOT_RUN; pytest duration is not recommendation latency",
        },
        "local_tests": {
            "status": status,
            "exit_code": returncode,
            "counts": counts,
            "duration_seconds": round(elapsed, 4),
            "command": command,
            "error": error,
        },
        "integration": {
            "status": "BLOCKED" if blockers else "NOT_VERIFIED",
            "blockers": blockers,
            "note": "Static preflight and local tests do not validate end-to-end scenarios.",
        },
    }
    (output / "pytest.log").write_text(log, encoding="utf-8")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if status != "PASS" and log:
        print(log, end="" if log.endswith("\n") else "\n")
    print(f"Scope: {scope}; local tests: {status}; integration: {report['integration']['status']}")
    print(f"Report: {output / 'report.json'}")
    if status != "PASS":
        return 1
    return 2 if require_integration else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verification-results"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--scope",
        choices=("all", "web"),
        default="all",
        help="Select all offline tests (default) or only web/adapter and verification tests.",
    )
    parser.add_argument(
        "--require-integration",
        action="store_true",
        help="Fail while integration is BLOCKED or NOT_VERIFIED, even when local tests pass.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return run_verification(
        Path(__file__).resolve().parents[1],
        args.output,
        timeout=args.timeout,
        require_integration=args.require_integration,
        scope=args.scope,
    )


if __name__ == "__main__":
    raise SystemExit(main())
