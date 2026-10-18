"""Live runner boundary. Importing or inspecting it never creates an agent or calls an API."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from er_finder.config import Settings
from er_finder.memory.store import ERFinderStore
from er_finder.memory.visit_plan import PendingVisitPlan
from er_finder.web.contracts import UIReply, WebResult

REQUIRED_KEYS = ("OPENAI_API_KEY", "KAKAO_REST_API_KEY", "EGEN_SERVICE_KEY")
AGENT_REQUIRED_KEYS = ("OPENAI_API_KEY", "EGEN_SERVICE_KEY")
_MESSAGES = {
    "not_connected": "실제 서비스에 아직 연결하지 않았습니다. 연결 확인을 실행해 주세요.",
    "keys": "필수 API 키가 설정되지 않았습니다. 설정 상태를 확인해 주세요.",
    "module": "에이전트 모듈을 불러오지 못했습니다. 백엔드 연결 준비가 필요합니다.",
    "contract": "에이전트와 의료 API의 연결 계약이 준비되지 않았습니다.",
    "runtime": "실제 요청을 완료하지 못했습니다. 연결 상태를 다시 확인해 주세요.",
    "approval": "저장 결과를 확인하지 못했습니다. 중복 방지를 위해 자동 재시도하지 않습니다.",
    "storage": "저장 정보 삭제를 확인하지 못했습니다. 삭제를 다시 시도해 주세요.",
    "ready": "에이전트가 연결되었습니다. 외부 API 응답은 실제 요청 시 확인합니다.",
}


class ConnectionUnavailable(RuntimeError):
    """Only fixed messages cross this boundary; exception details can contain credentials."""

    def __init__(self, category: str):
        self.category = category if category in _MESSAGES else "runtime"
        super().__init__(_MESSAGES[self.category])


@dataclass(frozen=True)
class ConnectionStatus:
    connected: bool
    checks: dict[str, bool]
    category: str
    message: str


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Read the project .env without replacing values supplied by the launching process."""
    path = dotenv_path if dotenv_path is not None else Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=path, override=False)


def credential_status() -> dict[str, bool]:
    load_environment()
    return {name: bool(os.environ.get(name, "").strip()) for name in REQUIRED_KEYS}


def default_runner_factory(**kwargs: Any) -> Any:
    """Construct the real model, classifier and runner only after explicit connection."""
    runner_module = importlib.import_module("er_finder.agent.runner")
    factory_module = importlib.import_module("er_finder.agent.factory")
    provider_module = importlib.import_module("er_finder.web.provider")
    settings = Settings()
    http_client = httpx.Client(timeout=settings.model_timeout)
    provider = None
    try:
        provider = provider_module.LiveProvider(kakao_key=os.environ.get("KAKAO_REST_API_KEY", ""))
        model, classifier = factory_module.build_models(settings, http_client)
        runner = runner_module.ERFinder(
            provider=provider, model=model, classifier=classifier, **kwargs
        )
        return _ManagedRunner(runner, provider, http_client)
    except Exception:
        try:
            if provider is not None:
                provider.close()
        finally:
            http_client.close()
        raise


class _ManagedRunner:
    """Close only explicitly owned resources; leave SDK private clients untouched."""

    def __init__(self, runner: Any, provider: Any, http_client: httpx.Client):
        self._runner, self._provider, self._http_client = runner, provider, http_client

    def chat(self, text: str, *, force_refresh: bool = False) -> Any:
        return self._runner.chat(text, force_refresh=force_refresh)

    def approve(self, approved: bool) -> Any:
        return self._runner.approve(approved)

    def end_session(self) -> None:
        try:
            self._runner.end_session()
        finally:
            self._provider.clear_cache()

    def close(self) -> None:
        try:
            self.end_session()
        finally:
            try:
                self._provider.close()
            finally:
                self._http_client.close()


class RunnerBackend:
    """Adapt ChatResult to UI-owned models and invalidate unsuccessful/past approvals.

    runner_factory receives user_id, transport and the same store used by profile.
    Injection exists for offline boundary tests and a future compatible app factory.
    """

    def __init__(
        self,
        *,
        runner_factory: Callable[..., Any] | None = None,
        user_id: str | None = None,
        store: BaseStore | None = None,
    ) -> None:
        self.user_id = user_id or str(uuid4())
        self._store = store if store is not None else InMemoryStore()
        self.profile = ERFinderStore(self._store, self.user_id)
        self._factory = runner_factory or default_runner_factory
        self._runner: Any = None
        self._transport = "car"
        self._category = "not_connected"
        self._reply: UIReply | None = None
        self._pending: PendingVisitPlan | None = None

    def status(self) -> ConnectionStatus:
        return ConnectionStatus(
            connected=self._runner is not None,
            checks=credential_status(),
            category=self._category,
            message=_MESSAGES[self._category],
        )

    def connect(self, *, transport: str = "car") -> ConnectionStatus:
        if transport not in {"car", "walk", "transit"}:
            self._invalidate("contract")
            return self.status()
        checks = credential_status()
        if not all(checks.get(key, False) for key in AGENT_REQUIRED_KEYS):
            self._invalidate("keys")
            return self.status()
        if self._runner is not None and transport == self._transport:
            return self.status()
        self._invalidate("not_connected")
        candidate = None
        try:
            candidate = self._factory(user_id=self.user_id, transport=transport, store=self._store)
            if not all(
                callable(getattr(candidate, name, None))
                for name in (
                    "chat",
                    "approve",
                    "end_session",
                )
            ):
                raise ConnectionUnavailable("contract")
        except ConnectionUnavailable as exc:
            self._category = exc.category
        except (ImportError, SyntaxError):
            self._category = "module"
        except (TypeError, AttributeError):
            self._category = "contract"
        except Exception:
            self._category = "runtime"
        else:
            self._runner = candidate
            self._transport = transport
            self._category = "ready"
        return self.status()

    def _clear_results(self) -> None:
        self._reply = None
        self._pending = None

    def _invalidate(self, category: str) -> None:
        runner, self._runner = self._runner, None
        self._clear_results()
        self._category = category
        if runner is not None:
            try:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
                else:
                    runner.end_session()
            except Exception:
                # Discard the runner even when checkpoint cleanup fails. Never
                # reuse an uncertain graph, especially after a write attempt.
                self._category = category

    def _failed(self, category: str) -> WebResult:
        self._invalidate(category)
        return WebResult(note=_MESSAGES[category])

    def _receive(self, result: Any, *, selected: str | None = None) -> WebResult:
        domain_reply = result.reply
        if not callable(getattr(domain_reply, "model_dump", None)):
            raise ConnectionUnavailable("contract")
        reply = UIReply.model_validate(domain_reply.model_dump())
        raw_pending = result.pending_approval
        pending = None
        if raw_pending is not None:
            if not isinstance(raw_pending, dict):
                raise ConnectionUnavailable("contract")
            hpid, name, summary = (
                raw_pending.get(key)
                for key in (
                    "hpid",
                    "name",
                    "symptom_summary",
                )
            )
            if not all(isinstance(value, str) and value.strip() for value in (hpid, name, summary)):
                raise ConnectionUnavailable("contract")
            hospitals = self._reply.hospitals if self._reply is not None else []
            hospital = next((item for item in hospitals if item.hpid == hpid), None)
            if hospital is None or hospital.name != name or selected != hpid:
                raise ConnectionUnavailable("contract")
            pending = PendingVisitPlan(hpid=hpid, name=name, symptom_summary=summary)
        note = getattr(result, "note", None)
        if note is not None and not isinstance(note, str):
            raise ConnectionUnavailable("contract")
        self._reply, self._pending = reply, pending
        return WebResult(reply=reply, pending_approval=pending, note=note)

    def chat(self, text: str, *, transport: str, force_refresh: bool = False) -> WebResult:
        if self._runner is None:
            self._clear_results()
            return WebResult(note=self.status().message)
        if transport != self._transport:
            if not self.connect(transport=transport).connected:
                return WebResult(note=self.status().message)
        if self._pending is not None:
            self.reset()
        if self._runner is None:
            return WebResult(note=self.status().message)
        self._pending = None
        try:
            return self._receive(self._runner.chat(text, force_refresh=force_refresh))
        except ConnectionUnavailable as exc:
            return self._failed(exc.category)
        except Exception:
            return self._failed("runtime")

    def select_hospital(self, hpid: str) -> WebResult:
        hospitals = self._reply.hospitals if self._reply is not None else []
        index = next((i for i, item in enumerate(hospitals) if item.hpid == hpid), None)
        if self._runner is None or index is None:
            return WebResult(reply=self._reply, note="현재 결과에 있는 후보만 선택할 수 있습니다.")
        if self._pending is not None:
            return WebResult(
                reply=self._reply,
                pending_approval=self._pending,
                note="현재 방문 계획의 승인 또는 거절을 먼저 완료해 주세요.",
            )
        try:
            return self._receive(self._runner.chat(f"{index + 1}번"), selected=hpid)
        except ConnectionUnavailable as exc:
            return self._failed(exc.category)
        except Exception:
            return self._failed("runtime")

    def approve(self, approved: bool) -> WebResult:
        if self._runner is None or self._pending is None:
            return WebResult(reply=self._reply, note="승인 대기 중인 방문 계획이 없습니다.")
        self._pending = None
        try:
            result = self._runner.approve(approved)
            if result.pending_approval is not None:
                return self._failed("approval")
            return self._receive(result)
        except Exception:
            return self._failed("approval")

    def reset(self) -> None:
        self._clear_results()
        if self._runner is not None:
            try:
                self._runner.end_session()
            except Exception:
                self._invalidate("runtime")

    def forget(self) -> None:
        had_runner = self._runner is not None
        self.reset()
        cleanup_failed = had_runner and self._runner is None
        try:
            for name in ("profile", "visits"):
                namespace = ("er_finder", self.user_id, name)
                while items := self._store.search(namespace, limit=100):
                    for item in items:
                        self._store.delete(item.namespace, item.key)
                    remaining = self._store.search(namespace, limit=100)
                    if {item.key for item in items} & {item.key for item in remaining}:
                        raise ConnectionUnavailable("storage")
                if self._store.search(namespace, limit=1):
                    raise ConnectionUnavailable("storage")
            if cleanup_failed:
                raise ConnectionUnavailable("storage")
        except Exception:
            self._invalidate("storage")
            raise ConnectionUnavailable("storage") from None
