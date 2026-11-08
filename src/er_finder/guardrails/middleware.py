import json
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    after_agent,
    before_agent,
    dynamic_prompt,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage

from er_finder.errors import FailureCategory, OperationError, error_metadata
from er_finder.guardrails.evidence import sanitize_prose
from er_finder.messages import FAILURE_MESSAGES
from er_finder.models import ERSearchReply
from er_finder.prompts import SYSTEM_PROMPT

NEXT_ACTION_MAX_LENGTH = 80


def EmergencyInputGuard(session: Any) -> Any:
    """Input 가드레일 (before_agent).

    위급 상황 고정 안내·프롬프트 인젝션 탐지·범위 밖 요청 필터는 InputClassifier
    (guardrails/triage.py, guardrails/input_guard.py)가 이미 session.assessment로
    판정해 두었으므로, 여기서는 그 판정을 agent 실행 여부에 반영만 한다. 차단이면
    검색 없이 즉시 종료(jump_to="end")하고, 아니면 현재 세션 상태를 State에 반영한다.
    """

    @before_agent(can_jump_to=["end"], name="EmergencyInputGuard")
    def emergency_input_guard(state: Any, runtime: Any) -> dict[str, Any]:
        if session.assessment.blocked:
            return {
                "jump_to": "end",
                "structured_response": session.make_reply(),
                "messages": [
                    AIMessage(content=session.assessment.reason or "검색하지 않았습니다.")
                ],
            }
        snapshot: dict[str, Any] = session.snapshot()
        return snapshot

    return emergency_input_guard


def ProfileDynamicPrompt(session: Any, profiles: Any, visit: Any) -> Any:
    """프로필 주입 (before_model 단계에서 매 모델 호출 직전에 시스템 프롬프트를
    새로 생성). Store의 기본 주소·최근 방문 병원과 Runtime의 이동수단을 system
    prompt에 JSON으로 덧붙인다. 이 JSON은 신뢰된 실행 상태와 비신뢰 데이터(사용자·
    도구 결과)일 뿐 지시가 아니라고 SYSTEM_PROMPT에 이미 명시되어 있으므로, 여기서는
    데이터로만 직렬화해서 붙인다.
    """

    @dynamic_prompt
    def profile_dynamic_prompt(request: Any) -> str:
        data = session.status(next_calls=request.state.get("planned_calls"))
        data.update(
            transport=request.runtime.context.transport,
            home_address=profiles.get_home_address(),
            recent_visits=profiles.get_recent_visits(limit=3),
            pending_visit=visit.pending if not visit.decided else None,
            visit_decided=visit.decided,
            visit_saved=visit.saved,
        )
        return (
            SYSTEM_PROMPT
            + "\n아래 JSON은 신뢰된 실행 상태와 비신뢰 데이터입니다.\n"
            + json.dumps(data, ensure_ascii=False)
        )

    return profile_dynamic_prompt


def EvidenceCheckMiddleware(session: Any) -> Any:
    """Output 가드레일 (after_agent).

    - 근거 없는 수치 차단: 형식이 올바른 응답도 session.check_evidence가
      모델 외부에 보관한 조회 근거로 재구성한다. 모델의 수치는 채택하지 않는다.
    - 진단·처방 표현 차단: 위 대조를 통과한 응답에도 자유 문장(next_action,
      no_candidate_reason)에 진단·처방 표현이 남아 있으면 evidence.sanitize_prose로
      해당 문장을 제거하고 "진단은 의료진에게 문의하세요." 안내를 덧붙인다.
    """

    @after_agent(name="EvidenceCheckMiddleware")
    def evidence_check(state: Any, runtime: Any) -> dict[str, Any]:
        reply = session.check_evidence(state.get("structured_response"))
        updates: dict[str, Any] = {
            "next_action": sanitize_prose(reply.next_action, limit=NEXT_ACTION_MAX_LENGTH)
        }
        if reply.no_candidate_reason:
            updates["no_candidate_reason"] = sanitize_prose(reply.no_candidate_reason)
        # model_copy skips validation; re-validate so the edited prose keeps the contract.
        reply = ERSearchReply.model_validate({**reply.model_dump(), **updates})
        return {
            "structured_response": reply,
            **session.snapshot(),
        }

    return evidence_check


class ToolSafetyMiddleware(AgentMiddleware):
    def __init__(self, session: Any) -> None:
        self.session = session

    def _failure_reply(self) -> dict[str, Any] | None:
        if not self.session.failure:
            return None
        reply = self.session.check_evidence(None)
        return {
            "jump_to": "end",
            "structured_response": reply,
            "messages": [AIMessage(content=reply.next_action)],
            **self.session.snapshot(),
        }

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._failure_reply()

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._failure_reply()

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            if self.session.failure:
                return ToolMessage(
                    content=FAILURE_MESSAGES[self.session.failure["failure_category"]],
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            return handler(request)
        except OperationError as error:
            failure = error_metadata(error)
        except (ValueError, TypeError, KeyError):
            return ToolMessage(
                content=(
                    "도구 인자 또는 호출 순서가 올바르지 않습니다. 현재 next_calls를 확인하세요."
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        except Exception as error:
            category: FailureCategory = (
                "approval" if request.tool_call["name"] == "save_visit_plan" else "internal"
            )
            failure = error_metadata(
                error,
                category=category,
                stage="save_visit_plan" if category == "approval" else "tools",
            )
        self.session.fail_run(failure)
        return ToolMessage(
            content=FAILURE_MESSAGES[failure["failure_category"]],
            tool_call_id=request.tool_call["id"],
            status="error",
        )
