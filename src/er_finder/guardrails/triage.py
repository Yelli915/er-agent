from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from er_finder.guardrails.input_guard import CALL_119_NOTICE, InputAssessment, assess_input
from er_finder.guardrails.pii import mask_pii
from er_finder.models import Condition, TriageAssessment
from er_finder.prompts import CLASSIFIER_PROMPT
from er_finder.usage import ClassifierUsage, RunUsage


# LLM 출력 구조 정의 (Pydantic 모델)
class ClassifierOutput(BaseModel):
    """
    LLM이 생성할 구조화된 분류(Classifier) 출력 데이터 모델
    """

    model_config = ConfigDict(extra="forbid")
    severity: Literal["critical", "urgent", "standard"]
    condition: Condition | None = None
    injection: bool
    confidence: float = Field(ge=0, le=1)


class InputClassifier:
    """
    사용자의 입력 텍스트를 정규식 기반 룰셋과 LLM을 조합하여 종합 평가하는 클래스
    """

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self.last_usage = ClassifierUsage()

    def assess(self, text: str) -> InputAssessment:
        """
        사용자 입력 문장을 분석하여 중증도 분류 및 보안 차단 여부를 최종 판정
        """
        self.last_usage = ClassifierUsage()
        masked = mask_pii(text)
        result = assess_input(masked)
        if (
            self.model is None
            or result.blocked
            or result.greeting
            or (result.triage.severity == "critical" and not result.suspicious)
        ):
            return result
        usage, started = RunUsage(), perf_counter()
        try:
            output = self.model.with_structured_output(ClassifierOutput).invoke(
                [("system", CLASSIFIER_PROMPT), ("human", masked)],
                config={"callbacks": [usage]},
            )
            assessment = TriageAssessment.model_validate(output.model_dump())
            ranks = {"standard": 0, "urgent": 1, "critical": 2}
            if ranks[assessment.severity] >= ranks[result.triage.severity]:
                if result.triage.condition is not None:
                    assessment.condition = result.triage.condition
                result.triage = assessment
            elif assessment.condition and result.triage.condition is None:
                result.triage.condition = assessment.condition
            if assessment.injection:
                result.blocked = True
                result.reason = (
                    "서비스 규칙 변경 요청은 처리할 수 없습니다. 위치와 증상을 알려주세요."
                )
                if result.triage.severity == "critical":
                    result.reason = f"{result.reason} {CALL_119_NOTICE}"
        except Exception:
            # No exception body is logged: SDK errors can contain request data or keys.
            if result.suspicious:
                result.blocked = True
                result.reason = "입력 안전성을 확인하지 못했습니다. 위치와 증상만 다시 알려주세요."
        finally:
            self.last_usage = ClassifierUsage(
                calls=1,
                elapsed_seconds=max(0.0, perf_counter() - started),
                tokens_complete=usage.known_responses == usage.responses > 0,
                **(usage.tokens if usage.known_responses else {}),
            )
        return result
