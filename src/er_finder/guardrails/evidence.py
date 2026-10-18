import re

from er_finder.guardrails.input_guard import INJECTION
from er_finder.guardrails.pii import normalize

# 의료 진단/처방성 문장 감지용 정규식
MEDICAL_SENTENCE = re.compile(
    r"(?:심근경색|뇌출혈|뇌졸중|감기|암|골절|질환|[가-힣]+병)(?:입니다|이에요|이네요|으로\s*진단)"
    r"|(?:드세요|복용하|투여하|처방하|치료하세요|먹으세요)"
    r"|(?:진단은|진단을).{0,10}(?:확실|확정)",
    re.I,
)


# LLM 생성 텍스트 정화 함수
def sanitize_prose(text: str) -> str:
    """
    LLM이 생성한 응답 문장에서 확정적 진단이나 처방 성격의 문장을 제거하고,
    제거된 문장이 있을 경우 안내 면책 문구를 추가
    """
    sentences = re.split(r"(?<=[.!?。])\s*|\n", text)
    safe = [s for s in sentences if s and not MEDICAL_SENTENCE.search(s)]
    if len(safe) != len([s for s in sentences if s]):
        safe.append("진단은 의료진에게 문의하세요.")
    return " ".join(safe)


# 외부 제공 데이터(병원 정보 등) 안전 검증 함수
def safe_data(value: object, default: str = "확인 불가") -> str:
    """
    들어온 문자열 데이터를 검증 및 정화
    프롬프트 인젝션, 위험 문장, 과도하게 긴 데이터 등을 제거하고 안전하게 반환
    """
    if not isinstance(value, str) or not value.strip():
        return default
    value = normalize(value)
    if INJECTION.search(value) or MEDICAL_SENTENCE.search(value) or len(value) > 250:
        return default
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value).strip()
