import re
from dataclasses import dataclass

from er_finder.guardrails.pii import normalize
from er_finder.models import Condition, Severity, TriageAssessment

# 설계서 부록 A. 키워드 사전 (어간 형태, 공백 제거 후 부분 문자열 일치)
# 설계서 기준 40개에 골든셋 실측으로 확인된 과소 분류 표현을 보강했다(호흡 +3,
# 심장 +1, 출혈 +2, 신경 +2). 오탐은 과잉 안내(119)로 이어지므로 구체적 표현만 추가한다.
CRITICAL_PHRASES = (
    # 의식·반응 (7)
    "의식없",
    "의식이없",
    "쓰러졌",
    "깨어나지않",
    "반응이없",
    "실신",
    "기절",
    # 호흡 (8)
    "숨을못",
    "숨쉬기힘들",
    "숨이안",
    "호흡곤란",
    "호흡이멈",
    "숨을안쉬",
    "입술이파랗",
    "입술이파래",
    "숨이막",
    "호흡이가빠",
    "질식",
    # 심장·흉통 (5)
    "심한흉통",
    "가슴을쥐어짜",
    "가슴통증이심",
    "심정지",
    "맥박이없",
    "맥박이안잡",
    # 출혈 (6)
    "피가멈추지않",
    "대량출혈",
    "피를많이",
    "피가계속",
    "토혈",
    "피를토",
    "손가락이잘려",
    "팔다리가잘려",
    # 신경 (7)
    "경련",
    "발작",
    "마비",
    "한쪽팔에힘",
    "말이어눌",
    "얼굴이한쪽으로",
    "극심한두통",
    "힘이갑자기빠",
    "말도잘안나",
    # 외상·중독 (7)
    "추락",
    "교통사고",
    "심한화상",
    "감전",
    "물에빠",
    "음독",
    "약을많이먹",
    "과다복용",
    "한움큼먹",
    "한줌먹",
    "약을다먹",
    "약을한통",
    "한통다먹",
    "한통을다먹",
    "연탄가스",
    "일산화탄소",
    "층에서떨어",
    "높은곳에서떨어",
    "일어나지못",
    "진통제를많이",
    "수면제를많이",
)

# 프롬프트 인젝션(보안 우회 공격) 감지용 정규식
# - 시스템 지시 무시, 시스템 프롬프트 출력 요청, API 키 유출 시도 등을 탐지
INJECTION = re.compile(
    r"(?:이전|기존|모든|위의|안전|시스템).{0,18}(?:지시|규칙|프롬프트).{0,12}(?:무시|잊|출력|보여|공개|변경)"
    r"|(?:시스템|개발자)\s*프롬프트.{0,15}(?:출력|보여|공개)"
    r"|ignore.{0,30}(?:instructions|rules)|(?:reveal|print).{0,30}system\s*prompt"
    r"|(?:규칙|지시|프롬프트).{0,12}(?:전부|모두|다)\s*(?:잊|무시)"
    r"|(?:관리자|개발자|admin|developer)\s*(?:모드|mode)"
    r"|(?:API|인증|비밀)\s*(?:키|key).{0,12}(?:보여|출력|공개)",
    re.I | re.S,
)

# 의심스러운 의도 탐지용 정규식 (인젝션 공격의 징후 탐지)
SUSPICIOUS = re.compile(
    r"프롬프트|역할.{0,8}(?:바꿔|변경)|지시|jailbreak|system|developer|ignore", re.I
)

# 서비스 범위 외(Out of Scope) 질문 감지용 정규식
# - 진단/처방 요청, 잡담, 관련 없는 주제(주식, 날씨 등)를 차단
OUT_OF_SCOPE = re.compile(
    r"무슨\s*(?:병|약)|어떤\s*약|약.{0,8}(?:먹어야|먹어도|먹을까|복용해도|복용량|추천)"
    r"|처방해|진단해|치료법|주식|코인|날씨|시\s*써줘|타이레놀|몇\s*알"
)

URGENT_PHRASES = (
    "가슴",
    "흉통",
    "식은땀",
    "고열",
    "화상",
    "진통",
    "출산",
    "교통사고",
    "부러",
    "골절",
    "축처",
    "한쪽눈",
    "퉁퉁",
    "모양이이상",
    "데어",
    "데었",
    "데였",
    "물집",
    "양수",
    "오토바이",
)

# One alternation scans the compacted text once instead of once per phrase.
_TRAUMA = ("교통사고", "중증외상", "추락", "오토바이", "떨어", "일어나지못")
_CRITICAL = re.compile("|".join(map(re.escape, CRITICAL_PHRASES)))
_URGENT = re.compile("|".join(map(re.escape, URGENT_PHRASES)))
_WHITESPACE = re.compile(r"\s")
# 증상이 아닌 일반 명사: urgent·조건 판정에서만 제외한다(critical 사전은 전체 문장 기준).
_NON_SYMPTOM = re.compile("진통제|화상회의|화상통화|화상채팅|화상면접")
CALL_119_NOTICE = "위급하다면 지금 바로 119에 신고하세요."
_GREETING = re.compile(r"\s*(?:안녕(?:하세요)?|안녕하세요|반가워요|hello|hi)[!?.\s]*", re.I)


@dataclass
class InputAssessment:
    triage: TriageAssessment
    blocked: bool = False
    reason: str | None = None
    greeting: bool = False
    suspicious: bool = False


def assess_input(text: str) -> InputAssessment:
    text = normalize(text)
    compact = _WHITESPACE.sub("", text)
    critical = _CRITICAL.search(compact) is not None
    symptoms = _NON_SYMPTOM.sub("", compact)
    urgent = _URGENT.search(symptoms) is not None
    severity: Severity = "critical" if critical else "urgent" if urgent else "standard"
    condition: Condition | None = None
    if severity != "standard":
        # 외상·화상은 가슴 부위 통증과 함께 나올 수 있으므로 심근경색보다 먼저 판정한다.
        if any(p in symptoms for p in ("말이어눌", "한쪽팔에힘", "얼굴이한쪽으로")):
            condition = "뇌졸중"
        elif any(p in symptoms for p in ("극심한두통", "벼락두통")):
            condition = "뇌출혈"
        elif any(p in symptoms for p in ("힘이갑자기빠", "말도잘안나")):
            condition = "뇌졸중"
        elif any(p in symptoms for p in _TRAUMA):
            condition = "중증외상"
        elif any(p in symptoms for p in ("화상", "데어", "데었", "데였", "물집")):
            condition = "화상"
        elif any(p in symptoms for p in ("가슴", "흉통", "식은땀")):
            condition = "심근경색"
        elif any(p in symptoms for p in ("진통", "출산", "분만", "양수")):
            condition = "분만"

    injection = bool(INJECTION.search(text))
    out = bool(OUT_OF_SCOPE.search(text)) and not critical
    reason = None
    if injection:
        reason = (
            "서비스 규칙 변경 요청은 처리할 수 없습니다. 응급실을 찾을 위치와 증상을 알려주세요."
        )
    elif out:
        reason = (
            "진단·처방·복약 상담은 제공하지 않습니다. 응급실을 찾으려면 현재 위치를 알려주세요."
        )
    if reason and critical:
        reason = f"{reason} {CALL_119_NOTICE}"
    greeting = _GREETING.fullmatch(text) is not None
    return InputAssessment(
        TriageAssessment(
            severity=severity, condition=condition, injection=injection, confidence=0.95
        ),
        blocked=injection or out,
        reason=reason,
        greeting=greeting,
        suspicious=bool(SUSPICIOUS.search(text)),
    )
