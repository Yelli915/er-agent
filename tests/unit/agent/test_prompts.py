"""프롬프트 문자열이 models.py의 스키마와 어긋나지 않는지 확인한다.

실제 값은 하드코딩하지 않고 models.py에서 그대로 읽어와 비교한다.
프롬프트 텍스트나 스키마 중 한쪽만 바뀌었을 때 바로 잡아내기 위한 회귀 테스트다.
"""

from er_finder.models import Condition, ERSearchReply
from er_finder.prompts import CLASSIFIER_PROMPT, SYSTEM_PROMPT


def test_system_prompt_contains_the_fixed_disclaimer_sentence():
    (disclaimer,) = ERSearchReply.model_fields["disclaimer"].annotation.__args__
    assert disclaimer in SYSTEM_PROMPT


def test_classifier_prompt_mentions_every_condition_value():
    for condition in Condition.__args__:
        assert condition in CLASSIFIER_PROMPT


def test_system_prompt_mentions_119_for_critical_cases():
    assert "119" in SYSTEM_PROMPT
