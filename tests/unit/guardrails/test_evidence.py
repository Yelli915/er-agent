from er_finder.guardrails.evidence import DIAGNOSIS_NOTICE, sanitize_prose


def test_clean_prose_is_unchanged():
    text = "가까운 응급실에 전화로 확인하세요."
    assert sanitize_prose(text, limit=80) == text


def test_diagnosis_sentence_is_removed_and_notice_added():
    result = sanitize_prose("전화로 확인하세요. 심근경색입니다.")
    assert result == f"전화로 확인하세요. {DIAGNOSIS_NOTICE}"


def test_limit_drops_trailing_sentences_so_the_notice_still_fits():
    text = "첫 번째 안내 문장입니다. " + "두 번째 안내 문장이 아주 깁니다. " * 3 + "이 약을 드세요."
    result = sanitize_prose(text, limit=40)
    assert len(result) <= 40
    assert result.endswith(DIAGNOSIS_NOTICE)


def test_limit_leaves_only_the_notice_when_nothing_else_fits():
    result = sanitize_prose("아주 긴 문장입니다 " * 10 + "이 약을 드세요.", limit=20)
    assert result == DIAGNOSIS_NOTICE
