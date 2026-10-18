from types import SimpleNamespace


def test_renderer_handles_missing_fields_without_claiming_availability():
    from er_finder.agent.renderer import render_reply

    reply = SimpleNamespace(
        call_119_first=False,
        hospitals=[],
        no_candidate_reason="확인된 후보 없음",
        next_action="전화 확인",
        disclaimer="상황은 변할 수 있습니다",
    )
    text = render_reply(reply, note="조회 결과", pending=None)
    assert "확인된 후보 없음" in text
    assert "조회 결과" in text
    assert "전화 확인" in text
