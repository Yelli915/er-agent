"""Real API lookup and agent conversation, with private browser session state."""

from collections.abc import Callable
from typing import Any

import streamlit as st

from er_finder.guardrails.input_guard import assess_input
from er_finder.models import EMERGENCY
from er_finder.web.api_view import render_api_view
from er_finder.web.components import page_style, render_header, render_pending, render_reply
from er_finder.web.live import load_environment
from er_finder.web.state import WebSession

DEFAULT_LATITUDE = 37.497942
DEFAULT_LONGITUDE = 127.027621
NOTICE_KEY, ERROR_KEY = "ui_notice", "ui_error"
TRANSPORT_LABELS = {"car": "자동차", "walk": "도보", "transit": "대중교통"}
KEY_LABELS = {
    "OPENAI_API_KEY": "OpenAI · 대화와 분류",
    "KAKAO_REST_API_KEY": "카카오 · 위치 검색",
    "EGEN_SERVICE_KEY": "E-Gen · 응급의료 정보",
}


def _clear_feedback() -> None:
    st.session_state[NOTICE_KEY] = st.session_state[ERROR_KEY] = None


def _reset_home_inputs() -> None:
    st.session_state["home_address"] = ""
    st.session_state["home_consent"] = False


def _perform(action: Callable[[], None], notice: str | None = None) -> None:
    _clear_feedback()
    try:
        action()
        st.session_state[NOTICE_KEY] = notice
    except Exception:
        st.session_state[ERROR_KEY] = "요청을 처리하지 못했습니다. 연결 상태를 확인해 주세요."


def _sync_context(web: WebSession) -> bool:
    transport = st.session_state.get("transport", web.transport)
    changed = transport != web.transport
    web.set_transport(transport)
    location = (
        (
            st.session_state.get("chat_lat", DEFAULT_LATITUDE),
            st.session_state.get("chat_lon", DEFAULT_LONGITUDE),
        )
        if st.session_state.get("use_coordinates", False)
        else None
    )
    changed = changed or location != web.location
    web.set_location(location)
    if changed:
        _clear_feedback()
    return changed


def _perform_current(
    web: WebSession, action: Callable[[], None], *, allow_changed: bool = False
) -> None:
    def current_action() -> None:
        changed = _sync_context(web)
        if changed and not allow_changed:
            return
        if web.backend.status().connected:
            action()

    _perform(current_action)


def _save_home(web: WebSession) -> None:
    _perform(
        lambda: web.set_home_address(
            st.session_state["home_address"], consent=st.session_state["home_consent"]
        )
    )


def _clear_home(web: WebSession) -> None:
    _perform(web.clear_home_address, "기본 주소를 삭제했습니다.")
    _reset_home_inputs()


def _forget(web: WebSession) -> None:
    def forget_all() -> None:
        web.forget()
        for key in list(st.session_state):
            if isinstance(key, str) and key.startswith("api_"):
                del st.session_state[key]

    _perform(forget_all, "대화·저장 정보와 API 조회 기록을 모두 삭제했습니다.")
    _reset_home_inputs()
    st.session_state["use_coordinates"] = False
    st.session_state["chat_lat"] = DEFAULT_LATITUDE
    st.session_state["chat_lon"] = DEFAULT_LONGITUDE


def _connect(web: WebSession) -> None:
    def connect() -> None:
        _sync_context(web)
        web.reset()
        web.backend.connect(transport=web.transport)

    _perform(connect)


def _sidebar(web: WebSession) -> None:
    with st.sidebar:
        st.markdown("## ✚ ER Finder")
        st.caption("실제 응급의료 정보 조회")
        st.divider()
        _location_controls(web)
        status = _connection_controls(web)
        st.divider()
        _conversation_controls(web, status)
        _home_address_panel(web)
        _visit_plan_panel(web)
        st.caption("저장 정보는 이 브라우저 세션의 서버 메모리에 보관됩니다.")


def _location_controls(web: WebSession) -> None:
    st.selectbox(
        "이동수단",
        list(TRANSPORT_LABELS),
        format_func=lambda value: TRANSPORT_LABELS[value],
        key="transport",
    )
    with st.expander("상담 위치 직접 지정", expanded=True):
        st.checkbox("주소 검색 없이 좌표로 조회", key="use_coordinates")
        st.number_input(
            "상담 위도",
            min_value=-90.0,
            max_value=90.0,
            value=DEFAULT_LATITUDE,
            format="%.6f",
            key="chat_lat",
            disabled=not st.session_state["use_coordinates"],
        )
        st.number_input(
            "상담 경도",
            min_value=-180.0,
            max_value=180.0,
            value=DEFAULT_LONGITUDE,
            format="%.6f",
            key="chat_lon",
            disabled=not st.session_state["use_coordinates"],
        )
        st.caption("입력된 좌표 주변을 조회합니다. 현재 위치와 일치하는지 확인해 주세요.")
    _sync_context(web)


def _connection_controls(web: WebSession) -> Any:
    status = web.backend.status()
    st.caption("● 에이전트 연결됨" if status.connected else "○ 에이전트 연결 확인 필요")
    st.button(
        "다시 연결 확인" if status.connected else "에이전트 연결 확인",
        key="connect_runner",
        width="stretch",
        type="primary",
        on_click=_connect,
        args=(web,),
    )
    st.caption("의료기관 직접 조회는 에이전트 연결 없이 사용할 수 있습니다.")
    return status


def _conversation_controls(web: WebSession, status: Any) -> None:
    left, right = st.columns(2)
    left.button(
        "새 대화",
        key="new_conversation",
        width="stretch",
        on_click=_perform,
        args=(web.reset, "대화와 승인 대기를 초기화했습니다. 저장 정보는 유지됩니다."),
    )
    right.button(
        "기록 전체 삭제",
        key="forget_all",
        width="stretch",
        on_click=_forget,
        args=(web,),
    )
    pending = bool(web.result and web.result.pending_approval)
    st.button(
        "상담 다시 조회",
        key="refresh_search",
        width="stretch",
        disabled=not status.connected or web.result is None or pending,
        on_click=_perform_current,
        args=(web, web.refresh),
    )


def _home_address_panel(web: WebSession) -> None:
    with st.expander("기본 주소"):
        home = web.backend.profile.get_home_address()
        if home:
            st.caption("저장된 주소")
            st.text(home)
        with st.form("home_form"):
            st.text_input("기본 주소 입력", key="home_address", max_chars=150)
            st.checkbox("이 브라우저 세션에 기본 주소 저장에 동의합니다.", key="home_consent")
            st.form_submit_button(
                "주소 저장",
                key="save_home",
                on_click=_save_home,
                args=(web,),
                width="stretch",
            )
        st.button(
            "기본 주소 삭제",
            key="clear_home",
            on_click=_clear_home,
            args=(web,),
            disabled=home is None,
            width="stretch",
        )


def _visit_plan_panel(web: WebSession) -> None:
    with st.expander("저장한 방문 계획"):
        visits = web.backend.profile.get_recent_visits()
        if not visits:
            st.caption("아직 저장한 방문 계획이 없습니다.")
        for visit in visits:
            st.text(visit["name"])
            st.caption(visit["saved_at"])
        st.caption("방문 계획 저장은 병원 예약이나 접수가 아닙니다.")


def _connection_view(web: WebSession) -> None:
    status = web.backend.status()
    st.subheader("서비스 연결 상태")
    st.caption("키의 설정 여부입니다. 인증 성공 여부는 실제 요청 이후에 확인됩니다.")
    for column, (key, label) in zip(st.columns(3), KEY_LABELS.items(), strict=True):
        with column:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.write("설정됨" if status.checks.get(key) else "설정 필요")
    (st.success if status.connected else st.info)(status.message)
    with st.expander("로컬 실행 설정"):
        st.markdown(
            "프로젝트의 `.env`에 키를 설정하세요. 키를 바꿨다면 서버를 재시작한 뒤 "
            "**에이전트 연결 확인**을 눌러주세요. `.env.example`에는 실제 키를 넣지 않습니다."
        )
        st.code(
            "OPENAI_API_KEY=...\nKAKAO_REST_API_KEY=...\nEGEN_SERVICE_KEY=...", language="dotenv"
        )
    st.markdown("**기능별 연결**")
    st.markdown(
        "- 의료기관 조회: E-Gen의 실제 주변 기관과 상세 정보를 조회합니다.\n"
        "- 에이전트 상담: 위치·증상 입력, 후보 확인, 방문 계획 승인 흐름을 연결합니다.\n"
        "- 기록 관리: 기본 주소 동의, 새 대화, 방문 계획 조회, 전체 삭제를 제공합니다."
    )
    st.caption("팀 모듈 연결에 실패하면 원인을 안내하며 임의의 병원 데이터를 만들지 않습니다.")


def _conversation(web: WebSession) -> None:
    status = web.backend.status()
    if not status.connected:
        st.info(status.message)
        st.caption("에이전트 연결 전에도 ‘의료기관 조회’에서 E-Gen 정보를 확인할 수 있습니다.")
    else:
        st.success("증상을 입력해 주세요." if web.location else "위치와 증상을 입력해 주세요.")
        if web.location:
            st.caption(f"상담 위치 · 위도 {web.location[0]:.6f}, 경도 {web.location[1]:.6f}")
        else:
            st.caption("카카오 주소 검색을 사용할 수 없다면 왼쪽에서 좌표 직접 조회를 선택하세요.")
    if web.messages:
        with st.expander("대화 기록", expanded=True):
            for message in web.messages:
                with st.chat_message(message.role):
                    st.text(message.text)
    result = web.result
    if result:
        if result.note:
            st.info(result.note)
        if result.reply:
            render_reply(
                result.reply,
                lambda hpid: _perform_current(web, lambda: web.select_hospital(hpid)),
                selection_disabled=result.pending_approval is not None or not status.connected,
            )
        if result.pending_approval:
            render_pending(
                result.pending_approval,
                lambda approved: _perform_current(web, lambda: web.decide(approved)),
            )
    else:
        with st.container(border=True):
            st.subheader("위치와 상황을 알려주세요")
            st.write("현재 계신 동네나 주소, 불편한 증상을 함께 입력해 주세요.")
            for column, title, detail in zip(
                st.columns(3),
                ("01  위치와 증상", "02  응급실 후보", "03  방문 계획"),
                ("현재 상황 입력", "거리·병상·수용 정보 확인", "직접 선택하고 저장 승인"),
                strict=True,
            ):
                column.markdown(f"**{title}**")
                column.caption(detail)
    text = st.chat_input(
        "예: 강남구 역삼동, 손가락이 부었어요",
        key="message",
        max_chars=2000,
        disabled=not status.connected or bool(result and result.pending_approval),
    )
    if text:
        if assess_input(text).triage.severity == "critical":
            st.error(EMERGENCY)
        with st.status("응급실 정보를 확인하고 있습니다…", expanded=True) as progress:
            set_progress = getattr(web.backend, "set_progress", None)
            if callable(set_progress):
                set_progress(lambda line: progress.write(f"✓ {line}"))
            try:
                _perform_current(web, lambda: web.submit(text), allow_changed=True)
            finally:
                if callable(set_progress):
                    set_progress(None)
            progress.update(label="확인을 마쳤습니다", state="complete", expanded=False)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ER Finder · 응급실 안내", page_icon="✚", layout="wide")
    page_style()
    load_environment()
    if "web" not in st.session_state or not hasattr(st.session_state["web"].backend, "status"):
        st.session_state["web"] = WebSession()
    web: WebSession = st.session_state["web"]
    _sidebar(web)
    render_header()
    if st.session_state.get(ERROR_KEY):
        st.error(st.session_state[ERROR_KEY])
    elif st.session_state.get(NOTICE_KEY):
        st.success(st.session_state[NOTICE_KEY])
    api, conversation, connection = st.tabs(["의료기관 조회", "에이전트 상담", "연결 상태"])
    with api:
        render_api_view()
    with conversation:
        _conversation(web)
    with connection:
        _connection_view(web)
    st.divider()
    st.caption(
        "생명이 위급한 상황에서는 즉시 119에 연락하세요. 응급실 방문 전 전화 확인을 권장합니다."
    )


if __name__ == "__main__":
    main()
