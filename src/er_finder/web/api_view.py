"""Explicit E-Gen location lookup, separate from agent recommendations."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from er_finder.web.live import credential_status


def render_api_view() -> None:
    """Query the existing medical client only on submit; retain the last result per session."""
    st.subheader("주변 응급의료기관 찾기")
    st.caption("E-Gen 위치정보 원자료 · 반경 내 기관을 거리순으로 확인하고 상세 정보를 조회하세요.")
    st.info(
        "기관 주소와 대표전화를 확인할 수 있습니다. 가용 병상과 진료 가능 여부는 전화로 확인하세요."
    )
    configured = credential_status()["EGEN_SERVICE_KEY"]
    if not configured:
        st.warning("EGEN_SERVICE_KEY 설정이 필요합니다. 서버의 환경변수 또는 .env를 확인하세요.")

    with st.form("api_location_form"):
        latitude_column, longitude_column, radius_column = st.columns((2, 2, 1))
        lat = latitude_column.number_input(
            "위도",
            min_value=-90.0,
            max_value=90.0,
            value=37.497942,
            format="%.6f",
            key="api_lat",
            disabled=not configured,
        )
        lon = longitude_column.number_input(
            "경도",
            min_value=-180.0,
            max_value=180.0,
            value=127.027621,
            format="%.6f",
            key="api_lon",
            disabled=not configured,
        )
        radius = radius_column.selectbox(
            "반경",
            (5, 10, 20, 30),
            format_func=lambda value: f"{value} km",
            key="api_radius",
            disabled=not configured,
        )
        submitted = st.form_submit_button(
            "주변 기관 조회",
            key="api_search",
            type="primary",
            width="stretch",
            disabled=not configured,
        )

    if submitted and configured:
        result: Any = {
            "lat": lat,
            "lon": lon,
            "radius": radius,
            "queried_at": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"),
            "rows": [],
            "error": False,
        }
        try:
            with st.spinner("E-Gen 기관 정보를 조회하고 있습니다…"):
                from er_finder.medical_api.client import list_nearby_ers

                rows = list_nearby_ers(lat, lon, radius, raise_on_error=True)
                result["rows"] = [
                    {
                        "기관 ID": row.get("hpid") or "확인 불가",
                        "기관명": row.get("name") or "확인 불가",
                        "거리 (km)": row.get("distance_km"),
                        "주소": row.get("address") or "확인 불가",
                        "대표전화": row.get("er_tel") or "확인 불가",
                    }
                    for row in rows
                ]
        except Exception:
            # Import/configuration errors can include credentials and request URLs.
            result["error"] = True
        st.session_state["api_lookup_result"] = result

    result = st.session_state.get("api_lookup_result")
    if result is None:
        st.caption("조회 버튼을 누르면 입력한 좌표 주변의 기관 목록이 표시됩니다.")
        return

    st.caption(
        f"마지막 조회 · 위도 {result['lat']:.6f}, 경도 {result['lon']:.6f} · "
        f"반경 {result['radius']} km · {result['queried_at']}"
    )
    if result["error"]:
        st.error("기관 정보를 조회하지 못했습니다. 설정과 연결 상태를 확인한 후 다시 조회하세요.")
    elif not result["rows"]:
        st.info(
            "API가 빈 응답을 반환했습니다. 선택한 반경의 기관 자료를 확인할 수 없습니다. "
            "반경을 조정해 다시 조회하세요."
        )
    else:
        st.caption(f"조회 결과 · {len(result['rows'])}곳 · 거리순")
        st.dataframe(
            result["rows"],
            hide_index=True,
            width="stretch",
            column_config={"거리 (km)": st.column_config.NumberColumn(format="%.2f")},
        )
        st.caption("주소 등 응답에 없는 항목은 ‘확인 불가’로 표시합니다.")
        _render_detail(result, configured=configured)


def _render_detail(result: dict[str, Any], *, configured: bool) -> None:
    hospitals = {
        row["기관 ID"]: row["기관명"] for row in result["rows"] if row["기관 ID"] != "확인 불가"
    }
    if not hospitals:
        return

    st.subheader("기관 상세")
    selected = st.selectbox(
        "상세 정보를 확인할 기관",
        list(hospitals),
        format_func=lambda hpid: f"{hospitals[hpid]} · {hpid}",
        key="api_detail_hpid",
        disabled=not configured,
    )
    if st.button("상세 조회", key="api_detail_search", disabled=not configured):
        if configured and selected in hospitals:
            detail: Any = {"hpid": selected, "data": {}, "error": False}
            try:
                with st.spinner("기관 상세 정보를 조회하고 있습니다…"):
                    from er_finder.medical_api.client import get_er_detail

                    detail["data"] = get_er_detail(selected, raise_on_error=True)
            except Exception:
                detail["error"] = True
            result["detail"] = detail

    detail = result.get("detail")
    if detail is None or detail["hpid"] != selected:
        st.caption("기관을 선택하고 상세 조회를 누르면 주소와 대표전화가 표시됩니다.")
        return
    if detail["error"]:
        st.error("상세 조회에 실패했습니다. 연결 상태를 확인하고 다시 조회하세요.")
        return
    if not detail["data"]:
        st.info("API가 빈 응답을 반환했습니다. 선택한 기관의 상세 자료가 없습니다.")
        return

    data = detail["data"]
    with st.container(border=True):
        st.text(data.get("name") or hospitals[selected])
        st.caption("기관 ID")
        st.text(selected)
        st.caption("주소")
        st.text(data.get("address") or "확인 불가")
        st.caption("대표전화")
        st.text(data.get("main_tel") or data.get("er_tel") or "확인 불가")
