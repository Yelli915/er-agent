"""Exercise actual nearby parsing through Streamlit; no external requests are allowed."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = "from er_finder.web.api_view import render_api_view\nrender_api_view()"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "egen"


def open_view():
    app = AppTest.from_string(APP, default_timeout=10).run()
    assert not app.exception
    return app


def visible_text(app):
    return "\n".join(
        str(element.value)
        for kind in ("text", "markdown", "caption", "info", "warning", "error", "success")
        for element in getattr(app, kind)
    )


@pytest.fixture
def nearby_transport(monkeypatch):
    monkeypatch.setenv("EGEN_SERVICE_KEY", "offline-synthetic-key")
    from er_finder import http_retry
    from er_finder.medical_api import client

    monkeypatch.setattr(client, "SERVICE_KEY", "offline-synthetic-key")
    requests = []

    def respond(request):
        requests.append(request)
        filename = "nearby_gangnam.xml"
        if request.url.path.endswith("getEgytBassInfoInqire"):
            filename = f"detail_{request.url.params['HPID']}.xml"
        return httpx.Response(200, text=(FIXTURES / filename).read_text(encoding="utf-8"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(http_retry.httpx, "request", transport.request)
        yield requests


def test_missing_key_disables_api_form_without_importing_the_client(monkeypatch):
    import builtins

    monkeypatch.setenv("EGEN_SERVICE_KEY", "")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("er_finder.medical_api"):
            raise AssertionError("medical_api must load only after explicit query")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    app = open_view()
    assert app.button(key="api_search").disabled
    assert all(widget.disabled for widget in app.number_input)
    assert "EGEN_SERVICE_KEY" in visible_text(app)
    assert not app.dataframe


def test_query_displays_recorded_source_data_and_reruns_make_no_request(nearby_transport):
    app = open_view()
    app.run()
    assert nearby_transport == []
    app.button(key="api_search").click().run()
    assert not app.exception
    assert len(nearby_transport) == 1
    query = nearby_transport[0].url.params
    assert float(query["WGS84_LAT"]) == 37.497942
    assert float(query["WGS84_LON"]) == 127.027621
    table = app.dataframe[0].value
    assert "A1100141" in list(table["기관 ID"])
    assert table.loc[table["기관 ID"] == "A1100141", "기관명"].item() == "강남베드로병원"
    assert table.loc[table["기관 ID"] == "A1100141", "주소"].item().startswith("서울특별시 강남구")
    assert any(value != "확인 불가" for value in table["대표전화"])
    assert "원자료" in visible_text(app)
    assert "offline-synthetic-key" not in visible_text(app)
    app.number_input(key="api_lat").set_value(37.6).run()
    assert len(nearby_transport) == 1
    assert app.dataframe[0].value.equals(table)
    assert "37.497942" in visible_text(app)


def test_submit_uses_new_coordinates_and_radius_in_the_same_event(nearby_transport):
    app = open_view()
    app.number_input(key="api_lat").set_value(37.5)
    app.number_input(key="api_lon").set_value(127.1)
    app.selectbox(key="api_radius").set_value(10)
    app.button(key="api_search").click().run()
    assert not app.exception
    assert len(nearby_transport) == 1
    query = nearby_transport[0].url.params
    assert float(query["WGS84_LAT"]) == 37.5
    assert float(query["WGS84_LON"]) == 127.1
    assert "10 km" in visible_text(app)
    assert any(value > 5 for value in app.dataframe[0].value["거리 (km)"])


def test_empty_or_failed_source_does_not_claim_no_hospitals(monkeypatch):
    monkeypatch.setenv("EGEN_SERVICE_KEY", "offline-synthetic-key")
    from er_finder import http_retry
    from er_finder.medical_api import client

    monkeypatch.setattr(client, "SERVICE_KEY", "offline-synthetic-key")
    empty_xml = (FIXTURES / "empty_items_blank.xml").read_text(encoding="utf-8")
    response_transport = httpx.MockTransport(lambda request: httpx.Response(200, text=empty_xml))
    with httpx.Client(transport=response_transport) as transport:
        monkeypatch.setattr(http_retry.httpx, "request", transport.request)
        app = open_view()
        app.button(key="api_search").click().run()
    assert not app.exception
    assert not app.dataframe
    assert "빈 응답" in visible_text(app)
    assert "실패" not in visible_text(app)


def test_failure_clears_old_rows_and_hides_raw_exception(nearby_transport, monkeypatch):
    from er_finder.medical_api import client

    app = open_view()
    app.button(key="api_search").click().run()
    assert app.dataframe

    def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE_REQUEST serviceKey=secret-dummy-key")

    monkeypatch.setattr(client, "list_nearby_ers", unavailable)
    app.button(key="api_search").click().run()
    assert not app.exception
    assert app.error
    assert not app.dataframe
    assert "PRIVATE_REQUEST" not in visible_text(app)
    assert "secret-dummy-key" not in visible_text(app)
    app.run()
    assert app.error
    assert not app.dataframe


def test_other_browser_session_does_not_receive_previous_query(nearby_transport):
    first = open_view()
    first.button(key="api_search").click().run()
    assert first.dataframe
    second = open_view()
    assert not second.dataframe
    assert len(nearby_transport) == 1


def test_detail_requires_click_and_displays_selected_hospital_source(nearby_transport):
    app = open_view()
    app.button(key="api_search").click().run()
    app.selectbox(key="api_detail_hpid").set_value("A1100141").run()
    assert len(nearby_transport) == 1
    app.button(key="api_detail_search").click().run()
    assert not app.exception
    assert len(nearby_transport) == 2
    assert nearby_transport[-1].url.params["HPID"] == "A1100141"
    assert "서울특별시 강남구 남부순환로 2649, 베드로병원 (도곡동)" in visible_text(app)
    assert "1544-7522" in visible_text(app)
    app.run()
    assert len(nearby_transport) == 2
    assert "1544-7522" in visible_text(app)
    app.selectbox(key="api_detail_hpid").set_value("A1100057").run()
    assert len(nearby_transport) == 2
    assert "서울특별시 강남구 남부순환로 2649, 베드로병원 (도곡동)" not in visible_text(app)


def test_detail_uses_new_selection_and_a_new_search_clears_the_detail(nearby_transport):
    app = open_view()
    app.button(key="api_search").click().run()
    app.selectbox(key="api_detail_hpid").set_value("A1100141")
    app.button(key="api_detail_search").click().run()
    assert not app.exception
    assert nearby_transport[-1].url.params["HPID"] == "A1100141"
    assert "1544-7522" in visible_text(app)
    app.button(key="api_search").click().run()
    assert not app.exception
    assert len(nearby_transport) == 3
    assert "서울특별시 강남구 남부순환로 2649, 베드로병원 (도곡동)" not in visible_text(app)
    assert app.dataframe


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_detail_failure_keeps_nearby_table_and_hides_previous_detail(
    nearby_transport, monkeypatch, failure
):
    from er_finder.medical_api import client

    app = open_view()
    app.button(key="api_search").click().run()
    app.selectbox(key="api_detail_hpid").set_value("A1100141")
    app.button(key="api_detail_search").click().run()
    assert "1544-7522" in visible_text(app)

    def unavailable(hpid, **kwargs):
        if failure == "exception":
            raise RuntimeError("PRIVATE_DETAIL serviceKey=never-display-this")
        return {}

    monkeypatch.setattr(client, "get_er_detail", unavailable)
    app.button(key="api_detail_search").click().run()
    assert not app.exception
    assert app.dataframe
    assert "PRIVATE_DETAIL" not in visible_text(app)
    assert "never-display-this" not in visible_text(app)
    assert "서울특별시 강남구 남부순환로 2649, 베드로병원 (도곡동)" not in visible_text(app)
    assert ("실패" if failure == "exception" else "빈 응답") in visible_text(app)
