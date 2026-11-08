"""cache.py 단위 테스트."""

from er_finder.medical_api import cache


def test_make_key():
    assert cache.make_key("서울특별시", "강남구") == "서울특별시:강남구"


def test_get_missing_key_returns_none():
    store = {}
    assert cache.get(store, "없는키") is None


def test_save_then_get_returns_saved_value():
    store = {}
    cache.save(store, "키", [{"hpid": "A1"}])
    assert cache.get(store, "키") == [{"hpid": "A1"}]


def test_get_expired_returns_none():
    store = {}
    cache.save(store, "키", [{"hpid": "A1"}])
    value, _ = store["키"]
    store["키"] = (value, 0)  # 아주 오래전(유닉스 시각 0)에 저장된 것처럼 만든다
    assert cache.get(store, "키") is None
