import pytest

from er_finder.search.radius import radius_sequence


def test_car_and_transit_default_to_five_km_start():
    assert radius_sequence("car") == [5, 10, 20, 30]
    assert radius_sequence("transit") == [5, 10, 20, 30]


def test_walk_defaults_to_three_km_start():
    assert radius_sequence("walk") == [3, 10, 20, 30]


def test_invalid_transport_raises():
    with pytest.raises(ValueError):
        radius_sequence("bike")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("반경 10km 이내", [10, 20, 30]),
        ("20KM", [20, 30]),
        ("30 킬로", [30]),
        ("5킬로미터 이내", [5, 10, 20, 30]),
    ],
)
def test_explicit_supported_radius_in_text_overrides_default_and_bounds_expansion(text, expected):
    assert radius_sequence("car", text) == expected


def test_unsupported_explicit_radius_falls_back_to_transport_default():
    # 7 is not one of the recognized radii (3/5/10/20/30), so the regex
    # doesn't match and the transport's default sequence is used instead.
    assert radius_sequence("car", "7km 이내") == [5, 10, 20, 30]
    assert radius_sequence("walk", "7km 이내") == [3, 10, 20, 30]


def test_two_digit_number_containing_a_supported_digit_is_misread():
    # Known quirk: the regex matches a bare digit inside a larger number,
    # so "15km"/"25km" are read as "5km" instead of being rejected.
    assert radius_sequence("walk", "15km 이내") == [5, 10, 20, 30]
    assert radius_sequence("walk", "25km 이내") == [5, 10, 20, 30]
