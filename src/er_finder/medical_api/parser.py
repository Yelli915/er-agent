"""E-Gen API가 돌려주는 XML을 dict로 바꾸는 모듈.

design.md B.1.2~B.1.6을 따른다. 여기서 만든 dict가 도구 경계를 넘어가는 값이라
XML 관련 코드는 이 파일 밖으로 나가지 않게 한다.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

KOREA_TZ = ZoneInfo("Asia/Seoul")


class EgenResponseError(Exception):
    """resultCode가 00이 아니거나 XML을 파싱할 수 없을 때 발생시키는 예외."""


# 카카오 시도 표기 -> E-Gen 시도 표기 (design.md §B.2)
# 강원(2023년), 전북(2024년)은 "OO특별자치도"로 행정구역명이 바뀌었다. 실제 E-Gen
# 응답 주소로 확인함(강원): STAGE1="강원도"로 조회하면 0건, "강원특별자치도"로 조회해야 나옴.
SIDO_ALIASES = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
    "강원도": "강원특별자치도",
    "전라북도": "전북특별자치도",
}


def normalize_sido(name: str) -> str:
    """카카오 시도 표기를 E-Gen 시도 표기로 바꾼다. 이미 E-Gen 표기면 그대로 둔다."""
    return SIDO_ALIASES.get(name, name)


def region_from_address(address: str | None) -> tuple[str | None, str | None]:
    """Use the hospital's own address; never substitute the user's search region."""
    if not address:
        return None, None
    parts = address.split()
    sido = normalize_sido(parts[0])
    if sido not in set(SIDO_ALIASES.values()):
        return None, None
    if sido == "세종특별자치시":
        return sido, ""  # There is no lower municipality; omit STAGE2 on requests.
    if len(parts) < 2 or not re.fullmatch(r"[가-힣]+[시군구]", parts[1]):
        return sido, None
    return sido, parts[1]


def find_text(element: ET.Element, tag: str) -> str | None:
    """element 안에서 tag의 텍스트를 찾는다. 태그가 없거나 빈 문자열이면 None."""
    found = element.find(tag)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text if text else None


def to_float(value: str | None) -> float | None:
    """문자열을 float으로 바꾼다. 없거나 숫자가 아니면 None."""
    if value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def to_bed_count(value: str | None) -> int | None:
    """hvec(가용 병상 수) 규칙: 음수·빈 값은 None (design.md §B.1.4)."""
    number = to_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def hvidate_to_iso(value: str | None) -> str | None:
    """hvidate(yyyyMMddHHmmss)를 ISO 8601 문자열로 바꾼다."""
    if value is None or not re.fullmatch(r"[0-9]{14}", value):
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=KOREA_TZ).isoformat()


def yn_to_bool(value: str | None) -> bool | None:
    """E-Gen의 Y/N 문자열을 True/False로 바꾼다. 결측이면 None."""
    if value is None:
        return None
    value = value.strip().upper()
    if value in {"Y", "가능"}:
        return True
    if value in {"N", "불가능"}:
        return False
    return None


def get_items(xml_text: str) -> list[ET.Element]:
    """공통 응답 봉투(resultCode/items)를 확인하고 item 목록을 돌려준다.

    resultCode가 00이 아니면 EgenResponseError를 던진다.
    item이 하나도 없는 경우(태그 자체가 없는 경우와 빈 문자열인 경우 둘 다)는
    같은 빈 리스트로 정규화한다.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raise EgenResponseError("E-Gen XML을 파싱할 수 없습니다.") from None

    result_code = find_text(root, "header/resultCode")
    if result_code != "00":
        raise EgenResponseError("E-Gen 응답 오류입니다.")

    if root.find("body") is None:
        raise EgenResponseError("E-Gen 응답 본문이 없습니다.")

    items_element = root.find("body/items")
    if items_element is None:
        return []
    return items_element.findall("item")


def parse_nearby(xml_text: str) -> list[dict[str, Any]]:
    """위치정보 조회(getEgytLcinfoInqire) 응답 -> list_nearby_ers용 dict 목록."""
    result = []
    for item in get_items(xml_text):
        address = find_text(item, "dutyAddr")
        sido, sigungu = region_from_address(address)
        result.append(
            {
                "hpid": find_text(item, "hpid"),
                "name": find_text(item, "dutyName"),
                # dutyTel3(응급실 직통 가정)는 실제로 거의 결측이라 대표전화(dutyTel1)를 쓴다
                # (design.md §B.1.3, 팀 합의).
                "er_tel": find_text(item, "dutyTel1"),
                "lat": to_float(find_text(item, "latitude")),
                "lon": to_float(find_text(item, "longitude")),
                "distance_km": to_float(find_text(item, "distance")),
                "address": address,
                "sido": sido,
                "sigungu": sigungu,
            }
        )
    return result


def parse_bed_status(xml_text: str) -> list[dict[str, Any]]:
    """실시간 가용병상 조회(getEmrrmRltmUsefulSckbdInfoInqire) 응답 -> dict 목록."""
    result = []
    for item in get_items(xml_text):
        result.append(
            {
                "hpid": find_text(item, "hpid"),
                "name": find_text(item, "dutyName"),
                "er_beds_available": to_bed_count(find_text(item, "hvec")),
                "beds_updated_at": hvidate_to_iso(find_text(item, "hvidate")),
            }
        )
    return result


def parse_severe(xml_text: str) -> list[dict[str, Any]]:
    """중증질환자 수용가능정보 조회(getSrsillDissAceptncPosblInfoInqire) 응답 -> dict 목록.

    mkioskty1~28을 전부 True/False/None으로 바꿔서 "mkioskty" 안에 번호별로 담는다.
    어떤 번호가 어떤 질환에 해당하는지는 client.py가 결정한다 (design.md §B.1.5).
    """
    result = []
    for item in get_items(xml_text):
        mkioskty = {}
        for number in range(1, 29):
            value = find_text(item, f"MKioskTy{number}")
            if value is None:
                value = find_text(item, f"mkioskty{number}")
            mkioskty[number] = yn_to_bool(value)
        result.append(
            {
                "hpid": find_text(item, "hpid"),
                "name": find_text(item, "dutyName"),
                "mkioskty": mkioskty,
            }
        )
    return result


def parse_er_detail(xml_text: str) -> dict[str, Any]:
    """기관 기본정보 조회(getEgytBassInfoInqire) 응답 -> get_er_detail용 dict.

    결과가 없으면 빈 dict를 돌려준다.
    """
    items = get_items(xml_text)
    if not items:
        return {}
    item = items[0]

    days = ["월", "화", "수", "목", "금", "토", "일", "공휴일"]
    hours_parts = []
    for day_index, day_name in enumerate(days, start=1):
        start = find_text(item, f"dutyTime{day_index}s")
        end = find_text(item, f"dutyTime{day_index}c")
        if start and end:
            hours_parts.append(f"{day_name} {start}~{end}")

    # dutyTel3(응급실 직통 가정)는 실제로 거의 결측이라 대표전화(dutyTel1)를 쓴다
    # (design.md §B.1.6, 팀 합의). er_tel/main_tel이 당분간 같은 값이 된다.
    main_tel = find_text(item, "dutyTel1")
    return {
        "hpid": find_text(item, "hpid"),
        "name": find_text(item, "dutyName"),
        "address": find_text(item, "dutyAddr"),
        "er_tel": main_tel,
        "main_tel": main_tel,
        "hours": ", ".join(hours_parts),
    }
