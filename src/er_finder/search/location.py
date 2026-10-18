"""Offline extraction of a location phrase from user text."""

import re

from er_finder.search.coordinates import extract_coordinates


def extract_location(text: str) -> str | None:
    coordinates = extract_coordinates(text)
    if coordinates is not None:
        return coordinates
    # Offline extraction; live model may additionally extract a precise query from the input.
    if "우리 동네" in text or "우리동네" in text:
        return "우리 동네"
    known = (
        "강남구 역삼동",
        "역삼동",
        "강남구",
        "강남역",
        "경포대",
        "강릉",
        "영월군 상동읍",
        "상동읍",
        "영월군",
        "마포구 합정동",
        "합정동",
        "마포구",
    )
    for place in known:
        if place in text:
            return place
    # Strip Korean particles following administrative names rather than swallowing symptoms.
    tokens = re.findall(
        r"[가-힣0-9]+?(?:특별자치도|특별시|광역시|시|군|구|읍|면|동|역|대로|로)(?=\s|[,.!?]|(?:에서|이에요|인데|입니다|이야)|$)",
        text,
    )
    tokens = [t for t in tokens if t not in {"다시", "혹시", "응급시", "응급실로", "병원으로"}]
    if tokens:
        return " ".join(tokens)
    m = re.search(r"(?:^|[,.!?]\s*)([가-힣A-Za-z0-9 ]{2,35}?)\s*(?:근처|주변|앞에서)", text)
    return m[1].strip() if m else None
