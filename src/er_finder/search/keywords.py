"""Korean keyword rules that decide how a new message reshapes the search context."""

import re

FOLLOWUP_MARKERS = ("다시", "같은 위치", "반경", "지난번")
SPECIALTY_MARKERS = ("소아", "봉합", "정형외과", "산부인과")
# A location answer with no new symptom must not erase the prior assessment.
NEW_SYMPTOM = re.compile(
    r"아프|통증|불편|답답|식은땀|부었|붓|찢|다쳤|넘어|구토|토하|설사|두통|어지|기침|발열|고열|출혈|화상|의식|경련|숨|마비"
)


def is_followup(text: str) -> bool:
    return any(marker in text for marker in FOLLOWUP_MARKERS)


def mentions_new_symptom(text: str) -> bool:
    return bool(NEW_SYMPTOM.search(text))


def requests_specialty(text: str) -> bool:
    return any(marker in text for marker in SPECIALTY_MARKERS)
