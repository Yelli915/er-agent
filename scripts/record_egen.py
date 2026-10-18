"""E-Gen 응급의료정보 API 응답을 녹화해 tests/fixtures/egen/에 저장하는 1회성 스크립트.

단위 테스트는 이 스크립트로 만든 fixture만 사용

사용법:
    python scripts/record_egen.py                        # 전체 지역 × 전체 오퍼레이션
    python scripts/record_egen.py --regions gangnam      # 특정 지역만
    python scripts/record_egen.py --ops nearby,bed       # 특정 오퍼레이션만
    python scripts/record_egen.py --dry-run              # 실제 호출 없이 요청 내용만 출력

    - 이 스크립트는 config.py(조원 1, 아직 구현 전)에 의존하지 않고 .env를 직접 읽는다.
      config.py가 준비되면 그쪽 Settings로 옮겨도 된다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "egen"

ENDPOINTS = {
    "nearby": "/getEgytLcinfoInqire",
    "bed": "/getEmrrmRltmUsefulSckbdInfoInqire",
    "severe": "/getSrsillDissAceptncPosblInfoInqire",
    "detail": "/getEgytBassInfoInqire",
}


@dataclass(frozen=True)
class Region:
    key: str
    label: str  # fixture 파일명에 쓰는 이름
    lat: float
    lon: float
    sido: str  # STAGE1
    sigungu: str  # STAGE2


# 위경도는 각 지역 중심가 기준 근사값. 반경 조회 결과가 비어 있으면 조정한다.
REGIONS: dict[str, Region] = {
    "gangnam": Region(
        key="gangnam",
        label="gangnam",
        lat=37.497942,
        lon=127.027621,
        sido="서울특별시",
        sigungu="강남구",
    ),
    "gangneung": Region(
        key="gangneung",
        label="gangneung",
        lat=37.751853,
        lon=128.876191,
        sido="강원특별자치도",  # 2023년 개칭. "강원도"로 조회하면 STAGE1 필터가 0건 나옴
        sigungu="강릉시",
    ),
    "yeongwol": Region(
        key="yeongwol",
        label="yeongwol",
        lat=37.183740,
        lon=128.461418,
        sido="강원특별자치도",  # 2023년 개칭. "강원도"로 조회하면 STAGE1 필터가 0건 나옴
        sigungu="영월군",
    ),
}

# 중증질환 구분 코드: docs/design.md §B.1.5 표가 아직 비어 있어 미확정.
# 실제 응답을 받은 뒤 이 값과 표를 함께 채운다.
DEFAULT_CONDITION = "TODO_CONFIRM_CONDITION_CODE"


def mask_key(text: str, service_key: str) -> str:
    """로그에 serviceKey 원문이 남지 않게 마스킹한다."""
    if not service_key:
        return text
    return text.replace(service_key, "***MASKED***")


def masked_params(params: dict[str, str]) -> dict[str, str]:
    """출력용으로 serviceKey 값을 가린 params 사본을 만든다."""
    if "serviceKey" not in params:
        return params
    return {**params, "serviceKey": "***MASKED***"}


def build_params(base_params: dict[str, str], service_key: str) -> dict[str, str]:
    return {"serviceKey": service_key, "pageNo": "1", "numOfRows": "10", **base_params}


def fetch(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    params: dict[str, str],
    *,
    timeout: float,
    dry_run: bool,
) -> str | None:
    url = base_url.rstrip("/") + endpoint
    if dry_run:
        print(f"[dry-run] GET {url} params={masked_params(params)}")
        return None

    try:
        resp = client.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        masked = mask_key(str(exc), params.get("serviceKey", ""))
        print(f"[error] {endpoint} 호출 실패: {masked}", file=sys.stderr)
        return None

    return resp.text


def extract_hpids(nearby_xml: str, limit: int = 2) -> list[str]:
    """위치정보 조회 응답에서 detail 조회용 hpid를 뽑아낸다.

    parser.py에 아직 의존하지 않기 위해 최소한의 ElementTree 파싱만 한다.
    """
    try:
        root = ET.fromstring(nearby_xml)
    except ET.ParseError:
        return []
    hpids = [el.text.strip() for el in root.iter("hpid") if el.text and el.text.strip()]
    return hpids[:limit]


def save_fixture(name: str, content: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(content, encoding="utf-8")
    print(f"saved: {path.relative_to(REPO_ROOT)}")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def run(
    *,
    regions: list[str],
    ops: list[str],
    condition: str,
    out_dir: Path,
    dry_run: bool,
) -> None:
    # 공공데이터포털은 서비스키를 Encoding/Decoding 두 버전으로 준다. .env에 어느 쪽이
    # 들어있든 httpx가 params=에서 자체적으로 다시 인코딩하므로, 여기서 미리 한 번
    # unquote 해서 이중 인코딩(예: %2B → %252B)으로 인증이 깨지는 걸 방지한다.
    service_key = unquote(os.environ.get("EGEN_SERVICE_KEY", ""))
    base_url = os.environ.get(
        "EGEN_BASE_URL", "http://apis.data.go.kr/B552657/ErmctInfoInqireService"
    )
    timeout = float(os.environ.get("ER_REQUEST_TIMEOUT", "5"))

    if not dry_run and not service_key:
        print("EGEN_SERVICE_KEY가 .env에 없습니다. --dry-run으로 먼저 확인하세요.", file=sys.stderr)
        raise SystemExit(1)

    if "severe" in ops and condition == DEFAULT_CONDITION:
        print(
            "[warn] --condition 미지정: 중증질환 구분 코드가 아직 미확정입니다 "
            "(docs/design.md §B.1.5). 임시값으로 호출합니다.",
            file=sys.stderr,
        )

    with httpx.Client() as client:
        for region_key in regions:
            region = REGIONS[region_key]

            nearby_xml: str | None = None

            if "nearby" in ops:
                params = build_params(
                    {"WGS84_LAT": str(region.lat), "WGS84_LON": str(region.lon)}, service_key
                )
                nearby_xml = fetch(
                    client, base_url, ENDPOINTS["nearby"], params, timeout=timeout, dry_run=dry_run
                )
                if nearby_xml:
                    save_fixture(f"nearby_{region.label}.xml", nearby_xml, out_dir)

            if "bed" in ops:
                params = build_params(
                    {"STAGE1": region.sido, "STAGE2": region.sigungu}, service_key
                )
                xml_text = fetch(
                    client, base_url, ENDPOINTS["bed"], params, timeout=timeout, dry_run=dry_run
                )
                if xml_text:
                    save_fixture(f"bed_status_{region.label}.xml", xml_text, out_dir)

            if "severe" in ops:
                # SM_TYPE은 옵션이라 생략하면 지역 내 병원 전체가 mkioskty1~28을 다 붙여서
                # 온다(docs/design.md §B.1.5) — 그래서 condition을 넘겨도 필터링용
                # SM_TYPE으로는 안 쓰고, 전체를 받아 fixture로 남긴다.
                params = build_params(
                    {"STAGE1": region.sido, "STAGE2": region.sigungu}, service_key
                )
                xml_text = fetch(
                    client, base_url, ENDPOINTS["severe"], params, timeout=timeout, dry_run=dry_run
                )
                if xml_text:
                    save_fixture(f"severe_{region.label}.xml", xml_text, out_dir)

            if "detail" in ops:
                if dry_run:
                    print(f"[dry-run] detail: {region.label}은 nearby 응답에서 hpid를 얻어야 함")
                    continue
                if nearby_xml is None:
                    print(
                        f"[skip] {region.label} detail: nearby 응답이 없어 hpid를 알 수 없습니다 "
                        "(--ops에 nearby를 포함하세요).",
                        file=sys.stderr,
                    )
                    continue
                for hpid in extract_hpids(nearby_xml):
                    params = build_params({"HPID": hpid}, service_key)
                    xml_text = fetch(
                        client,
                        base_url,
                        ENDPOINTS["detail"],
                        params,
                        timeout=timeout,
                        dry_run=dry_run,
                    )
                    if xml_text:
                        save_fixture(f"detail_{sanitize_filename(hpid)}.xml", xml_text, out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        default="all",
        help=f"쉼표로 구분한 지역 키 (선택: {', '.join(REGIONS)}) 또는 all (기본값)",
    )
    parser.add_argument(
        "--ops",
        default="all",
        help=f"쉼표로 구분한 오퍼레이션 (선택: {', '.join(ENDPOINTS)}) 또는 all (기본값)",
    )
    parser.add_argument(
        "--condition",
        default=DEFAULT_CONDITION,
        help="중증질환 수용가능 조회 파라미터 값 (docs/design.md §B.1.5 확정 전 임시)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="fixture 저장 경로 (기본: tests/fixtures/egen)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 호출 없이 요청 URL·파라미터만 출력",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv(REPO_ROOT / ".env")

    regions = list(REGIONS) if args.regions == "all" else args.regions.split(",")
    ops = list(ENDPOINTS) if args.ops == "all" else args.ops.split(",")

    unknown_regions = set(regions) - set(REGIONS)
    if unknown_regions:
        raise SystemExit(f"알 수 없는 지역: {unknown_regions} (선택 가능: {list(REGIONS)})")
    unknown_ops = set(ops) - set(ENDPOINTS)
    if unknown_ops:
        raise SystemExit(f"알 수 없는 오퍼레이션: {unknown_ops} (선택 가능: {list(ENDPOINTS)})")

    run(
        regions=regions,
        ops=ops,
        condition=args.condition,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
