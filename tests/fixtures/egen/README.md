# E-Gen fixture

`scripts/record_egen.py`로 녹화한 실제 응답 + 손으로 만든 오류/엣지 케이스.
단위 테스트(`tests/unit/medical_api/`)는 실제 API를 부르지 않고 이 파일들만 사용한다.

## 정상 응답 (실제 API, 녹화일 2026-09-11)

지역: 강남구(서울), 강릉시·영월군(강원특별자치도)

| 파일 | 오퍼레이션 |
|---|---|
| `nearby_<region>.xml` | 위치정보 조회 (`getEgytLcinfoInqire`) |
| `bed_status_<region>.xml` | 실시간 가용병상 조회 (`getEmrrmRltmUsefulSckbdInfoInqire`) |
| `severe_<region>.xml` | 중증질환자 수용가능정보 조회 (`getSrsillDissAceptncPosblInfoInqire`) |
| `detail_<hpid>.xml` | 기관 기본정보 조회 (`getEgytBassInfoInqire`), 지역별 상위 병원 1~2곳 |

## 오류·엣지 케이스 (수동 작성 — 실 API로 재현하기 어려움)

| 파일 | 내용 |
|---|---|
| `error_result_code.xml` | `resultCode` ≠ `00` |
| `empty_items_no_tag.xml` | `items` 태그 자체가 없음 |
| `empty_items_blank.xml` | `items` 태그는 있지만 `item`이 없음 |
| `bed_status_missing_values.xml` | `hvec` 음수/빈 문자열, `hvidate` 누락 |

타임아웃·5xx 같은 네트워크 오류는 파일로 만들 수 없어서 테스트에서 직접 예외를
발생시켜 확인한다 (`test_resilience.py`).

## 참고

- 강원(2023년), 전북(2024년)은 `OO특별자치도`로 개칭됨. `STAGE1`에 옛 이름을 쓰면
  실제 데이터가 있어도 0건이 나온다 (design.md §B.2).
- `SM_TYPE`(중증질환 구분) 파라미터는 생략하고 지역 전체를 받아왔다 (design.md §B.1.5).
