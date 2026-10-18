# 설계·계약

이 문서는 모듈 간 경계(A), 외부 API 계약(B), 화면 연결 계약(C)을 정리한다.
여기에 적힌 타입과 시그니처는 모든 조원이 공유하는 경계이므로, 변경할 때는 이 문서를 먼저 수정하고 공유한다.
검증 결과와 측정 절차는 [verification.md](verification.md)에서 다룬다.

---

## A. 공통 인터페이스

### A.1 데이터 모델 (`er_finder/models.py`)

모두 Pydantic 모델. 모델의 Structured Output 스키마이자 모듈 간 전달 타입이다.

`TriageAssessment`는 분류 모델(gemini-3.5-flash-lite)의 출력이다.

| 필드 | 타입 | 제약 |
|---|---|---|
| `severity` | `critical` \| `urgent` \| `standard` | 3단계 고정 |
| `condition` | `심근경색` \| `뇌출혈` \| `뇌졸중` \| `중증외상` \| `화상` \| `분만` \| None | `critical`/`urgent`일 때만 값 허용 |
| `injection` | `bool` | `True`면 가드레일 차단 |
| `confidence` | `float` 0.0~1.0 | 0.6 미만이면 `severity` 한 단계 상향 |

`HospitalCandidate`는 후보 병원이다.

| 필드 | 타입 | 제약 |
|---|---|---|
| `hpid`, `name`, `address` | `str` | 도구 결과에 존재하는 값만, 도구 결과와 동일 |
| `distance_km` | `float` | 0.0 이상, 소수 첫째 자리 |
| `er_beds_available` | `int \| None` | `None`이면 '확인 불가' |
| `beds_updated_at` | `str \| None` | `hvidate` 값, 15분 초과 시 `stale=True` |
| `accepts_condition` | `yes` \| `no` \| `unknown` | `standard`이면 항상 `unknown` |
| `er_tel` | `str \| None` | 도구 결과와 동일 |

`ERSearchReply`는 최종 응답이다.

| 필드 | 제약 |
|---|---|
| `severity` | `critical` \| `urgent` \| `standard` |
| `call_119_first` | `severity=critical`이면 항상 `True` |
| `search_radius_km` | 5, 10, 20, 30 중 하나 |
| `hospitals` | 0~3개, `distance_km` 오름차순 |
| `no_candidate_reason` | `hospitals`가 비면 필수 |
| `data_timestamp` | ISO 8601 |
| `next_action` | 80자 이내, 진단·처방 표현 금지 |
| `disclaimer` | 고정 문장과 완전 일치: `"응급실 상황은 수시로 변하므로 방문 전 전화 확인을 권장합니다"` |

### A.2 도구 (`er_finder/agent/tools.py`)

`tools.py`는 각 모듈의 함수를 `@tool`로 감싸 등록만 한다. 실제 로직은 구현 모듈에 둔다.

| 도구 | 입력 | 출력 | 구현 |
|---|---|---|---|
| `geocode` | `query` | `dict(lat, lon, address, sido, sigungu, found)` | `search/geocoder.py` |
| `list_nearby_ers` | `lat, lon, radius_km=5` | `list[dict(hpid, name, distance_km, er_tel, lat, lon)]` | `medical_api/client.py` + `search/distance.py` |
| `get_er_bed_status` | `regions, hpids` | `list[dict(hpid, name, er_beds_available, beds_updated_at, stale)]` | `medical_api/client.py` + `cache.py` |
| `get_severe_acceptance` | `regions, condition` | `list[dict(hpid, name, acceptable)]` | `medical_api/client.py` |
| `get_er_detail` | `hpid` | `dict(name, address, er_tel, main_tel, hours)` | `medical_api/client.py` |
| `save_visit_plan` | `hpid, name, symptom_summary` | `dict(saved, visit_id)` | `memory/visit_plan.py` + `store.py` |

도구는 다음 순서로 호출한다. `geocode`, `list_nearby_ers`, `get_er_bed_status`를 차례로 호출하고, 위급도가 critical 또는 urgent이면서 질환 조건이 있으면 `get_severe_acceptance`를 추가한다. 이후 상위 3곳에 `get_er_detail`을 호출하며, 사용자가 승인한 뒤에야 `save_visit_plan`을 호출한다. 후보가 없으면 `list_nearby_ers`부터 다시 시작해 반경을 5km, 10km, 20km, 30km 순으로 최대 3회 넓힌다.

도구는 다음 규약을 따른다.

- 예외를 밖으로 던지지 않고, 실패는 `found=False`, 빈 목록, `acceptable=None`, `stale=True` 같은 값으로 표현한다.
- 읽기 전용이다. 유일한 쓰기 도구인 `save_visit_plan`은 HITL 승인 이후에만 실행된다.
- E-Gen의 XML 응답은 도구 경계를 넘기 전에 dict로 변환한다.
- `get_er_bed_status`는 시도·시군구 단위 응답을 `hpids`로 걸러, 모델이 긴 목록을 읽지 않도록 한다.

### A.3 Context (`er_finder/memory/`)

| 항목 | 유형 | 모듈 | 용도 |
|---|---|---|---|
| `user_id`, `transport(walk/car/transit)` | Runtime Context | `context.py` | 미들웨어, `save_visit_plan`, `list_nearby_ers` |
| `messages`, `current_location`, `triage`, `search_radius_km`, `candidates` | State | `state.py` | 도구 선택, 반경 확대, `EvidenceCheck`, 최종 응답 |
| `bed_cache` | State(임시) | `medical_api/cache.py` | `get_er_bed_status` (60초) |
| `home_address`, `recent_visits` | Store | `store.py` | `ProfileDynamicPrompt` |

체크포인터는 `InMemorySaver`(`thread_id`는 `user_id`, `memory/session.py`)를 사용하고, 프로필은 `InMemoryStore`(`memory/store.py`)에 보관한다. `current_location`은 세션이 끝나면 폐기하며, Store에는 사용자가 동의한 `home_address`와 `recent_visits`만 남긴다.

### A.4 모듈 경계

- `search`는 좌표, 거리, 정렬만 담당한다. 병상과 중증 수용 데이터는 `medical_api`가 정규화한 dict로만 받는다.
- `guardrails/middleware.py`는 미들웨어를 제공하고, `agent/graph.py`의 `build_graph()`가 이를 조립하고 `agent/runner.py`의 `ERFinder`가 호출한다. `agent/factory.py`는 모델과 분류기의 생성만 맡으며 웹과 벤치마크가 함께 사용한다.
- `ERFinder`는 체크포인터와 Store를 외부에서 주입받아 사용한다.
- `cli/app.py`는 `ERFinder`의 실행과 승인 재개 인터페이스만 호출한다. 렌더링은 `ERSearchReply`만 입력으로 받는다.

### A.5 미들웨어 조립 순서 (`agent/graph.py`)

```
before_agent   : EmergencyInputGuard
before_model   : ProfileDynamicPrompt → PIIMiddleware
wrap_model_call: ModelCallLimitMiddleware(run_limit=10)
                 → ServerPlannedCalls (server_planned_calls=True일 때만)
                 → ModelCounter
wrap_tool_call : ToolRetryMiddleware(max_retries=2, 초기 지연 1s)
after_model    : HumanInTheLoopMiddleware(save_visit_plan), ToolSafetyMiddleware
                 (after_model은 목록 역순 실행: 검증이 HITL보다 먼저)
after_agent    : EvidenceCheckMiddleware
```

`ServerPlannedCalls`는 서버가 이미 다음 도구 호출(`session.next_calls()`)을 알고 있으면 모델을 호출하지 않고 해당 도구 호출을 직접 생성한다. 같은 턴의 `ToolSafetyMiddleware.after_model`이 호출 계획을 다시 계산하지 않도록 `session.planned_calls`로 전달하며, 이 값은 다음 `before_model`에서 비운다. 모델은 최종 응답을 생성할 때와 방문 계획의 승인을 기다릴 때만 호출한다. `ERFinder(server_planned_calls=False)`로 이 기능을 끄면 개선 전 동작인 검색 단계별 모델 호출로 돌아간다.

`ERFinder.chat()`의 결과에는 계측 정보인 `RunDiagnostics`가 포함된다. 메인 모델(`model_calls`, `input_tokens` 등)과 분류기(`classifier_calls`, `classifier_*_tokens`)를 따로 집계하며, 분류기를 호출하지 않은 턴에서는 `classifier_calls`가 0이다. 토큰 수가 일부 누락되면 `tokens_complete`가 `False`가 되고 이 경우 비용은 계산하지 않는다. 입력과 출력의 원문은 담지 않는다.

### A.6 오류 처리

| 상황 | 처리 | 사용자에게 보이는 형태 |
|---|---|---|
| 외부 API 타임아웃·5xx·429 | 2회 재시도(백오프 1초) | 실패 시 아래로 |
| 재시도 실패 + 캐시 있음 | 60초 캐시, `stale=True` | '갱신 지연' + 전화 확인 권고 |
| 재시도 실패 + 캐시 없음 | 빈 목록 | 모델이 상황 설명 |
| `geocode` 0건 | `found=False` | 위치 재질문 |
| 중증 수용 조회 실패 | `acceptable=None` | `unknown`이며 '확인 불가'로 표시 |
| `get_er_detail` 실패 | 목록 값으로 대체 | 누락 필드 '확인 불가' |
| 모델 호출 10회 도달 | 현재 후보로 종료 응답 | 정상 종료 |
| 반경 3회 확대 후 후보 0 | `no_candidate_reason` | 후보 없음 + 119 안내 |
| `save_visit_plan` 거절 / 실패 | 실행 안 함 / `saved=False`, 재시도 없음 | 저장 없이 대화 계속 / 실패 안내 |

로깅: API 키, 마스킹 전 전화번호·주민번호는 로그에 남기지 않는다.

---

## B. 외부 API 계약

내부 필드명은 A.1의 `HospitalCandidate`와 A.2의 도구 반환 dict를 따른다. 파서는 XML 태그명을 상수로 분리하고, 없는 태그는 `None`으로 처리한다. 실제 응답은 2026-09-10에 강남구, 강릉, 영월, 역삼동, 경포대, 상동읍을 대상으로 녹화하여 `tests/fixtures/egen/`과 `tests/fixtures/kakao/`에 오류 케이스와 함께 보관한다.

### B.1 E-Gen 응급의료정보 API

- Base URL은 `http://apis.data.go.kr/B552657/ErmctInfoInqireService`이고, 인증은 쿼리 파라미터 `serviceKey`(`EGEN_SERVICE_KEY`)로 한다. 응답은 XML이며 공통 파라미터는 `pageNo`와 `numOfRows`다.
- 개발계정의 호출 한도는 하루 1,000건이다. 단위 테스트는 fixture를 사용하고, 실제 호출은 통합 테스트 5건으로 제한한다. 외상센터 API 3종은 Sprint 2에서 다룬다.

| 오퍼레이션 | 엔드포인트 | 파라미터 | 도구 |
|---|---|---|---|
| 위치정보 | `/getEgytLcinfoInqire` | `WGS84_LAT`, `WGS84_LON` | `list_nearby_ers` |
| 실시간 가용병상 | `/getEmrrmRltmUsefulSckbdInfoInqire` | `STAGE1`(시도), `STAGE2`(시군구) | `get_er_bed_status` |
| 중증질환 수용가능 | `/getSrsillDissAceptncPosblInfoInqire` | `STAGE1`, `STAGE2` | `get_severe_acceptance` |
| 기관 기본정보 | `/getEgytBassInfoInqire` | `HPID` | `get_er_detail` |

**공통 응답 봉투.** `response/header/resultCode`가 `00`이 아니면 오류로 처리하고, `resultMsg`는 로그에만 기록한다. `body/totalCount`가 0이면 빈 목록이며, `body/items/item`은 항목별 dict로 변환한다. `item` 태그가 없는 경우와 빈 문자열인 경우는 같은 빈 목록으로 정규화한다.

**B.1.3 위치정보 (`list_nearby_ers`).** `hpid`는 모든 도구의 조인 키로 그대로 사용한다. `dutyName`은 `name`, `dutyTel1`은 `er_tel`(병원 대표전화이며 응급실 전용 번호가 아니다), `latitude`와 `longitude`는 float로 변환하여 `lat`과 `lon`, 서버가 계산한 `distance`는 `distance_km`에 대응한다.

**B.1.4 가용병상 (`get_er_bed_status`).** `hvec`는 `er_beds_available`에 대응하며 음수이거나 비어 있으면 `None`으로 둔다. `hvidate`(`yyyyMMddHHmmss`)는 ISO 8601로 변환하여 `beds_updated_at`에 담고, 현재 시각과의 차이가 15분을 넘으면 `stale`을 `True`로 한다. 응답이 시도·시군구 단위이므로 `hpids`로 거르고, 같은 `(sido, sigungu)` 키의 결과를 60초 동안 캐시한다. 재시도에 실패하여 캐시를 대신 사용할 때는 `stale`을 반드시 `True`로 한다.

**B.1.5 중증질환 수용가능 (`get_severe_acceptance`).** 질환 파라미터 `SM_TYPE`(선택, 1~28)을 지정하면 해당 질환이 `Y`인 병원만 응답에 포함되므로 `acceptable=False`를 표현할 수 없다. 따라서 `SM_TYPE`은 생략하고 지역 전체를 받아 파서가 `mkioskty<N>` 컬럼을 선택한다. 값이 `Y`이면 `True`, `N`이면 `False`, 결측이거나 조회에 실패하면 `None`이며, 각각 `"yes"`, `"no"`, `"unknown"`으로 `accepts_condition`에 반영한다. 위급도가 `standard`이면 이 도구를 호출하지 않고 항상 `"unknown"`으로 둔다.

| `condition` | 사용 컬럼 |
|---|---|
| 심근경색 | `mkioskty1` ([재관류중재술] 심근경색) |
| 뇌출혈 | `mkioskty3`(거미막하출혈)과 `mkioskty4`(그 외 뇌출혈수술) 중 하나라도 `Y` |
| 뇌졸중 | `mkioskty2`(뇌경색 재관류). 실질적으로 뇌경색만 의미 |
| 중증외상 | 없음 (Sprint 2로 연기) |
| 화상 | `mkioskty19` ([중증화상] 전문치료) |
| 분만 | `mkioskty16` ([산부인과응급] 분만) |

`mkioskty`는 1번부터 28번까지 있으며 재관류, 뇌출혈수술, 대동맥, 담낭, 복부, 내시경, 저체중출생아, 산부인과, 화상, 사지접합, 투석, 정신과, 안과, 영상의학혈관중재, 응급실 gatekeeper를 포함한다. 위 표에 없는 코드는 사용하지 않는다. `mkioskty10`, `12`, `14`, `15`, `27`은 가능 연령을 담은 `…Msg` 텍스트를 함께 제공하므로, 필요해지면 `acceptable`과 별도로 보존한다. 뇌출혈과 뇌졸중은 사용하는 코드가 겹치지 않도록 위와 같이 구분하였다.

**B.1.6 기관 기본정보 (`get_er_detail`).** `dutyName`은 `name`, `dutyAddr`은 `address`, `dutyTel1`은 `er_tel`과 `main_tel`(두 필드는 같은 값이다)에 대응한다. `dutyTime<1~8>s`와 `dutyTime<1~8>c`는 요일별 시작·종료 시각이며(8은 공휴일) 하나의 문자열로 합쳐 `hours`에 담는다. 이 도구는 상위 3곳에만 호출하고, 목록 조회에 `er_tel`이 이미 있으면 생략한다.

**B.1.7 제약.** 데이터는 병원이 직접 입력하므로 실제 상황과 다를 수 있으며, 이를 응답 문구와 시스템 프롬프트에 명시한다. `hvidate`가 15분 이상 지났으면 '갱신 지연'으로 표시한다. 알려진 이상값은 음수이거나 빈 `hvec`, 그리고 누락된 `latitude`와 `longitude`이며, 좌표가 없는 기관은 후보에서 제외한다. 도구 결과에 포함된 병원명과 주소는 데이터로만 취급하고, 지시문이 섞여 있어도 따르지 않는다.

### B.2 카카오 로컬 API

- Base URL은 `https://dapi.kakao.com`이고, 인증은 헤더 `Authorization: KakaoAK {KAKAO_REST_API_KEY}`로 한다. 응답은 JSON이다. 주소 검색(`GET /v2/local/search/address.json`)을 먼저 사용하고, 결과가 없으면 키워드 검색(`GET /v2/local/search/keyword.json`)으로 넘어간다. 키워드 검색은 "경포대", "역삼역" 같은 표현을 처리한다.
- `geocode`의 매핑은 다음과 같다. `documents[].y`와 `x`는 float로 변환하여 `lat`과 `lon`에 담고, `road_address_name`(없으면 `address_name`)은 `address`에 담는다. `address.region_1depth_name`은 `sido`(E-Gen의 `STAGE1`), `region_2depth_name`은 `sigungu`(`STAGE2`)가 되며, `meta.total_count`가 0이면 `found=False`다.
- 주소 검색과 키워드 검색이 모두 0건이면 `found=False`를 반환하고 이후 도구를 호출하지 않는다. 키워드 검색 결과에 `address`가 없으면 `road_address`에서 시도와 시군구를 추출한다.
- 시도 표기는 카카오와 E-Gen이 다르다(예: `서울`과 `서울특별시`). 이를 `parser.py`의 `SIDO_ALIASES`로 정규화한다. 강원(2023년)과 전북(2024년)은 `특별자치도`로 명칭이 바뀌어서, `STAGE1`에 `강원도`를 쓰면 데이터가 있어도 결과가 0건이므로 이 표에 반영하였다.
- 429와 5xx 응답은 2회 재시도한다. 개인 위치는 로그에 남기지 않고 세션이 끝나면 폐기한다.

### B.3 복원력 정책 (`medical_api/resilience.py`)

타임아웃은 `ER_REQUEST_TIMEOUT`(기본 5초)으로 지정한다. 타임아웃, 5xx, 429 응답은 백오프 1초로 2회 재시도하며, 그 밖의 4xx와 스키마 오류는 재시도하지 않는다. 재시도에도 실패하면 60초 캐시(`stale=True`)를 사용하고, 캐시가 없으면 빈 결과를 반환한다.

---

## C. 화면(Streamlit) 연결 계약

화면은 의료기관 직접 조회, 에이전트 상담, 연결 상태의 세 탭으로 구성한다. 실제 API 응답과 runner 결과를 그대로 표시하며, 백엔드가 실패하더라도 원인을 숨기거나 다른 데이터로 대체하지 않는다.

```text
streamlit_app.py → web/app.py
  ├ 의료기관 조회 → web/api_view.py → 실제 의료 API
  ├ 에이전트 상담 → WebSession → RunnerBackend → ERFinder
  └ 연결 상태 → RunnerBackend.status()
```

`WebSession(backend=RunnerBackend())`는 브라우저 세션마다 하나씩 보관한다. 조회, 대화, 승인은 사용자의 입력과 버튼 이벤트에서만 실행하며, 화면을 다시 그리거나 연결 상태를 조회할 때는 새 검색을 시작하지 않는다. 의료기관 조회(목록과 상세)의 결과는 상담에서 이루어지는 병상 및 중증 수용 판정과 구분한다.

### C.1 화면↔runner 경계

| 호출 | 계약 |
|---|---|
| `status()` | `connected`, `checks`, `category`, `message`를 반환한다. 키 값은 포함하지 않으며 외부 요청을 보내지 않는다. |
| `connect(transport=...)` | 실제 provider, 모델, runner를 조립한다. import, 계약, 실행 단계의 오류는 화면용 상태로 변환한다. |
| `chat(text, *, transport, force_refresh=False)` | 입력을 runner에 전달한다. 강제 갱신은 마지막 검색을 다시 조회할 때만 사용한다. |
| `select_hospital(hpid)` | 현재 응답에 포함된 후보만 선택할 수 있다. 저장하지 않고 승인 대기 상태를 반환한다. |
| `approve(approved)` | 현재의 승인 대기를 한 번만 처리한다. 실패하거나 interrupt가 반복되면 이전 승인을 재사용하지 않는다. |
| `reset()` | 대화, 위치, 후보, 승인 대기를 초기화하고 프로필은 유지한다. |
| `forget()` | 사용자의 대화와 프로필을 제거하고, 화면에 남은 API 조회 기록도 삭제한다. |

`WebResult`는 `reply`, `pending_approval`, `note`를 가진다. `UIReply`와 `UIHospital`은 표시를 위한 변환일 뿐 근거 검증을 대신하지 않는다. 승인 대기는 선택한 후보의 ID 및 이름과 일치해야 한다.

### C.2 세션과 저장

브라우저마다 `user_id`와 Store를 분리하며, 기본 주소는 사용자가 동의한 경우에만 저장한다. 이동수단 변경이나 새 대화처럼 검색 문맥이 바뀌면 기존 후보와 승인 대기를 먼저 취소한 뒤 해당 이벤트를 처리한다. 따라서 승인과 설정 변경이 같은 이벤트로 들어오더라도 이전 계획은 저장되지 않는다. 저장 횟수는 승인 전 0회, 거절 후 0회, 승인 후 1회여야 하며, 결과를 확인하지 못한 경우에도 자동으로 재시도하여 중복 저장하지 않는다. 메모리에 저장하는 것은 병원 예약이나 접수가 아니며, 서버가 종료되면 사라진다.

### C.3 코드 책임과 백엔드 경계

| 경로 | 담당 |
|---|---|
| `web/app.py`, `web/components.py` | 세 탭, 결과 표시, 입력·승인·삭제 이벤트 |
| `web/api_view.py` | 명시적 의료 API 조회와 마지막 조회 결과 |
| `web/live.py` | 실제 provider·모델·runner 조립(`default_runner_factory`), 연결 상태 |
| `web/state.py`, `web/contracts.py` | 브라우저별 대화와 표시용 계약 |
| `scripts/verify.py` | 오프라인 테스트 실행과 JSON/JUnit/로그 기록 |

모델, 검색, 메모리, 가드레일의 오류는 해당 모듈에서 수정하며, 상담 어댑터는 실패를 표시하기만 한다. 좌표를 지정하면 주소 API를 건너뛰고, 기관 주소에서 얻은 시도와 시군구로 병상 및 중증 수용 API를 조회한다. 조회 실패와 정상적인 빈 응답을 구분하며, 실패한 조회는 같은 턴에서 반복하지 않는다. Gemini와 E-Gen은 필수이고 카카오는 주소 검색을 쓸 때만 필요하다. `scripts/run_local.py`는 프로젝트의 `.env`를 우선 적용하고, 일반 factory와 `streamlit run`은 기존 환경변수를 그대로 사용한다.

### C.4 확인해야 할 사용자 흐름

다음 항목은 `tests/integration/`과 `tests/unit/web/`에서 실제 runner, 메모리, LangGraph(interrupt와 resume 포함)로 확인한다.

- 브라우저별로 대화, 주소, 저장 기록이 섞이지 않는다.
- 기본 주소의 저장 동의와 철회가 반영된다.
- 저장 횟수가 승인 전 0회, 거절 후 0회, 승인 후 1회다.
- 이동수단 변경이나 새 대화가 승인 이벤트와 겹쳐도 이전 계획이 저장되지 않는다.
- 화면을 다시 그리는 것만으로 조회나 저장이 반복되지 않는다.
- 전체 삭제 후에는 상담 기록, 프로필, 의료기관 조회 결과가 남지 않는다.
- API 오류, 빈 응답, 누락 필드를 구분하여 표시하고 원문 예외나 키 값을 노출하지 않는다.
- 위치 재질문, 검색 반경, 병원 ID·전화·병상 근거의 일치, 위급 입력에 대한 119 안내가 올바르게 동작한다.
