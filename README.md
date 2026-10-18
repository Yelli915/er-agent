# ER Finder · skala-er-agent

카카오·E-Gen·OpenAI API를 연결하는 응급의료기관 조회와 에이전트 상담 화면입니다. 의료기관 직접 조회와 에이전트 상담은 별도 경로로 실행됩니다. 화면에 표시할 기관 정보는 실제 API 응답을 사용합니다.

`main`의 `00971aa`를 기준으로 실제 메모리·LangGraph·의료 API 연결을 수정했습니다. 좌표로 직접 조회하면 카카오 없이 사용할 수 있습니다. 최신 실행 결과는 [검증 기록](docs/verification.md)에 기록합니다.

## 실행

Python 3.11 이상이 필요합니다. 프로젝트 루트에서 실행합니다.

```bash
uv sync --frozen --extra dev
uv run --frozen python scripts/run_local.py
```

로컬 확인 주소는 **http://localhost:8503**입니다. 이미 설치된 환경에서는 다음 명령으로 실행합니다.

```bash
.venv/bin/python scripts/run_local.py
```

이 실행기는 프로젝트 `.env`를 명시적으로 적용하므로 다른 작업에서 상속된 API 키가 우선하지 않습니다. 포트는 `--port 8504`처럼 바꿀 수 있습니다. 일반 `streamlit run` 또는 `er-finder` 실행은 기존 환경변수 우선순위를 유지합니다.

기존 `.env` 또는 실행 환경에 다음 키를 설정합니다. 필요한 항목은 [.env.example](.env.example)을 참고하세요. 화면은 키의 설정 여부만 표시하며 키 값을 출력하지 않습니다.

```dotenv
OPENAI_API_KEY=...
KAKAO_REST_API_KEY=... # 주소 검색을 사용할 때만 필요
EGEN_SERVICE_KEY=...
```

카카오 없이 상담하려면 왼쪽 **주소 검색 없이 좌표로 조회**를 체크하고 위도·경도를 지정한 뒤 **에이전트 연결 확인**을 누릅니다. **에이전트 상담** 탭에 증상을 입력합니다. 입력된 좌표가 실제 조회할 위치인지 확인하세요.

## 사용 흐름

| 탭 | 동작 |
|---|---|
| 의료기관 조회 | 좌표와 반경을 입력하고 조회하면 주변 기관 정보를 요청합니다. 기관명·거리·대표전화 등을 확인하고 기관을 선택해 상세 정보를 조회합니다. |
| 에이전트 상담 | 연결 확인 후 위치와 증상을 입력합니다. runner가 반환한 후보 선택과 방문 계획 승인·거절을 전달합니다. 연결 오류는 해당 상태로 표시합니다. |
| 연결 상태 | OpenAI·카카오·E-Gen 키 설정 여부와 에이전트 연결 결과를 확인합니다. 설정됨은 API 인증 성공을 의미하지 않습니다. |

기관 목록은 병상이나 진료 수용 가능 여부의 확정 결과가 아닙니다. 누락된 정보는 확인 불가로 표시하며 방문 전에 병원 대표전화로 확인해야 합니다. 위급한 상황에서는 즉시 119에 연락하세요.

화면 재실행만으로 API를 다시 호출하지 않습니다. 새 대화는 대화와 승인 대기를 초기화하고, 기록 전체 삭제는 해당 브라우저의 저장 정보와 조회 기록도 제거합니다. 주소는 동의 후 저장하며, 방문 계획 저장은 병원 예약이나 접수가 아닙니다. 저장소는 서버 메모리이므로 영구 보관을 제공하지 않습니다.

## 검증

```bash
uv run --frozen python scripts/verify.py
uv run --frozen python scripts/verify.py --require-integration
```

검증 도구는 live 테스트를 제외하고 Python 소켓의 외부 통신을 차단합니다. 오프라인 검증은 실제 메모리·검색·LangGraph를 실행하며 모델과 HTTP 응답만 테스트 자료로 대체합니다. 별도의 화면 계약 테스트에는 주입한 runner를 사용합니다. 실제 키로 실행한 결과는 별도 로컬 보고서로 구분합니다. 수집 오류가 남으면 실행 가능한 테스트를 계속 기록하되 전체 결과는 FAIL입니다.

실제 API 실측과 품질 측정은 별도 명령이며 비용이 발생하므로 명시적으로만 실행합니다.

```bash
python scripts/benchmark.py --live --runs 20 --prices 0.25,2,0.15,0.6 --output docs/benchmarks/baseline.json
python scripts/benchmark.py --check docs/benchmarks/after.json     # 커밋 전 키·필수 항목 점검
python scripts/benchmark.py --compare docs/benchmarks/baseline.json docs/benchmarks/after.json
python scripts/eval_golden.py [--live]      # 위급도 골든셋(tests/fixtures/golden/triage.json)
ER_RUN_LIVE=1 pytest tests/live -m live     # 실제 E-Gen·OpenAI 계약 점검(매일 CI에서도 실행)
```

`--prices`는 예시 값이 아니라 사용 중인 모델의 실제 단가(USD/1M 토큰)로 바꿔야 합니다. Windows에서는 `PYTHONUTF8=1`을 설정하세요.

`verification-results/report.json`, `junit.xml`, `pytest.log`에 실행 근거를 저장합니다. 전체 연결을 검증하지 못하면 `integration.status`는 BLOCKED 또는 NOT_VERIFIED이며, 파일이 존재하거나 단위 테스트가 통과한 것만으로 통합 PASS를 표시하지 않습니다. 추천 정확도는 미측정이며, 지연 시간의 단일 실행 실측은 [검증 기록](docs/verification.md)을 따릅니다.

## 담당과 문서

| 담당 | 경로 |
|---|---|
| 1 에이전트·공통 모델 | `agent/`, `models.py` |
| 2 의료 API | `medical_api/` |
| 3 위치·검색 | `search/` |
| 4 메모리 | `memory/` |
| 5 가드레일 | `guardrails/` |
| 6 화면·검증 | `web/`, `cli/`, `scripts/verify.py`, 화면 테스트 |

| 문서 | 내용 |
|---|---|
| [설계·계약](docs/design.md) | A 공통 인터페이스(데이터 모델·도구·Context·미들웨어·오류 처리), B E-Gen·카카오 API 매핑, C 화면↔runner 계약 |
| [검증](docs/verification.md) | 실행 기록, 검증 계획(성공 기준·시나리오·안전), 벤치마크 절차 |

CI는 `Web and verification tooling`(6번)과 `All modules integration`(전체)을 각각 실행합니다. 6번 통과를 전체 서비스 통과로 해석하지 않습니다. 로컬에서 담당 범위만 실행하려면 `python scripts/verify.py --scope web`을 사용합니다. 기본 실행은 전체 범위입니다. Windows에서도 `verify.py`가 동작하며(같은 프로세스의 loopback 연결만 허용), 외부 통신은 차단합니다.
