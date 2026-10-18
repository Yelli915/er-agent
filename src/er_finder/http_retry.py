"""API 호출 재시도 공통 유틸.

E-Gen(`medical_api`)과 카카오(`search`) 둘 다 이 모듈을 쓴다. 어느 한쪽 API에만
속한 게 아니라서 `medical_api` 밑이 아니라 여기(최상위)에 둔다.

- 타임아웃: 기본 5초
- 재시도 대상: 연결 실패, 타임아웃, 5xx, 429 (최대 2회, 매번 1초 대기)
- 재시도 제외: 그 외 4xx 등 - 재시도해도 똑같이 실패할 오류라 바로 예외
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

SAFE_TIMEOUT = 5.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF = 1.0


def is_retryable(error: httpx.HTTPError) -> bool:
    """연결 실패, 타임아웃, 5xx, 429만 재시도 대상으로 본다."""
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 429 or status_code >= 500
    return False


def request_with_transport_retries(
    client: httpx.Client | None,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    timeout: float = SAFE_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """요청을 보내고 응답을 돌려준다.

    연결 실패·타임아웃·5xx·429가 나면 최대 max_retries번 다시 시도한다.
    시도 사이에 backoff초 대기한다.
    그 외 오류는 재시도하지 않고 바로 예외를 던진다.
    client를 넘기면 그 클라이언트로 보내고, `None`이면 매번 새로 연결해서 보낸다.
    """
    send = client.request if client is not None else httpx.request

    last_error: httpx.HTTPError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = send(method, url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            if not is_retryable(error):
                raise
            if attempt < max_retries:
                sleep(backoff)

    raise last_error  # type: ignore[misc]
