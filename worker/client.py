"""HTTP client for worker ↔ orchestrator communication."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from worker.protocol.submission import WorkerNextResponse, WorkerSubmitRequest

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (1.0, 2.0, 4.0)
_DEFAULT_TIMEOUT = 60.0


class OrchestratorError(RuntimeError):
    """Orchestrator request failed after retries."""


def _orch_base_url() -> str:
    host = os.environ.get("ORCH_HOST", "127.0.0.1")
    port = os.environ.get("ORCH_PORT", "8899")
    return f"http://{host}:{port}"


async def _get_with_retry(
    full_url: str,
    *,
    client: httpx.AsyncClient,
    timeout: float,
    indefinite: bool = False,
) -> WorkerNextResponse:
    delay_idx = 0
    last_exc: Exception | None = None
    while True:
        attempt = delay_idx + 1
        try:
            resp = await client.get(full_url, timeout=timeout)
            if resp.status_code == 503:
                raise OrchestratorError(f"orchestrator not ready: {full_url}")
            if resp.status_code >= 400:
                raise OrchestratorError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return WorkerNextResponse.model_validate(resp.json())
        except (httpx.RequestError, httpx.TimeoutException, OrchestratorError) as e:
            last_exc = e
            delay = _RETRY_DELAYS[min(delay_idx, len(_RETRY_DELAYS) - 1)]
            logger.warning(
                "orchestrator GET attempt %d to %s failed: %s; retry in %.1fs",
                attempt, full_url, e, delay,
            )
            await asyncio.sleep(delay)
            delay_idx += 1
            if not indefinite and delay_idx >= len(_RETRY_DELAYS):
                raise OrchestratorError(f"all retries failed: {last_exc}") from last_exc


async def _post_with_retry(
    full_url: str,
    payload: dict,
    *,
    client: httpx.AsyncClient,
    timeout: float,
) -> None:
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            resp = await client.post(full_url, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise OrchestratorError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return
        except (httpx.RequestError, httpx.TimeoutException, OrchestratorError) as e:
            last_exc = e
            logger.warning(
                "orchestrator POST attempt %d to %s failed: %s",
                attempt, full_url, e,
            )
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
    raise OrchestratorError(f"all retries failed: {last_exc}") from last_exc


async def fetch_next_assignment(
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> WorkerNextResponse:
    """GET /worker/next — blocks with retry until orchestrator responds."""
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        return await _get_with_retry(
            f"{_orch_base_url()}/worker/next",
            client=cli,
            timeout=timeout,
            indefinite=True,
        )
    finally:
        if own_client:
            await cli.aclose()


async def submit_rollouts(
    request: WorkerSubmitRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> None:
    """POST /worker/submit with unsigned rollout batch."""
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        await _post_with_retry(
            f"{_orch_base_url()}/worker/submit",
            request.model_dump(mode="json"),
            client=cli,
            timeout=timeout,
        )
    finally:
        if own_client:
            await cli.aclose()
