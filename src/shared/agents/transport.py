from __future__ import annotations

import httpx
from langgraph.types import RetryPolicy
from openai import APIConnectionError, APITimeoutError, RateLimitError

TRANSPORT_ERRORS = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)


def transport_retry_policy(*, max_attempts: int) -> RetryPolicy:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    return RetryPolicy(
        max_attempts=max_attempts,
        retry_on=TRANSPORT_ERRORS,
        jitter=True,
    )


__all__ = ["TRANSPORT_ERRORS", "transport_retry_policy"]
