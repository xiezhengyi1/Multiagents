from __future__ import annotations

import httpx
from langgraph.types import RetryPolicy
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError

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

def retryable_transport_error(error: Exception) -> bool:
    return isinstance(error, TRANSPORT_ERRORS) or (
        isinstance(error, BadRequestError)
        and "Invalid JSON: EOF while parsing a string" in str(error)
    )


    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    return RetryPolicy(
        max_attempts=max_attempts,
        retry_on=retryable_transport_error,
        jitter=True,
    )


__all__ = ["TRANSPORT_ERRORS", "retryable_transport_error", "transport_retry_policy"]
