from __future__ import annotations

import httpx
from openai import BadRequestError

from shared.agents import retryable_transport_error


def test_retries_only_the_vllm_tool_schema_eof() -> None:
    request = httpx.Request("POST", "http://vllm.local/v1/chat/completions")
    response = httpx.Response(400, request=request)
    error = BadRequestError(
        "Invalid JSON: EOF while parsing a string", response=response, body=None
    )

    assert retryable_transport_error(error)


def test_does_not_retry_another_bad_request() -> None:
    request = httpx.Request("POST", "http://vllm.local/v1/chat/completions")
    response = httpx.Response(400, request=request)
    error = BadRequestError(
        "invalid tool contract", response=response, body=None
    )

    assert not retryable_transport_error(error)
