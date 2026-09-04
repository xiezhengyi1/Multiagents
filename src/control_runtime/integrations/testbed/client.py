from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from .contracts import NativeActionReceipt, NativeSnapshot


@dataclass(frozen=True, slots=True)
class TestbedConfig:
    snapshot_base_url: str
    action_base_url: str
    subscriber_base_url: str
    run_id: str
    plmn_id: str
    action_api_key: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> TestbedConfig:
        return cls(
            snapshot_base_url=os.environ["TESTBED_SNAPSHOT_BASE_URL"].rstrip("/"),
            action_base_url=os.environ["TESTBED_ACTION_BASE_URL"].rstrip("/"),
            subscriber_base_url=os.environ["TESTBED_SUBSCRIBER_BASE_URL"].rstrip("/"),
            run_id=os.environ["TESTBED_RUN_ID"],
            plmn_id=os.environ["TESTBED_PLMN_ID"],
            action_api_key=os.environ["TESTBED_ACTION_API_KEY"],
            timeout_seconds=float(os.environ["TESTBED_HTTP_TIMEOUT_SECONDS"]),
        )


class TestbedClient:
    def __init__(self, *, config: TestbedConfig, http: httpx.Client) -> None:
        self.config = config
        self.http = http

    @classmethod
    def create(cls, config: TestbedConfig) -> TestbedClient:
        return cls(
            config=config,
            http=httpx.Client(timeout=config.timeout_seconds),
        )

    def latest_snapshot(self) -> dict[str, Any]:
        response = self.http.get(
            f"{self.config.snapshot_base_url}/v1/snapshots/latest",
            params={"run_id": self.config.run_id},
        )
        response.raise_for_status()
        return NativeSnapshot.model_validate(response.json()).model_dump(mode="json")

    def snapshot(self, snapshot_id: str) -> dict[str, Any]:
        response = self.http.get(
            f"{self.config.snapshot_base_url}/v1/snapshots/{quote(snapshot_id, safe='')}"
        )
        response.raise_for_status()
        return NativeSnapshot.model_validate(response.json()).model_dump(mode="json")

    def subscriber(self, supi: str) -> dict[str, Any]:
        response = self.http.get(
            f"{self.config.subscriber_base_url}/api/subscriber/"
            f"{quote(supi, safe='')}/{quote(self.config.plmn_id, safe='')}"
        )
        response.raise_for_status()
        return response.json()

    def dispatch_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(
            f"{self.config.action_base_url}/v1/actions/policy",
            headers={"X-API-Key": self.config.action_api_key},
            json=payload,
        )
        response.raise_for_status()
        return NativeActionReceipt.model_validate(response.json()).model_dump(mode="json")


__all__ = ["TestbedClient", "TestbedConfig"]
