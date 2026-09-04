from __future__ import annotations

import re
from typing import Any

import httpx

from control_runtime.simulation import (
    EnvironmentObservation,
    EnvironmentSnapshot,
    ExecutionAction,
    ExecutionFeedback,
    ObservationRequest,
    OperationReceipt,
)
from shared.contracts import utc_now

from .client import TestbedClient, TestbedConfig
from .projection import (
    build_am_context,
    build_flow_catalog,
    build_optimizer_preview,
    project_environment_snapshot,
    project_subscription,
)

OBSERVATION_QUERIES = frozenset(
    {
        "get_sm_ue_flow_catalog",
        "search_sm_flow_targets",
        "get_am_policy_context",
        "search_am_policy_targets",
        "get_ue_slice_subscription",
        "preview_qos_optimizer",
        "inspect_mobility_ue_policies",
    }
)


def _parse_mbps(value: str) -> float:
    amount, unit = value.split()
    if unit != "Mbps":
        raise ValueError(f"testbed policy projection requires Mbps, got {unit}")
    return float(amount)


def _search_text(value: object) -> str:
    return " ".join(
        part for part in re.split(r"[^0-9a-z]+", str(value).casefold()) if part
    )


class TestbedSimulatorAdapter:
    def __init__(self, *, client: TestbedClient) -> None:
        self.client = client
        self._native_snapshot: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> TestbedSimulatorAdapter:
        return cls(client=TestbedClient.create(TestbedConfig.from_env()))

    def capture_snapshot(self) -> EnvironmentSnapshot:
        self._native_snapshot = self.client.latest_snapshot()
        return project_environment_snapshot(self._native_snapshot)

    def _native(self) -> dict[str, Any]:
        if self._native_snapshot is None:
            raise RuntimeError("capture_snapshot must run before observe or execute")
        return self._native_snapshot

    def _subscription(self, supi: str) -> dict[str, Any]:
        return project_subscription(
            supi=supi,
            plmn_id=self.client.config.plmn_id,
            subscriber=self.client.subscriber(supi),
        )

    def observe(self, request: ObservationRequest) -> EnvironmentObservation:
        if request.query_name not in OBSERVATION_QUERIES:
            raise KeyError(request.query_name)
        native = self._native()
        supi = str(request.arguments["supi"])
        handlers = {
            "get_sm_ue_flow_catalog": lambda: build_flow_catalog(native, supi),
            "search_sm_flow_targets": lambda: self._search_sm(
                native, supi, str(request.arguments["query"])
            ),
            "get_am_policy_context": lambda: build_am_context(
                native, self._subscription(supi), supi
            ),
            "search_am_policy_targets": lambda: self._search_am(
                native, supi, str(request.arguments["query"])
            ),
            "get_ue_slice_subscription": lambda: self._subscription(supi),
            "preview_qos_optimizer": lambda: build_optimizer_preview(
                native,
                supi,
                [str(item) for item in request.arguments["flow_ids"]],
            ),
            "inspect_mobility_ue_policies": lambda: {
                **build_am_context(native, self._subscription(supi), supi),
                "policies": [],
            },
        }
        return EnvironmentObservation(
            base_snapshot_id=native["snapshot_id"],
            observed_at=utc_now(),
            query=request,
            result=handlers[request.query_name](),
        )

    @staticmethod
    def _search_sm(native: dict[str, Any], supi: str, query: str) -> dict[str, Any]:
        catalog = build_flow_catalog(native, supi)
        needle = _search_text(query)
        needle_terms = frozenset(needle.split())
        matches = [
            flow
            for flow in catalog["flow_catalog"]
            if (
                needle
                in (
                    searchable := _search_text(
                        " ".join(
                            str(flow[field])
                            for field in ("id", "name", "app_id", "app_name")
                        )
                    )
                )
                or needle_terms <= frozenset(searchable.split())
            )
        ]
        return {
            "supi": supi,
            "query": query,
            "matches": matches,
            "source_snapshot_id": native["snapshot_id"],
        }

    def _search_am(
        self, native: dict[str, Any], supi: str, query: str
    ) -> dict[str, Any]:
        context = build_am_context(native, self._subscription(supi), supi)
        searchable = " ".join(
            [
                supi,
                context["serving_gnb"],
                *context["allowed_snssais"],
                *context["target_snssais"],
            ]
        ).casefold()
        return {
            "supi": supi,
            "query": query,
            "matches": [context] if query.casefold() in searchable else [],
            "source_snapshot_id": native["snapshot_id"],
        }

    def execute(self, action: ExecutionAction) -> ExecutionFeedback:
        native = self._native()
        receipts: list[OperationReceipt] = []
        current_snapshot_id = native["snapshot_id"]
        for operation in action.operations:
            if operation.resource_type != "SmPolicyDecision":
                receipts.append(
                    OperationReceipt(
                        operation_id=operation.operation_id,
                        status="failed",
                        error=(
                            f"testbed gateway does not support {operation.resource_type}"
                        ),
                    )
                )
                continue
            if operation.operation != "update":
                receipts.append(
                    OperationReceipt(
                        operation_id=operation.operation_id,
                        status="failed",
                        error="testbed gateway supports only SM policy updates",
                    )
                )
                continue
            try:
                payload = self._project_sm_policy_action(
                    operation.payload,
                    snapshot_id=current_snapshot_id,
                )
                result = self.client.dispatch_policy(payload)
                if result["status"] != "verified":
                    raise RuntimeError(
                        f"testbed action {result['action_id']} ended as {result['status']}"
                    )
                current_snapshot_id = result["snapshot_after"]
                receipts.append(
                    OperationReceipt(
                        operation_id=operation.operation_id,
                        status="success",
                        result=result,
                    )
                )
            except (
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                receipts.append(
                    OperationReceipt(
                        operation_id=operation.operation_id,
                        status="failed",
                        error=str(error),
                    )
                )

        self._native_snapshot = self.client.snapshot(current_snapshot_id)
        projected = project_environment_snapshot(self._native_snapshot)
        succeeded = sum(receipt.status == "success" for receipt in receipts)
        status = (
            "success"
            if succeeded == len(receipts)
            else "failed"
            if succeeded == 0
            else "partial_success"
        )
        objective_metrics = {
            f"{metric.owner_entity_id}.{metric.metric_name}": float(metric.value)
            for metric in projected.metrics
            if isinstance(metric.value, (int, float))
            and not isinstance(metric.value, bool)
        }
        return ExecutionFeedback(
            status=status,
            receipts=receipts,
            snapshot=projected,
            objective_metrics=objective_metrics,
        )

    def _project_sm_policy_action(
        self, policy: Any, *, snapshot_id: str
    ) -> dict[str, Any]:
        pcc_rules = policy.pccRules
        qos_decs = policy.qosDecs
        if pcc_rules is None or len(pcc_rules) != 1:
            raise ValueError("testbed policy action requires one PCC rule")
        if qos_decs is None or len(qos_decs) != 1:
            raise ValueError("testbed policy action requires one QoS decision")
        pcc_rule = next(iter(pcc_rules.values()))
        qos_data = next(iter(qos_decs.values()))
        if (
            qos_data.gbrUl is None
            or qos_data.maxbrUl is None
            or qos_data.gbrDl is None
            or qos_data.maxbrDl is None
        ):
            raise ValueError(
                "testbed policy action requires gbrUl, maxbrUl, gbrDl, and maxbrDl"
            )
        flow_id = pcc_rule.pccRuleId.removeprefix("pcc-")
        native_flow = next(
            flow for flow in self._native()["flows"] if flow["id"] == flow_id
        )
        return {
            "run_id": self.client.config.run_id,
            "snapshot_id": snapshot_id,
            "action_type": "sm_policy_update",
            "target": {
                "supi": native_flow["supi"],
                "flow_id": flow_id,
            },
            "parameters": {
                "five_qi": native_flow["session"]["five_qi"],
                "gbr_ul_mbps": _parse_mbps(qos_data.gbrUl),
                "mbr_ul_mbps": _parse_mbps(qos_data.maxbrUl),
                "gbr_dl_mbps": _parse_mbps(qos_data.gbrDl),
                "mbr_dl_mbps": _parse_mbps(qos_data.maxbrDl),
            },
        }


__all__ = ["OBSERVATION_QUERIES", "TestbedSimulatorAdapter"]
