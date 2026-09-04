from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import (
    ExecutionFeedbackPayload,
    HandoffPayload,
    ObservationPayload,
)

EntityType = Literal[
    "ue",
    "app",
    "flow",
    "session",
    "slice",
    "ran_node",
    "core_node",
    "policy_binding",
]
PolicyDomain = Literal["qos", "mobility"]
OperationKind = Literal["create", "update", "delete"]
PolicyResourceType = Literal[
    "SmPolicyDecision",
    "PcfAmPolicyControlPolicyAssociation",
    "UrspRuleRequest",
]
PolicyPayload = PcfAmPolicyControlPolicyAssociation | UrspRuleRequest | SmPolicyDecision
ReceiptStatus = Literal["success", "failed"]
ExecutionStatus = Literal["success", "partial_success", "failed"]


class SnapshotEntity(ContractModel):
    entity_id: NonEmptyStr
    entity_type: EntityType
    attributes: dict[str, JsonValue]


class SnapshotRelation(ContractModel):
    relation_id: NonEmptyStr
    relation_type: NonEmptyStr
    source_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class SnapshotMetric(ContractModel):
    owner_entity_id: NonEmptyStr
    metric_name: NonEmptyStr
    value: JsonValue
    observed_at: datetime


class EnvironmentSnapshot(ObservationPayload):
    snapshot_id: NonEmptyStr
    observed_at: datetime
    entities: list[SnapshotEntity]
    relations: list[SnapshotRelation]
    metrics: list[SnapshotMetric]

    @model_validator(mode="after")
    def validate_graph(self) -> EnvironmentSnapshot:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("snapshot entity_id values must be unique")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("snapshot relation_id values must be unique")
        metric_keys = [
            (metric.owner_entity_id, metric.metric_name) for metric in self.metrics
        ]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("snapshot owner/metric keys must be unique")
        known = set(entity_ids)
        for relation in self.relations:
            if (
                relation.source_entity_id not in known
                or relation.target_entity_id not in known
            ):
                raise ValueError(
                    f"relation {relation.relation_id} references an unknown entity"
                )
        unknown_metric_owners = sorted(
            {
                metric.owner_entity_id
                for metric in self.metrics
                if metric.owner_entity_id not in known
            }
        )
        if unknown_metric_owners:
            raise TypeError(
                "snapshot metrics reference unknown entities: "
                + ", ".join(unknown_metric_owners)
            )
        return self


class ObservationRequest(ContractModel):
    query_name: NonEmptyStr
    arguments: dict[str, JsonValue]


class EnvironmentObservation(ObservationPayload):
    base_snapshot_id: NonEmptyStr
    observed_at: datetime
    query: ObservationRequest
    result: JsonValue


class ExecutionOperation(ContractModel):
    operation_id: NonEmptyStr
    domain: PolicyDomain
    operation: OperationKind
    resource_type: PolicyResourceType
    resource_id: NonEmptyStr
    payload: PolicyPayload | None

    @model_validator(mode="after")
    def validate_payload(self) -> ExecutionOperation:
        if self.operation == "delete":
            if self.payload is not None:
                raise ValueError("delete operation cannot contain a policy payload")
            return self
        if self.payload is None:
            raise ValueError("create and update operations require a policy payload")
        expected = {
            "SmPolicyDecision": SmPolicyDecision,
            "PcfAmPolicyControlPolicyAssociation": PcfAmPolicyControlPolicyAssociation,
            "UrspRuleRequest": UrspRuleRequest,
        }[self.resource_type]
        if not isinstance(self.payload, expected):
            raise TypeError(
                f"resource_type {self.resource_type} requires payload {expected.__name__}"
            )
        if isinstance(self.payload, SmPolicyDecision) and not (
            self.payload.pccRules or self.payload.qosDecs
        ):
            raise ValueError("SmPolicyDecision requires pccRules or qosDecs")
        return self


class ExecutionAction(HandoffPayload):
    operations: list[ExecutionOperation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operations(self) -> ExecutionAction:
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("execution operation_id values must be unique")
        return self


class OperationReceipt(ContractModel):
    operation_id: NonEmptyStr
    status: ReceiptStatus
    result: JsonValue | None = None
    error: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_status(self) -> OperationReceipt:
        if self.status == "success" and self.error is not None:
            raise ValueError("successful receipt cannot contain error")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed receipt requires error")
        return self


class ExecutionFeedback(ExecutionFeedbackPayload):
    status: ExecutionStatus
    receipts: list[OperationReceipt] = Field(min_length=1)
    snapshot: EnvironmentSnapshot
    objective_metrics: dict[NonEmptyStr, float]

    @model_validator(mode="after")
    def validate_receipts(self) -> ExecutionFeedback:
        operation_ids = [receipt.operation_id for receipt in self.receipts]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("execution receipt operation_id values must be unique")
        succeeded = sum(receipt.status == "success" for receipt in self.receipts)
        failed = len(self.receipts) - succeeded
        expected = (
            "success"
            if failed == 0
            else "failed"
            if succeeded == 0
            else "partial_success"
        )
        if self.status != expected:
            raise ValueError(f"execution status must be {expected} for its receipts")
        return self


__all__ = [
    "EntityType",
    "EnvironmentObservation",
    "EnvironmentSnapshot",
    "ExecutionAction",
    "ExecutionFeedback",
    "ExecutionOperation",
    "ExecutionStatus",
    "ObservationRequest",
    "OperationKind",
    "OperationReceipt",
    "PolicyDomain",
    "PolicyPayload",
    "PolicyResourceType",
    "ReceiptStatus",
    "SnapshotEntity",
    "SnapshotMetric",
    "SnapshotRelation",
]
