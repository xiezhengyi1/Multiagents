from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, field_validator, model_validator

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import ValidationIssue


class RepairLimits(ContractModel):
    local_limits: dict[NonEmptyStr, int]
    upstream: dict[NonEmptyStr, NonEmptyStr]
    total_limit: int = Field(ge=1)
    controller_actor: NonEmptyStr

    @field_validator("local_limits")
    @classmethod
    def validate_local_limits(cls, values: dict[str, int]) -> dict[str, int]:
        if not values:
            raise ValueError("local_limits cannot be empty")
        if any(isinstance(value, bool) or value < 1 for value in values.values()):
            raise ValueError("every local repair limit must be a positive integer")
        return values

    @model_validator(mode="after")
    def validate_upstream(self) -> RepairLimits:
        if set(self.upstream) != set(self.local_limits):
            raise ValueError(
                "upstream must define one escalation target per local actor"
            )
        return self


@dataclass(frozen=True, slots=True)
class RepairRoute:
    target_actor: str
    attempt: int
    max_attempts: int
    issues: tuple[ValidationIssue, ...]


class RepairBudget:
    def __init__(self, limits: RepairLimits) -> None:
        self.limits = limits
        self._local_attempts = {actor: 0 for actor in limits.local_limits}
        self._total_attempts = 0

    @property
    def total_attempts(self) -> int:
        return self._total_attempts

    def request(
        self,
        *,
        owner_actor: str,
        issues: list[ValidationIssue],
    ) -> RepairRoute:
        if owner_actor not in self._local_attempts:
            raise KeyError(f"no repair budget configured for actor {owner_actor}")
        self._total_attempts += 1
        self._local_attempts[owner_actor] += 1
        local_attempt = self._local_attempts[owner_actor]
        local_limit = self.limits.local_limits[owner_actor]
        routed_issues = list(issues)

        if self._total_attempts > self.limits.total_limit:
            routed_issues.append(
                ValidationIssue(
                    code="episode_repair_budget_exhausted",
                    message="The episode exhausted its total repair budget",
                )
            )
            return RepairRoute(
                target_actor=self.limits.controller_actor,
                attempt=local_limit,
                max_attempts=local_limit,
                issues=tuple(routed_issues),
            )
        if local_attempt > local_limit:
            routed_issues.append(
                ValidationIssue(
                    code=f"{owner_actor}_repair_budget_exhausted",
                    message=f"{owner_actor} exhausted its local repair budget",
                )
            )
            return RepairRoute(
                target_actor=self.limits.upstream[owner_actor],
                attempt=local_limit,
                max_attempts=local_limit,
                issues=tuple(routed_issues),
            )
        return RepairRoute(
            target_actor=owner_actor,
            attempt=local_attempt,
            max_attempts=local_limit,
            issues=tuple(routed_issues),
        )

    def resolved(self, actor: str) -> None:
        if actor not in self._local_attempts:
            raise KeyError(f"no repair budget configured for actor {actor}")
        self._local_attempts[actor] = 0


__all__ = ["RepairBudget", "RepairLimits", "RepairRoute"]
