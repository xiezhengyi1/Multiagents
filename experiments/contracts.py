from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import ContractModel, NonEmptyStr
from shared.contracts.coordination import TerminalPayload
from training import EpisodeMetrics

ExperimentMethod = Literal[
    "multi_agent_control",
    "single_agent_control",
    "environment",
]


class ExperimentCase(ContractModel):
    case_id: NonEmptyStr
    episode_id: NonEmptyStr
    method: ExperimentMethod
    user_input: NonEmptyStr | None = None
    environment_request: dict[str, JsonValue] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input(self) -> ExperimentCase:
        if self.method == "environment":
            if self.environment_request is None or self.user_input is not None:
                raise ValueError(
                    "environment case requires environment_request and no user_input"
                )
        elif self.user_input is None or self.environment_request is not None:
            raise ValueError(
                "control case requires user_input and no environment_request"
            )
        return self


class ExperimentResult(ContractModel):
    case_id: NonEmptyStr
    episode_id: NonEmptyStr
    method: ExperimentMethod
    episode_directory: NonEmptyStr
    terminal: TerminalPayload
    metrics: EpisodeMetrics


__all__ = ["ExperimentCase", "ExperimentMethod", "ExperimentResult"]
