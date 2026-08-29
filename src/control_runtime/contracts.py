from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ControlDomain(str, Enum):
    QOS = "qos"
    MOBILITY = "mobility"


class ObjectiveProfile(BaseModel):
    profile_name: str = Field(default="balanced")
    sla_violation_cost: float = Field(default=1.0, ge=0.0)
    mobility_risk_cost: float = Field(default=0.8, ge=0.0)
    control_churn_cost: float = Field(default=0.5, ge=0.0)
    resource_pressure_cost: float = Field(default=0.7, ge=0.0)
    fairness_cost: float = Field(default=0.3, ge=0.0)


__all__ = ["ControlDomain", "ObjectiveProfile"]
