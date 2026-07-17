from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .base import BaseProjector


class GenericProjector(BaseProjector):
    model = None

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        from .base import json_mapping, without_empty_values

        return without_empty_values(json_mapping(instance))


class ProjectorRegistry:
    _by_model: dict[type[BaseModel], type[BaseProjector]] = {}

    @classmethod
    def for_model(cls, model: type[BaseModel]) -> type[BaseProjector]:
        if not cls._by_model:
            cls._discover()
        return cls._by_model.get(model, GenericProjector)

    @classmethod
    def for_instance(cls, instance: Any) -> type[BaseProjector]:
        return cls.for_model(type(instance))

    @classmethod
    def _discover(cls) -> None:
        from .flow_selector import FlowSelectorProjector
        from .global_intent import GlobalControlIntentProjector
        from .grounding_decision import GroundingDecisionProjector
        from .planning_context import PlanningContextProjector
        from .policy_plan import PolicyPlanDraftProjector

        for projector in (
            FlowSelectorProjector,
            GlobalControlIntentProjector,
            GroundingDecisionProjector,
            PlanningContextProjector,
            PolicyPlanDraftProjector,
        ):
            if projector.model is not None:
                cls._by_model[projector.model] = projector
