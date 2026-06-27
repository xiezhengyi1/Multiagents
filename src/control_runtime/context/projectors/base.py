from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel


@dataclass(frozen=True)
class FieldSpec:
    name: str
    doc: str = ""
    priority: int = 100


@dataclass(frozen=True)
class ExcludeSpec:
    name: str
    reason: str


def field(name: str, *, doc: str = "", priority: int = 100) -> FieldSpec:
    return FieldSpec(name=name, doc=doc, priority=priority)


def exclude(name: str, *, reason: str) -> ExcludeSpec:
    return ExcludeSpec(name=name, reason=reason)


def json_mapping(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return dict(payload) if isinstance(payload, dict) else {}


def without_empty_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, [], {})
    }


class BaseProjector:
    """Declarative model-to-context projector with import-time field checks."""

    model: ClassVar[type[BaseModel] | None] = None
    visible: ClassVar[tuple[FieldSpec, ...]] = ()
    excluded: ClassVar[tuple[ExcludeSpec, ...]] = ()
    nested: ClassVar[dict[str, type["BaseProjector"]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        model = getattr(cls, "model", None)
        if model is None:
            return
        model_fields = set(getattr(model, "model_fields", {}).keys())
        if not model_fields:
            raise TypeError(f"{cls.__name__}: model must be a Pydantic BaseModel type")
        for spec in getattr(cls, "visible", ()):
            if spec.name not in model_fields:
                raise TypeError(
                    f"{cls.__name__}: visible field '{spec.name}' does not exist on "
                    f"{model.__name__}. Available: {sorted(model_fields)}"
                )
        for spec in getattr(cls, "excluded", ()):
            if spec.name not in model_fields:
                raise TypeError(
                    f"{cls.__name__}: excluded field '{spec.name}' does not exist on "
                    f"{model.__name__}. Available: {sorted(model_fields)}"
                )

    @classmethod
    def project(cls, instance: Any) -> dict[str, Any]:
        raw = json_mapping(instance)
        projected: dict[str, Any] = {}
        for spec in cls.visible:
            value = raw.get(spec.name)
            nested_projector = cls.nested.get(spec.name)
            if nested_projector is not None:
                value = cls._project_nested(value, nested_projector)
            projected[spec.name] = value
        return without_empty_values(projected)

    @staticmethod
    def _project_nested(value: Any, projector: type["BaseProjector"]) -> Any:
        if isinstance(value, list):
            return [
                projected
                for item in value
                if (projected := projector.project(item))
            ]
        if isinstance(value, dict) or hasattr(value, "model_dump"):
            return projector.project(value)
        return value
