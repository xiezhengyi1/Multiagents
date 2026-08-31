from __future__ import annotations

import hashlib
import json
from typing import Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import Field, JsonValue, model_validator

from .common import CONTRACT_VERSION, ContractModel, ContractVersion, NonEmptyStr


ToolSideEffect = Literal["read", "write", "external_action"]
ToolResultFormat = Literal["json", "text"]


def _fingerprint(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ToolContract(ContractModel):
    name: NonEmptyStr
    description: NonEmptyStr
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = None
    capabilities: list[NonEmptyStr] = Field(default_factory=list)
    side_effect: ToolSideEffect
    result_format: ToolResultFormat

    @model_validator(mode="after")
    def validate_schema_and_capabilities(self) -> "ToolContract":
        if self.input_schema.get("type") != "object":
            raise ValueError(f"tool {self.name} input_schema.type must be object")
        for schema_name, schema in (
            ("input_schema", self.input_schema),
            ("output_schema", self.output_schema),
        ):
            if schema is None:
                continue
            try:
                validator_for(schema).check_schema(schema)
            except SchemaError as exc:
                raise ValueError(
                    f"tool {self.name} {schema_name} is invalid: {exc.message}"
                ) from exc
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError(f"tool {self.name} capabilities must be unique")
        return self


class ToolCatalog(ContractModel):
    schema_version: ContractVersion = CONTRACT_VERSION
    tools: list[ToolContract] = Field(default_factory=list)
    fingerprint: NonEmptyStr

    @classmethod
    def create(cls, tools: list[ToolContract]) -> "ToolCatalog":
        payload = [tool.model_dump(mode="json") for tool in tools]
        return cls(tools=tools, fingerprint=_fingerprint(payload))

    @model_validator(mode="after")
    def validate_catalog(self) -> "ToolCatalog":
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool catalog names must be unique")
        expected = _fingerprint([tool.model_dump(mode="json") for tool in self.tools])
        if self.fingerprint != expected:
            raise ValueError("tool catalog fingerprint does not match its contracts")
        return self


__all__ = [
    "ToolCatalog",
    "ToolContract",
    "ToolResultFormat",
    "ToolSideEffect",
]
