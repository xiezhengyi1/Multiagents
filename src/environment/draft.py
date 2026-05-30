from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ScenarioCandidate


SECTION_ORDER = (
    "metadata",
    "slices",
    "upfs",
    "gnbs",
    "ues",
    "apps",
    "flows",
    "runtime_config",
    "split_mode_overlay",
)

COLLECTION_ID_KEYS = {
    "slices": "label",
    "upfs": "name",
    "gnbs": "name",
    "ues": "name",
    "apps": "app_id",
    "flows": "flow_id",
}


@dataclass
class ScenarioDraftStore:
    sections: dict[str, Any] = field(default_factory=dict)
    completed_sections: list[str] = field(default_factory=list)
    validation_passed: bool = False
    scenario_path: str = ""
    split_mode_overlay_path: str = ""

    def initialize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        required = ("name", "scenario_id", "tick_ms", "seed")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError("metadata missing fields: " + ", ".join(missing))
        self.sections = {"metadata": deepcopy(metadata)}
        self.completed_sections = ["metadata"]
        self._invalidate_evidence()
        return self.summary()

    def replace_section(self, section: str, payload: Any) -> dict[str, Any]:
        self._require_initialized()
        self._require_known_section(section)
        if section == "metadata":
            raise ValueError("metadata must be set with initialize")
        next_section = self.next_section()
        if section not in self.completed_sections and section != next_section:
            raise ValueError(f"next required section is {next_section}")
        self.sections[section] = deepcopy(payload)
        if section not in self.completed_sections:
            self.completed_sections.append(section)
        self._invalidate_evidence()
        return self.summary()

    def next_section(self) -> str | None:
        for section in SECTION_ORDER:
            if section not in self.completed_sections:
                return section
        return None

    def patch_entity(self, section: str, entity_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        id_key = COLLECTION_ID_KEYS.get(section)
        if not id_key:
            raise ValueError(f"section {section} does not support entity patches")
        entities = self.sections.get(section)
        if not isinstance(entities, list):
            raise ValueError(f"section {section} has not been generated")
        if id_key in changes and str(changes[id_key]) != str(entity_id):
            raise ValueError(f"patch cannot change stable id {id_key}")
        for entity in entities:
            if isinstance(entity, dict) and str(entity.get(id_key) or "") == str(entity_id):
                entity.update(deepcopy(changes))
                self._invalidate_evidence()
                return self.summary()
        entity_name = id_key.removesuffix("_id")
        raise ValueError(f"{entity_name} {entity_id} does not exist")

    def inspect_section(self, section: str) -> dict[str, Any]:
        self._require_initialized()
        self._require_known_section(section)
        return {
            "status": "ok",
            "section": section,
            "payload": deepcopy(self.sections.get(section)),
            "draft": self.summary(),
        }

    def assemble_candidate(self) -> ScenarioCandidate:
        missing = [section for section in SECTION_ORDER if section not in self.completed_sections]
        if missing:
            raise ValueError("draft missing sections: " + ", ".join(missing))
        metadata = deepcopy(self.sections["metadata"])
        runtime = deepcopy(self.sections["runtime_config"])
        scenario = {
            **metadata,
            "slices": deepcopy(self.sections["slices"]),
            "upfs": deepcopy(self.sections["upfs"]),
            "gnbs": deepcopy(self.sections["gnbs"]),
            "ues": deepcopy(self.sections["ues"]),
            "apps": deepcopy(self.sections["apps"]),
            "flows": deepcopy(self.sections["flows"]),
            **runtime,
        }
        return ScenarioCandidate(
            scenario_id=str(metadata["scenario_id"]),
            name=str(metadata["name"]),
            scenario=scenario,
            split_mode_overlay=deepcopy(self.sections["split_mode_overlay"]),
        )

    def record_written_paths(self, *, scenario_path: Path, split_mode_overlay_path: Path | None) -> None:
        self.scenario_path = str(Path(scenario_path).resolve())
        self.split_mode_overlay_path = (
            str(Path(split_mode_overlay_path).resolve()) if split_mode_overlay_path else ""
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": str((self.sections.get("metadata") or {}).get("scenario_id") or ""),
            "completed_sections": list(self.completed_sections),
            "next_section": self.next_section(),
            "section_counts": {
                section: len(payload)
                for section, payload in self.sections.items()
                if isinstance(payload, list)
            },
            "reference_ids": self._reference_ids(),
            "validation_passed": self.validation_passed,
            "scenario_path": self.scenario_path,
        }

    def _reference_ids(self) -> dict[str, list[str]]:
        labels = {
            "slices": "slice_labels",
            "upfs": "upf_names",
            "gnbs": "gnb_names",
            "ues": "ue_names",
            "apps": "app_ids",
            "flows": "flow_ids",
        }
        return {
            labels[section]: [
                str(item.get(id_key) or "")
                for item in self.sections.get(section, [])
                if isinstance(item, dict) and str(item.get(id_key) or "")
            ]
            for section, id_key in COLLECTION_ID_KEYS.items()
            if section in self.sections
        }

    def _invalidate_evidence(self) -> None:
        self.validation_passed = False
        self.scenario_path = ""
        self.split_mode_overlay_path = ""

    def _require_initialized(self) -> None:
        if "metadata" not in self.sections:
            raise ValueError("draft is not initialized")

    @staticmethod
    def _require_known_section(section: str) -> None:
        if section not in SECTION_ORDER:
            raise ValueError(f"unsupported draft section: {section}")
