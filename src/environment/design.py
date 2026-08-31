from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ScenarioCandidate


SERVICE_TYPE_IDS = {"eMBB": 1, "URLLC": 2, "mMTC": 3}
DEFAULT_UE_KEY = "8baf473f2f8fd09487cccbd7097c6862"
DEFAULT_UE_OP = "8e27b6af0e692e750f32667a3b14605d"


class StrictDesignModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityCounts(StrictDesignModel):
    slices: int = Field(ge=1)
    upfs: int = Field(ge=1)
    gnbs: int = Field(ge=1)
    ues: int = Field(ge=1)
    apps: int = Field(ge=1)
    flows: int = Field(ge=1)


class EnvironmentCatalog(StrictDesignModel):
    scenario_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_counts: EntityCounts
    slice_keys: list[str]
    upf_keys: list[str]
    gnb_keys: list[str]
    ue_keys: list[str]
    app_keys: list[str]
    flow_keys: list[str]


class SliceResourceDesign(StrictDesignModel):
    capacity_dl_mbps: float = Field(gt=0)
    capacity_ul_mbps: float = Field(gt=0)
    guaranteed_dl_mbps: float = Field(ge=0)
    guaranteed_ul_mbps: float = Field(ge=0)
    priority: int = Field(ge=1)


class SliceQosDesign(StrictDesignModel):
    latency_ms: float = Field(gt=0)
    jitter_ms: float = Field(ge=0)
    loss_rate: float = Field(ge=0, le=1)
    processing_delay_ms: float = Field(ge=0)


class SliceDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    sst: int = Field(ge=1, le=255)
    sd: str = Field(pattern=r"^[0-9A-Fa-f]{6}$")
    resource: SliceResourceDesign
    qos: SliceQosDesign
    rationale: str = Field(min_length=1)


class UpfDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    dnn: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class GnbDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    alias: str = Field(min_length=1)
    tac: int = Field(ge=1)
    nci: str = Field(min_length=1)
    served_slice_keys: list[str] = Field(min_length=1)
    backhaul_upf_key: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class UeDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    attached_gnb_key: str = Field(min_length=1)
    target_gnb_key: str = Field(min_length=1)
    preferred_gnb_keys: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class AppDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    ue_key: str = Field(min_length=1)
    apn: str = Field(default="internet", min_length=1)
    pdu_session_type: str = Field(default="IPv4", min_length=1)
    rationale: str = Field(min_length=1)


class FlowTrafficDesign(StrictDesignModel):
    packet_size_bytes: float = Field(gt=0)
    arrival_rate_pps: float = Field(gt=0)
    dl_packet_size_bytes: float = Field(gt=0)
    ul_packet_size_bytes: float = Field(gt=0)
    dl_arrival_rate_pps: float = Field(gt=0)
    ul_arrival_rate_pps: float = Field(gt=0)


class FlowSlaDesign(StrictDesignModel):
    latency_ms: float = Field(gt=0)
    jitter_ms: float = Field(ge=0)
    loss_rate: float = Field(ge=0, le=1)
    bandwidth_dl_mbps: float = Field(ge=0)
    bandwidth_ul_mbps: float = Field(ge=0)
    guaranteed_bandwidth_dl_mbps: float = Field(ge=0)
    guaranteed_bandwidth_ul_mbps: float = Field(ge=0)
    priority: int = Field(ge=1)


class FlowDesign(StrictDesignModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    app_key: str = Field(min_length=1)
    slice_key: str = Field(min_length=1)
    five_qi: int = Field(ge=1, le=255)
    service_type: Literal["eMBB", "URLLC", "mMTC"]
    traffic: FlowTrafficDesign
    sla: FlowSlaDesign
    allocated_bandwidth_dl_mbps: float = Field(ge=0)
    allocated_bandwidth_ul_mbps: float = Field(ge=0)
    optimize_requested: bool = False
    rationale: str = Field(min_length=1)


class RuntimeDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    free5gc: dict[str, Any]
    ns3: dict[str, Any]
    writer: dict[str, Any]
    bridge: dict[str, Any]
    split_mode_ns3: dict[str, Any]
    split_mode_runtime: dict[str, Any]
    split_mode_radio: dict[str, Any]


class EnvironmentDesign(StrictDesignModel):
    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    tick_ms: int = Field(gt=0)
    seed: int = Field(ge=0)
    slices: list[SliceDesign] = Field(min_length=1)
    upfs: list[UpfDesign] = Field(min_length=1)
    gnbs: list[GnbDesign] = Field(min_length=1)
    ues: list[UeDesign] = Field(min_length=1)
    apps: list[AppDesign] = Field(min_length=1)
    flows: list[FlowDesign] = Field(min_length=1)
    runtime: RuntimeDesign


class EnvironmentDesignRevision(StrictDesignModel):
    """LLM-authored replacements for a previously accepted semantic design.

    Every collection entry is a complete semantic entity.  The deterministic
    merge only replaces the entity with the same key; it never invents values,
    adds entities, or alters an omitted entity.
    """

    name: str | None = Field(default=None, min_length=1)
    rationale: str | None = Field(default=None, min_length=1)
    tick_ms: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0)
    slices: list[SliceDesign] = Field(default_factory=list)
    upfs: list[UpfDesign] = Field(default_factory=list)
    gnbs: list[GnbDesign] = Field(default_factory=list)
    ues: list[UeDesign] = Field(default_factory=list)
    apps: list[AppDesign] = Field(default_factory=list)
    flows: list[FlowDesign] = Field(default_factory=list)
    runtime: RuntimeDesign | None = None


class EnvironmentDesignError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class CompiledEnvironmentDesign:
    candidate: ScenarioCandidate
    report: dict[str, Any]


class EnvironmentDesignCompiler:
    """Compile an LLM-authored semantic design without changing its metrics."""

    CATALOG_FIELDS = {
        "slices": "slice_keys",
        "upfs": "upf_keys",
        "gnbs": "gnb_keys",
        "ues": "ue_keys",
        "apps": "app_keys",
        "flows": "flow_keys",
    }

    def apply_revision(
        self,
        design: EnvironmentDesign,
        revision: EnvironmentDesignRevision,
    ) -> tuple[EnvironmentDesign, list[str]]:
        """Apply only explicit LLM-authored semantic replacements."""

        payload = design.model_dump(mode="json")
        changed: list[str] = []
        for field_name in ("name", "rationale", "tick_ms", "seed"):
            value = getattr(revision, field_name)
            if value is not None:
                payload[field_name] = value
                changed.append(field_name)

        errors: list[str] = []
        for section in self.CATALOG_FIELDS:
            replacements = list(getattr(revision, section))
            if not replacements:
                continue
            duplicate_keys = sorted(
                key
                for key, count in Counter(item.key for item in replacements).items()
                if count > 1
            )
            if duplicate_keys:
                errors.append(
                    f"revision {section} contains duplicate replacement keys: "
                    + ", ".join(duplicate_keys)
                )
                continue
            indexed = {str(item["key"]): item for item in payload[section]}
            for replacement in replacements:
                if replacement.key not in indexed:
                    errors.append(
                        f"revision cannot add undeclared {section} key {replacement.key}; "
                        "submit a new catalog and complete design to change the manifest"
                    )
                    continue
                indexed[replacement.key] = replacement.model_dump(mode="json")
                changed.append(f"{section}.{replacement.key}")
            payload[section] = [indexed[str(item["key"])] for item in payload[section]]

        if revision.runtime is not None:
            payload["runtime"] = revision.runtime.model_dump(mode="json")
            changed.append("runtime")
        if errors:
            raise EnvironmentDesignError(errors)
        if not changed:
            raise EnvironmentDesignError(
                ["revision contains no explicit semantic replacements"]
            )
        return EnvironmentDesign.model_validate(payload), changed

    def validate_catalog(self, catalog: EnvironmentCatalog) -> list[str]:
        errors: list[str] = []
        expected = catalog.expected_counts.model_dump()
        for section, key_field in self.CATALOG_FIELDS.items():
            keys = list(getattr(catalog, key_field))
            duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
            if duplicates:
                errors.append(f"catalog {section} contains duplicate keys: {', '.join(duplicates)}")
            if len(keys) != expected[section]:
                errors.append(
                    f"catalog expected {expected[section]} {section} but declared {len(keys)} keys"
                )
            if any(not str(key).strip() for key in keys):
                errors.append(f"catalog {section} contains an empty semantic key")
        return errors

    def compile(
        self,
        catalog: EnvironmentCatalog,
        design: EnvironmentDesign,
    ) -> CompiledEnvironmentDesign:
        errors = self.validate_catalog(catalog)
        if design.scenario_id != catalog.scenario_id:
            errors.append(
                f"design scenario_id {design.scenario_id} does not match catalog {catalog.scenario_id}"
            )
        if design.objective != catalog.objective:
            errors.append("design objective does not match the accepted catalog objective")
        errors.extend(self._validate_design_completeness(catalog, design))
        if errors:
            raise EnvironmentDesignError(errors)

        ordered = self._ordered_entities(catalog, design)
        slice_labels = {
            item.key: f"slice-{item.sst}-{item.sd.lower()}"
            for item in ordered["slices"]
        }
        upf_names = {item.key: item.name for item in ordered["upfs"]}
        gnb_names = {item.key: item.name for item in ordered["gnbs"]}
        ue_names = {item.key: item.name for item in ordered["ues"]}
        ue_supis = {
            item.key: f"imsi-208930000000{index:03d}"
            for index, item in enumerate(ordered["ues"], start=1)
        }
        app_ids = {
            item.key: f"app-{index:04d}"
            for index, item in enumerate(ordered["apps"], start=1)
        }
        flow_ids = {
            item.key: f"flow-{index:04d}"
            for index, item in enumerate(ordered["flows"], start=1)
        }

        slices = [self._compile_slice(item, slice_labels[item.key]) for item in ordered["slices"]]
        upfs = [
            {"name": item.name, "role": item.role, "dnn": item.dnn}
            for item in ordered["upfs"]
        ]
        gnbs = [
            {
                "name": item.name,
                "alias": item.alias,
                "tac": item.tac,
                "nci": item.nci,
                "slices": [slice_labels[key] for key in item.served_slice_keys],
                "backhaul_upf": upf_names[item.backhaul_upf_key],
            }
            for item in ordered["gnbs"]
        ]

        apps_by_key = {item.key: item for item in ordered["apps"]}
        flows_by_app: dict[str, list[FlowDesign]] = {key: [] for key in app_ids}
        for flow in ordered["flows"]:
            flows_by_app[flow.app_key].append(flow)

        sessions_by_ue = self._compile_sessions(
            ordered["ues"],
            apps_by_key,
            flows_by_app,
            app_ids,
            slice_labels,
            ue_supis,
        )
        ues = [
            {
                "name": item.name,
                "supi": ue_supis[item.key],
                "gnb": gnb_names[item.attached_gnb_key],
                "key": DEFAULT_UE_KEY,
                "op": DEFAULT_UE_OP,
                "op_type": "OPC",
                "amf": "8000",
                "free5gc_policy": {
                    "target_gnb": gnb_names[item.target_gnb_key],
                    "preferred_gnbs": [gnb_names[key] for key in item.preferred_gnb_keys],
                },
                "sessions": sessions_by_ue[item.key],
            }
            for item in ordered["ues"]
        ]
        apps = [
            {
                "app_id": app_ids[item.key],
                "name": item.name,
                "supi": ue_supis[item.ue_key],
                "ue_name": ue_names[item.ue_key],
                "flow_ids": [flow_ids[flow.key] for flow in flows_by_app[item.key]],
            }
            for item in ordered["apps"]
        ]
        flows = [
            self._compile_flow(
                item,
                app=apps_by_key[item.app_key],
                app_id=app_ids[item.app_key],
                flow_id=flow_ids[item.key],
                ue_name=ue_names[apps_by_key[item.app_key].ue_key],
                supi=ue_supis[apps_by_key[item.app_key].ue_key],
                slice_label=slice_labels[item.slice_key],
                slice_design=next(slice_ for slice_ in ordered["slices"] if slice_.key == item.slice_key),
            )
            for item in ordered["flows"]
        ]

        runtime = design.runtime.model_dump(mode="json")
        filename = design.scenario_id.lower().replace("-", "_") + ".yaml"
        scenario = {
            "name": design.name,
            "scenario_id": design.scenario_id,
            "tick_ms": design.tick_ms,
            "seed": design.seed,
            "slices": slices,
            "upfs": upfs,
            "gnbs": gnbs,
            "ues": ues,
            "apps": apps,
            "flows": flows,
            "free5gc": deepcopy(runtime["free5gc"]),
            "ns3": deepcopy(runtime["ns3"]),
            "writer": deepcopy(runtime["writer"]),
            "topology": {"graph_file": f"graphs/{filename}"},
            "bridge": deepcopy(runtime["bridge"]),
        }
        overlay = {
            "name": f"{design.name}-split",
            "scenario_id": f"{design.scenario_id}-split",
            "base_scenario": f"../{filename}",
            "ns3": deepcopy(runtime["split_mode_ns3"]),
            "runtime": deepcopy(runtime["split_mode_runtime"]),
            "radio": deepcopy(runtime["split_mode_radio"]),
        }
        candidate = ScenarioCandidate(
            scenario_id=design.scenario_id,
            name=design.name,
            scenario=scenario,
            split_mode_overlay=overlay,
        )
        report = {
            "version": 1,
            "scenario_id": design.scenario_id,
            "objective": design.objective,
            "scenario_rationale": design.rationale,
            "catalog": catalog.model_dump(mode="json"),
            "design": design.model_dump(mode="json"),
            "derived_identifiers": {
                "slice_labels": slice_labels,
                "upf_names": upf_names,
                "gnb_names": gnb_names,
                "ue_names": ue_names,
                "ue_supis": ue_supis,
                "app_ids": app_ids,
                "flow_ids": flow_ids,
            },
        }
        return CompiledEnvironmentDesign(candidate=candidate, report=report)

    def _validate_design_completeness(
        self,
        catalog: EnvironmentCatalog,
        design: EnvironmentDesign,
    ) -> list[str]:
        errors: list[str] = []
        for section, key_field in self.CATALOG_FIELDS.items():
            expected_keys = set(getattr(catalog, key_field))
            actual_items = list(getattr(design, section))
            actual_keys = [item.key for item in actual_items]
            duplicate_keys = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
            if duplicate_keys:
                errors.append(f"design {section} contains duplicate keys: {', '.join(duplicate_keys)}")
            missing = sorted(expected_keys - set(actual_keys))
            extra = sorted(set(actual_keys) - expected_keys)
            if missing:
                errors.append(f"design {section} is missing catalog keys: {', '.join(missing)}")
            if extra:
                errors.append(f"design {section} contains undeclared keys: {', '.join(extra)}")

        slice_keys = set(catalog.slice_keys)
        upf_keys = set(catalog.upf_keys)
        gnb_keys = set(catalog.gnb_keys)
        ue_keys = set(catalog.ue_keys)
        app_keys = set(catalog.app_keys)
        for item in design.gnbs:
            self._check_references(errors, f"gNB {item.key} slice", item.served_slice_keys, slice_keys)
            self._check_references(errors, f"gNB {item.key} backhaul UPF", [item.backhaul_upf_key], upf_keys)
        gnb_slice_map = {item.key: set(item.served_slice_keys) for item in design.gnbs}
        for item in design.ues:
            self._check_references(errors, f"UE {item.key} attached gNB", [item.attached_gnb_key], gnb_keys)
            self._check_references(errors, f"UE {item.key} target gNB", [item.target_gnb_key], gnb_keys)
            self._check_references(errors, f"UE {item.key} preferred gNB", item.preferred_gnb_keys, gnb_keys)
        for item in design.apps:
            self._check_references(errors, f"app {item.key} UE", [item.ue_key], ue_keys)
        app_by_key = {item.key: item for item in design.apps}
        ue_by_key = {item.key: item for item in design.ues}
        for item in design.flows:
            self._check_references(errors, f"flow {item.key} app", [item.app_key], app_keys)
            self._check_references(errors, f"flow {item.key} slice", [item.slice_key], slice_keys)
            app = app_by_key.get(item.app_key)
            ue = ue_by_key.get(app.ue_key) if app is not None else None
            if ue is not None and item.slice_key not in gnb_slice_map.get(ue.attached_gnb_key, set()):
                errors.append(
                    f"flow {item.key} requests slice {item.slice_key}, which is not served by "
                    f"UE {ue.key} attached gNB {ue.attached_gnb_key}"
                )
        flow_counts = Counter(item.app_key for item in design.flows)
        for app in design.apps:
            if flow_counts[app.key] == 0:
                errors.append(f"app {app.key} has no declared flows")
        app_counts = Counter(item.ue_key for item in design.apps)
        for ue in design.ues:
            if app_counts[ue.key] == 0:
                errors.append(f"UE {ue.key} has no declared applications")
        return errors

    @staticmethod
    def _check_references(
        errors: list[str],
        label: str,
        referenced_keys: list[str],
        available_keys: set[str],
    ) -> None:
        missing = sorted({key for key in referenced_keys if key not in available_keys})
        if missing:
            errors.append(f"{label} references undeclared keys: {', '.join(missing)}")

    def _ordered_entities(
        self,
        catalog: EnvironmentCatalog,
        design: EnvironmentDesign,
    ) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        for section, key_field in self.CATALOG_FIELDS.items():
            by_key = {item.key: item for item in getattr(design, section)}
            result[section] = [by_key[key] for key in getattr(catalog, key_field)]
        return result

    @staticmethod
    def _compile_slice(item: SliceDesign, label: str) -> dict[str, Any]:
        return {
            "sst": item.sst,
            "sd": item.sd.lower(),
            "label": label,
            "resource": item.resource.model_dump(mode="json"),
            "qos": item.qos.model_dump(mode="json"),
        }

    @staticmethod
    def _compile_sessions(
        ues: list[UeDesign],
        apps_by_key: dict[str, AppDesign],
        flows_by_app: dict[str, list[FlowDesign]],
        app_ids: dict[str, str],
        slice_labels: dict[str, str],
        ue_supis: dict[str, str],
    ) -> dict[str, list[dict[str, Any]]]:
        sessions: dict[str, list[dict[str, Any]]] = {item.key: [] for item in ues}
        for app_key, app in apps_by_key.items():
            grouped: dict[str, list[FlowDesign]] = {}
            for flow in flows_by_app[app_key]:
                grouped.setdefault(flow.slice_key, []).append(flow)
            for slice_key, flows in grouped.items():
                session_ref = (
                    f"{ue_supis[app.ue_key]}:{app_ids[app_key]}:"
                    f"{slice_labels[slice_key]}:{app.apn}"
                )
                highest_priority_flow = min(flows, key=lambda flow: flow.sla.priority)
                sessions[app.ue_key].append(
                    {
                        "slice_ref": slice_labels[slice_key],
                        "session_ref": session_ref,
                        "apn": app.apn,
                        "type": app.pdu_session_type,
                        "five_qi": highest_priority_flow.five_qi,
                        "app_id": app_ids[app_key],
                    }
                )
        return sessions

    @staticmethod
    def _compile_flow(
        item: FlowDesign,
        *,
        app: AppDesign,
        app_id: str,
        flow_id: str,
        ue_name: str,
        supi: str,
        slice_label: str,
        slice_design: SliceDesign,
    ) -> dict[str, Any]:
        traffic = item.traffic.model_dump(mode="json")
        session_ref = f"{supi}:{app_id}:{slice_label}:{app.apn}"
        return {
            "flow_id": flow_id,
            "name": item.name,
            "supi": supi,
            "app_id": app_id,
            "slice_ref": slice_label,
            "session_ref": session_ref,
            "ue_name": ue_name,
            "app_name": app.name,
            "five_qi": item.five_qi,
            "service_type": item.service_type,
            "service_type_id": SERVICE_TYPE_IDS[item.service_type],
            **traffic,
            "current_slice_snssai": f"{slice_design.sst:02d}{slice_design.sd.lower()}",
            "allocated_bandwidth_dl_mbps": item.allocated_bandwidth_dl_mbps,
            "allocated_bandwidth_ul_mbps": item.allocated_bandwidth_ul_mbps,
            "optimize_requested": item.optimize_requested,
            "sla_target": item.sla.model_dump(mode="json"),
        }


__all__ = [
    "AppDesign",
    "CompiledEnvironmentDesign",
    "EntityCounts",
    "EnvironmentCatalog",
    "EnvironmentDesign",
    "EnvironmentDesignCompiler",
    "EnvironmentDesignError",
    "FlowDesign",
    "FlowSlaDesign",
    "FlowTrafficDesign",
    "GnbDesign",
    "RuntimeDesign",
    "SliceDesign",
    "SliceQosDesign",
    "SliceResourceDesign",
    "UeDesign",
    "UpfDesign",
]
