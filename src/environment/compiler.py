from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .contracts import EnvironmentGenerationRequest, EnvironmentValidationReport, ScenarioCandidate
from .prompts import GENERATION_PROMPT_TEMPLATE


class EnvironmentAgentCompiler:
    """Pure scenario prompt and validation logic for the environment agent."""

    REQUIRED_ROOT_FIELDS = (
        "name",
        "scenario_id",
        "tick_ms",
        "seed",
        "slices",
        "upfs",
        "gnbs",
        "ues",
        "apps",
        "flows",
        "free5gc",
        "ns3",
        "writer",
        "topology",
        "bridge",
    )
    REQUIRED_OVERLAY_FIELDS = ("name", "scenario_id", "base_scenario", "ns3", "runtime", "radio")

    def build_generation_prompt(self, request: EnvironmentGenerationRequest) -> str:
        return GENERATION_PROMPT_TEMPLATE.format(
            scenario_id=request.scenario_id,
            objective=request.objective,
            complexity=request.complexity,
            target_flow_count=request.target_flow_count,
            topology_mode=request.topology_mode,
            stress_mode=request.stress_mode,
            output_dir=request.output_dir,
        )

    def validate_candidate(self, candidate: ScenarioCandidate) -> EnvironmentValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        scenario = candidate.scenario
        if not isinstance(scenario, dict):
            return EnvironmentValidationReport(
                scenario_id=candidate.scenario_id,
                ok=False,
                errors=("scenario payload must be a mapping",),
            )

        errors.extend(self._validate_root_fields(scenario))
        if errors:
            return EnvironmentValidationReport(candidate.scenario_id, ok=False, errors=tuple(errors))
        self._validate_loader_contract(scenario, errors)

        slice_labels = self._collect_unique_strings(scenario.get("slices"), "label", errors, "slice")
        upf_names = self._collect_unique_strings(scenario.get("upfs"), "name", errors, "upf")
        gnb_names = self._collect_unique_strings(scenario.get("gnbs"), "name", errors, "gNB")
        ue_names = self._collect_unique_strings(scenario.get("ues"), "name", errors, "UE")
        ue_supis = self._collect_unique_strings(scenario.get("ues"), "supi", errors, "UE")
        app_ids = self._collect_unique_strings(scenario.get("apps"), "app_id", errors, "app")
        flow_ids = self._collect_unique_strings(scenario.get("flows"), "flow_id", errors, "flow")

        gnb_slice_map: dict[str, set[str]] = {}
        for gnb in scenario.get("gnbs") or []:
            if isinstance(gnb, dict):
                name = str(gnb.get("name") or "").strip()
                if name:
                    gnb_slice_map[name] = {str(s) for s in (gnb.get("slices") or []) if str(s).strip()}

        self._validate_gnbs(scenario.get("gnbs") or [], slice_labels, upf_names, errors)
        self._validate_ues(scenario.get("ues") or [], slice_labels, gnb_names, gnb_slice_map, app_ids, errors)
        self._validate_apps(scenario.get("apps") or [], ue_names, ue_supis, flow_ids, errors)
        self._validate_flows(scenario.get("flows") or [], slice_labels, ue_names, ue_supis, app_ids, errors)
        self._validate_capacity(scenario, errors, warnings)
        self._validate_overlay(candidate.split_mode_overlay, errors)

        return EnvironmentValidationReport(
            scenario_id=candidate.scenario_id,
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _validate_root_fields(self, scenario: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for field in self.REQUIRED_ROOT_FIELDS:
            if field not in scenario:
                errors.append(f"missing root field {field}")
        for field in ("slices", "upfs", "gnbs", "ues", "apps", "flows"):
            if field in scenario and not isinstance(scenario.get(field), list):
                errors.append(f"root field {field} must be a list")
        return errors

    @staticmethod
    def _validate_loader_contract(scenario: dict[str, Any], errors: list[str]) -> None:
        for item in scenario.get("slices") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "<unknown>").strip()
            try:
                if isinstance(item.get("sst"), bool):
                    raise ValueError
                sst = int(item["sst"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"slice {label} field sst must be an integer")
                sst = None
            sd = str(item.get("sd") or "").strip()
            if not sd:
                errors.append(f"slice {label} missing field sd")
            if sst is not None and sd:
                derived_identifier = f"slice-{sst}-{sd.lower()}"
                if label != derived_identifier:
                    errors.append(
                        f"slice label {label} must equal derived identifier {derived_identifier}"
                    )
            resource = item.get("resource")
            if not isinstance(resource, dict):
                errors.append(f"slice {label} resource must be a mapping")
            else:
                for field in (
                    "capacity_dl_mbps",
                    "capacity_ul_mbps",
                    "guaranteed_dl_mbps",
                    "guaranteed_ul_mbps",
                ):
                    if field not in resource:
                        errors.append(f"slice {label} resource missing field {field}")

        for item in scenario.get("ues") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "<unknown>").strip()
            for field in ("key", "op"):
                if not str(item.get(field) or "").strip():
                    errors.append(f"UE {name} missing field {field}")
            if not isinstance(item.get("free5gc_policy"), dict):
                errors.append(f"UE {name} free5gc_policy must be a mapping")
            sessions = item.get("sessions")
            if not isinstance(sessions, list) or not sessions:
                errors.append(f"UE {name} sessions must be a non-empty list")
                continue
            for session in sessions:
                if not isinstance(session, dict):
                    errors.append(f"UE {name} session must be a mapping")
                    continue
                for field in ("slice_ref", "app_id"):
                    if not str(session.get(field) or "").strip():
                        errors.append(f"UE {name} session missing field {field}")

        for section, fields in (
            ("free5gc", ("compose_file", "config_root")),
            ("ns3", ("ns3_root",)),
        ):
            payload = scenario.get(section)
            if not isinstance(payload, dict):
                errors.append(f"{section} must be a mapping")
                continue
            for field in fields:
                if not str(payload.get(field) or "").strip():
                    errors.append(f"{section} missing field {field}")

    @staticmethod
    def _collect_unique_strings(
        items: Any,
        key: str,
        errors: list[str],
        entity_name: str,
    ) -> set[str]:
        values: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                errors.append(f"{entity_name} entry must be a mapping")
                continue
            value = str(item.get(key) or "").strip()
            if not value:
                errors.append(f"{entity_name} entry is missing {key}")
                continue
            values.append(value)
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        for value in duplicates:
            errors.append(f"duplicate {entity_name} {key} {value}")
        return set(values)

    @staticmethod
    def _validate_gnbs(
        gnbs: Iterable[dict[str, Any]],
        slice_labels: set[str],
        upf_names: set[str],
        errors: list[str],
    ) -> None:
        for gnb in gnbs:
            name = str(gnb.get("name") or "<unknown>").strip()
            for slice_ref in gnb.get("slices") or []:
                if str(slice_ref) not in slice_labels:
                    errors.append(f"gNB {name} references missing slice {slice_ref}")
            backhaul_upf = str(gnb.get("backhaul_upf") or "").strip()
            if backhaul_upf and backhaul_upf not in upf_names:
                errors.append(f"gNB {name} references missing backhaul_upf {backhaul_upf}")

    @staticmethod
    def _validate_ues(
        ues: Iterable[dict[str, Any]],
        slice_labels: set[str],
        gnb_names: set[str],
        gnb_slice_map: dict[str, set[str]],
        app_ids: set[str],
        errors: list[str],
    ) -> None:
        for ue in ues:
            name = str(ue.get("name") or "<unknown>").strip()
            gnb = str(ue.get("gnb") or "").strip()
            if gnb and gnb not in gnb_names:
                errors.append(f"UE {name} references missing gnb {gnb}")
            policy = ue.get("free5gc_policy") if isinstance(ue.get("free5gc_policy"), dict) else {}
            target_gnb = str(policy.get("target_gnb") or "").strip()
            if target_gnb and target_gnb not in gnb_names:
                errors.append(f"UE {name} references missing target_gnb {target_gnb}")
            for preferred in policy.get("preferred_gnbs") or []:
                if str(preferred) not in gnb_names:
                    errors.append(f"UE {name} references missing preferred_gnb {preferred}")
            for session in ue.get("sessions") or []:
                slice_ref = str(session.get("slice_ref") or "").strip()
                app_id = str(session.get("app_id") or "").strip()
                if slice_ref and slice_ref not in slice_labels:
                    errors.append(f"UE {name} session references missing slice_ref {slice_ref}")
                if app_id and app_id not in app_ids:
                    errors.append(f"UE {name} session references missing app_id {app_id}")
            # Cross-reference: UE's attached gNB must serve the slices in the UE's sessions
            if gnb in gnb_slice_map:
                for session in ue.get("sessions") or []:
                    session_slice = str(session.get("slice_ref") or "").strip()
                    if session_slice and session_slice not in gnb_slice_map[gnb]:
                        errors.append(
                            f"UE {name} attached to gNB {gnb} uses session slice {session_slice} "
                            "not advertised by that gNB"
                        )

    @staticmethod
    def _validate_apps(
        apps: Iterable[dict[str, Any]],
        ue_names: set[str],
        ue_supis: set[str],
        flow_ids: set[str],
        errors: list[str],
    ) -> None:
        for app in apps:
            app_id = str(app.get("app_id") or "<unknown>").strip()
            ue_name = str(app.get("ue_name") or "").strip()
            supi = str(app.get("supi") or "").strip()
            if ue_name and ue_name not in ue_names:
                errors.append(f"app {app_id} references missing ue_name {ue_name}")
            if supi and supi not in ue_supis:
                errors.append(f"app {app_id} references missing supi {supi}")
            for flow_id in app.get("flow_ids") or []:
                if str(flow_id) not in flow_ids:
                    errors.append(f"app {app_id} references missing flow_id {flow_id}")

    @staticmethod
    def _validate_flows(
        flows: Iterable[dict[str, Any]],
        slice_labels: set[str],
        ue_names: set[str],
        ue_supis: set[str],
        app_ids: set[str],
        errors: list[str],
    ) -> None:
        for flow in flows:
            flow_id = str(flow.get("flow_id") or "<unknown>").strip()
            app_id = str(flow.get("app_id") or "").strip()
            slice_ref = str(flow.get("slice_ref") or "").strip()
            ue_name = str(flow.get("ue_name") or "").strip()
            supi = str(flow.get("supi") or "").strip()
            if app_id and app_id not in app_ids:
                errors.append(f"flow {flow_id} references missing app_id {app_id}")
            if slice_ref and slice_ref not in slice_labels:
                errors.append(f"flow {flow_id} references missing slice_ref {slice_ref}")
            if ue_name and ue_name not in ue_names:
                errors.append(f"flow {flow_id} references missing ue_name {ue_name}")
            if supi and supi not in ue_supis:
                errors.append(f"flow {flow_id} references missing supi {supi}")

    @staticmethod
    def _validate_capacity(scenario: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
        capacity_by_slice: dict[str, tuple[float, float]] = {}
        for item in scenario.get("slices") or []:
            label = str(item.get("label") or "").strip()
            resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
            capacity_by_slice[label] = (
                float(resource.get("capacity_dl_mbps", 0.0) or 0.0),
                float(resource.get("capacity_ul_mbps", 0.0) or 0.0),
            )
        requested_by_slice: dict[str, list[float]] = {}
        for flow in scenario.get("flows") or []:
            slice_ref = str(flow.get("slice_ref") or "").strip()
            sla = flow.get("sla_target") if isinstance(flow.get("sla_target"), dict) else {}
            requested_by_slice.setdefault(slice_ref, [0.0, 0.0])
            requested_by_slice[slice_ref][0] += float(sla.get("guaranteed_bandwidth_dl_mbps", 0.0) or 0.0)
            requested_by_slice[slice_ref][1] += float(sla.get("guaranteed_bandwidth_ul_mbps", 0.0) or 0.0)
        for slice_ref, requested in requested_by_slice.items():
            capacity = capacity_by_slice.get(slice_ref)
            if not capacity:
                continue
            if requested[0] > capacity[0] or requested[1] > capacity[1]:
                errors.append(
                    f"slice {slice_ref} guaranteed bandwidth exceeds capacity "
                    f"(requested_dl={requested[0]}, capacity_dl={capacity[0]}, "
                    f"requested_ul={requested[1]}, capacity_ul={capacity[1]})"
                )
            elif requested[0] > capacity[0] * 0.9 or requested[1] > capacity[1] * 0.9:
                warnings.append(f"slice {slice_ref} guaranteed bandwidth is above 90 percent of capacity")

    def _validate_overlay(self, overlay: dict[str, Any] | None, errors: list[str]) -> None:
        if overlay is None:
            return
        if not isinstance(overlay, dict):
            errors.append("split_mode_overlay must be a mapping")
            return
        for field in self.REQUIRED_OVERLAY_FIELDS:
            if field not in overlay:
                errors.append(f"split_mode_overlay missing field {field}")
        for section in ("ns3", "runtime", "radio"):
            if section in overlay and not isinstance(overlay.get(section), dict):
                errors.append(f"split_mode_overlay {section} must be a mapping")
