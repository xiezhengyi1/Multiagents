from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExistingScenarioSpecExplorer:
    """Create compact specs for existing scenario YAML files."""

    def discover_specs(self, scenario_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
        root = Path(scenario_root)
        specs: list[dict[str, Any]] = []
        for path in sorted(root.glob("**/*.yaml"))[: max(1, int(limit))]:
            payload = self._load_mapping(path)
            if not isinstance(payload, dict):
                continue
            if "base_scenario" in payload and not {"slices", "flows"}.issubset(payload.keys()):
                specs.append(self.summarize_overlay(payload, source=str(path)))
                continue
            specs.append(self.summarize_mapping(payload, source=str(path)))
        return specs

    def summarize_mapping(self, payload: dict[str, Any], *, source: str = "") -> dict[str, Any]:
        slices = [item for item in payload.get("slices") or [] if isinstance(item, dict)]
        gnbs = [item for item in payload.get("gnbs") or [] if isinstance(item, dict)]
        upfs = [item for item in payload.get("upfs") or [] if isinstance(item, dict)]
        ues = [item for item in payload.get("ues") or [] if isinstance(item, dict)]
        apps = [item for item in payload.get("apps") or [] if isinstance(item, dict)]
        flows = [item for item in payload.get("flows") or [] if isinstance(item, dict)]
        service_types = sorted({str(item.get("service_type") or "").strip() for item in flows if item.get("service_type")})
        app_families = sorted({str(item.get("name") or item.get("app_id") or "").strip() for item in apps if item})
        slice_profiles = sorted({str(item.get("label") or "").strip() for item in slices if item.get("label")})
        return {
            "source": source,
            "kind": "base_scenario",
            "name": str(payload.get("name") or "").strip(),
            "scenario_id": str(payload.get("scenario_id") or "").strip(),
            "slice_count": len(slices),
            "gnb_count": len(gnbs),
            "upf_count": len(upfs),
            "ue_count": len(ues),
            "app_count": len(apps),
            "flow_count": len(flows),
            "slice_profiles": slice_profiles,
            "service_types": service_types,
            "app_families": app_families,
            "free5gc_mode": str((payload.get("free5gc") or {}).get("mode") or "").strip(),
            "ns3_scratch": str((payload.get("ns3") or {}).get("scratch_name") or "").strip(),
        }

    @staticmethod
    def summarize_overlay(payload: dict[str, Any], *, source: str = "") -> dict[str, Any]:
        return {
            "source": source,
            "kind": "split_mode_overlay",
            "name": str(payload.get("name") or "").strip(),
            "scenario_id": str(payload.get("scenario_id") or "").strip(),
            "base_scenario": str(payload.get("base_scenario") or "").strip(),
            "ns3_scratch": str((payload.get("ns3") or {}).get("scratch_name") or "").strip(),
            "scheduler_type": str((payload.get("radio") or {}).get("scheduler_type") or "").strip(),
        }

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, Any] | None:
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml

            payload = yaml.safe_load(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            try:
                payload = json.loads(text)
            except Exception:
                return None
            return payload if isinstance(payload, dict) else None
