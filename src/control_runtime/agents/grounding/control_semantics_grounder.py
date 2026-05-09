from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ...domain.control_plane import ControlSemantics, ControlStage, SemanticTarget, SemanticTargetType
from ...domain.policy_plan import FlowSelector


class ControlSemanticsGrounder:
    def ground(
        self,
        *,
        directives: Dict[str, Any],
        flows: List[FlowSelector],
        flow_catalog: List[Dict[str, Any]],
        app_catalog: List[Dict[str, Any]],
    ) -> ControlSemantics:
        raw_semantics = directives.get("control_semantics")
        if not isinstance(raw_semantics, dict):
            return ControlSemantics()
        semantics = ControlSemantics.model_validate(raw_semantics)
        if not semantics.stages:
            return semantics

        flow_rows = [self._flow_selector_to_row(flow) for flow in flows]
        all_flow_rows = flow_rows + [item for item in (flow_catalog or []) if isinstance(item, dict)]
        app_rows = [item for item in (app_catalog or []) if isinstance(item, dict)]

        grounded_stages: List[ControlStage] = []
        prior_active_flow_ids: List[str] = []
        for stage in semantics.stages:
            grounded_targets: List[SemanticTarget] = []
            active_flow_ids: List[str] = []
            active_app_ids: List[str] = []
            for target in stage.targets:
                grounded_target = target.model_copy(deep=True)
                matched_flows = self._match_semantic_target_flows(target=target, flow_rows=all_flow_rows)
                matched_apps = self._match_semantic_target_apps(target=target, app_rows=app_rows)
                grounded_target.matched_flow_ids = [
                    str(item.get("flow_id") or "").strip()
                    for item in matched_flows
                    if str(item.get("flow_id") or "").strip()
                ]
                grounded_target.matched_app_ids = [
                    str(item.get("app_id") or "").strip()
                    for item in matched_apps
                    if str(item.get("app_id") or "").strip()
                ]
                if grounded_target.target_type == SemanticTargetType.SCOPE and grounded_target.semantic_name == "其余业务":
                    grounded_target.matched_flow_ids = [
                        str(item.get("flow_id") or "").strip()
                        for item in all_flow_rows
                        if str(item.get("flow_id") or "").strip()
                        and str(item.get("flow_id") or "").strip() not in prior_active_flow_ids
                    ]
                if len(grounded_target.matched_flow_ids) == 1:
                    flow_row = matched_flows[0]
                    grounded_target.flow_id = grounded_target.matched_flow_ids[0]
                    grounded_target.flow_name = str(flow_row.get("flow_name") or grounded_target.semantic_name).strip()
                    grounded_target.app_id = str(flow_row.get("app_id") or "").strip()
                    grounded_target.app_name = str(flow_row.get("app_name") or "").strip()
                    grounded_target.supi = str(flow_row.get("supi") or "").strip()
                    grounded_target.resolution_status = "grounded_flow"
                elif len(grounded_target.matched_app_ids) == 1:
                    app_row = matched_apps[0]
                    grounded_target.app_id = grounded_target.matched_app_ids[0]
                    grounded_target.app_name = str(app_row.get("app_name") or grounded_target.semantic_name).strip()
                    grounded_target.resolution_status = "grounded_app"
                    grounded_target.matched_flow_ids = [
                        str(item.get("flow_id") or "").strip()
                        for item in all_flow_rows
                        if str(item.get("app_id") or "").strip() == grounded_target.app_id
                        and str(item.get("flow_id") or "").strip()
                    ]
                else:
                    grounded_target.resolution_status = "semantic"
                grounded_targets.append(grounded_target)
                for flow_id in grounded_target.matched_flow_ids:
                    if flow_id and flow_id not in active_flow_ids:
                        active_flow_ids.append(flow_id)
                for app_id in grounded_target.matched_app_ids:
                    if app_id and app_id not in active_app_ids:
                        active_app_ids.append(app_id)
            grounded_stages.append(
                stage.model_copy(
                    update={
                        "targets": grounded_targets,
                        "active_flow_ids": active_flow_ids,
                        "active_app_ids": active_app_ids,
                    },
                    deep=True,
                )
            )
            for flow_id in active_flow_ids:
                if flow_id and flow_id not in prior_active_flow_ids:
                    prior_active_flow_ids.append(flow_id)
        return semantics.model_copy(update={"stages": grounded_stages}, deep=True)

    @staticmethod
    def _flow_selector_to_row(flow: FlowSelector) -> Dict[str, Any]:
        return {
            "flow_id": str(flow.flow_id or "").strip(),
            "flow_name": str(flow.name or "").strip(),
            "app_id": str(flow.app_id or "").strip(),
            "app_name": "",
            "supi": str(flow.supi or "").strip(),
        }

    def _match_semantic_target_flows(
        self,
        *,
        target: SemanticTarget,
        flow_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semantic_name = str(target.semantic_name or "").strip()
        if not semantic_name:
            return []
        normalized_name = self._normalize_semantic_name(semantic_name)
        matches: List[Dict[str, Any]] = []
        for row in flow_rows:
            flow_name = str(row.get("flow_name") or row.get("name") or "").strip()
            if not flow_name:
                continue
            normalized_flow = self._normalize_semantic_name(flow_name)
            if normalized_flow == normalized_name:
                matches.append(row)
        return self._deduplicate_flow_rows(matches)

    def _match_semantic_target_apps(
        self,
        *,
        target: SemanticTarget,
        app_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semantic_name = str(target.semantic_name or "").strip()
        if not semantic_name:
            return []
        normalized_name = self._normalize_semantic_name(semantic_name)
        matches: List[Dict[str, Any]] = []
        for row in app_rows:
            app_name = str(row.get("app_name") or row.get("name") or "").strip()
            if not app_name:
                continue
            normalized_app = self._normalize_semantic_name(app_name)
            if normalized_app == normalized_name:
                matches.append(row)
        return self._deduplicate_app_rows(matches)

    @staticmethod
    def _normalize_semantic_name(value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _deduplicate_flow_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for row in rows:
            key = (str(row.get("app_id") or "").strip(), str(row.get("flow_id") or "").strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    @staticmethod
    def _deduplicate_app_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            app_id = str(row.get("app_id") or "").strip()
            if not app_id or app_id in seen:
                continue
            seen.add(app_id)
            deduped.append(row)
        return deduped
