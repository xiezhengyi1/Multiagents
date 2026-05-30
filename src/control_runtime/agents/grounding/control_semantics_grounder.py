from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ...domain.control_plane import ControlSemantics, ControlStage, SemanticTarget, SemanticTargetType
from ...domain.policy_plan import FlowSelector

_logger = logging.getLogger(__name__)

_DEFAULT_SEMANTIC_MATCH_MODEL = "qwen3-30b-a3b-instruct-2507"
_SEMANTIC_MATCH_MAX_CANDIDATES = 30  # cap rows sent to LLM to keep prompt size bounded


class ControlSemanticsGrounder:
    def __init__(self, *, llm_client: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        llm_client:
            Optional pre-built ``openai.OpenAI`` (or compatible) client used for
            semantic (LLM-based) flow matching.  When *None* the grounder will try
            to lazily construct a client from ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
            env vars.  Pass ``False`` to disable LLM matching entirely.
        """
        # ``False`` explicitly disables; ``None`` means "try lazily"
        self._llm_client: Any = llm_client
        self._llm_client_initialized: bool = llm_client is not None
        self._llm_model: str = (_DEFAULT_SEMANTIC_MATCH_MODEL)

    def _get_llm_client(self) -> Optional[Any]:
        """Return a cached OpenAI client, lazily initialising from env vars."""
        if self._llm_client_initialized:
            return self._llm_client  # may be False (disabled) or an actual client
        self._llm_client_initialized = True
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            _logger.debug("ControlSemanticsGrounder: no OPENAI_API_KEY — LLM semantic matching disabled")
            self._llm_client = None
            return None
        try:
            from openai import OpenAI  # local import to avoid hard dependency at module load
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url or None)
        except Exception as exc:  # pragma: no cover
            _logger.warning("ControlSemanticsGrounder: failed to init OpenAI client: %s", exc)
            self._llm_client = None
        return self._llm_client

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
        # Fast path: if flow_id is already grounded on the target, use exact lookup.
        explicit_flow_id = str(target.flow_id or "").strip()
        if explicit_flow_id:
            exact = [
                row for row in flow_rows
                if str(row.get("flow_id") or "").strip() == explicit_flow_id
            ]
            if exact:
                return self._deduplicate_flow_rows(exact)

        # IEA sometimes leaves semantic_name empty and puts the name in flow_name instead.
        semantic_name = str(target.semantic_name or "").strip() or str(target.flow_name or "").strip()
        if not semantic_name:
            return []
        normalized_name = self._normalize_semantic_name(semantic_name)

        # Restrict candidate rows to the target UE when SUPI is known.
        # This prevents cross-UE false positives from substring/fuzzy matching.
        target_supi = str(target.supi or "").strip()
        if target_supi:
            supi_rows = [
                row for row in flow_rows
                if str(row.get("supi") or "").strip() == target_supi
            ]
            candidate_rows = supi_rows if supi_rows else flow_rows
        else:
            candidate_rows = flow_rows

        # Pass 1: exact normalized name match.
        matches: List[Dict[str, Any]] = []
        for row in candidate_rows:
            flow_name = str(row.get("flow_name") or row.get("name") or "").strip()
            if not flow_name:
                continue
            normalized_flow = self._normalize_semantic_name(flow_name)
            if normalized_flow == normalized_name:
                matches.append(row)

        # Pass 2: substring fallback.
        # LLM sometimes appends action/modifier suffixes to semantic_name
        # (e.g. "Remote_Drive_video_1_migration" for flow "Remote_Drive_video_1").
        # Accept a match when the normalized flow name is a substring of the normalized
        # semantic name (or vice-versa), with a minimum length guard to avoid broad matches.
        if not matches:
            for row in candidate_rows:
                flow_name = str(row.get("flow_name") or row.get("name") or "").strip()
                if not flow_name:
                    continue
                normalized_flow = self._normalize_semantic_name(flow_name)
                if len(normalized_flow) >= 5 and (
                    normalized_flow in normalized_name or normalized_name in normalized_flow
                ):
                    matches.append(row)

        # Pass 3: LLM semantic matching as last resort.
        if not matches:
            matches = self._semantic_match_flows_with_llm(
                target=target,
                candidate_rows=candidate_rows,
            )

        return self._deduplicate_flow_rows(matches)

    def _semantic_match_flows_with_llm(
        self,
        *,
        target: SemanticTarget,
        candidate_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ask an LLM to select the best matching flows from *candidate_rows*.

        Returns an empty list when:
        - no LLM client is available, or
        - the candidate list is empty, or
        - the LLM call fails (non-fatal).
        """
        if not candidate_rows:
            return []
        client = self._get_llm_client()
        if not client:
            return []

        capped = candidate_rows[:_SEMANTIC_MATCH_MAX_CANDIDATES]
        candidates_text = "\n".join(
            f"  - flow_id={r.get('flow_id')!r}, flow_name={r.get('flow_name') or r.get('name')!r}, supi={r.get('supi')!r}"
            for r in capped
        )
        prompt = (
            "You are a 5G network flow identifier assistant.\n"
            "Given the semantic target description below, select the flow IDs from the "
            "candidate list that best match the intent.\n\n"
            "Semantic target:\n"
            f"  semantic_name: {target.semantic_name!r}\n"
            f"  flow_name hint: {target.flow_name!r}\n"
            f"  supi: {target.supi!r}\n"
            f"  goal: {target.goal}\n"
            f"  metric_focus: {target.metric_focus!r}\n\n"
            f"Candidate flows:\n{candidates_text}\n\n"
            'Return ONLY a JSON object with key "flow_ids" containing a list of matching '
            'flow_id strings. If nothing matches, return {"flow_ids": []}.'
        )
        try:
            response = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=30,
            )
            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)
            selected_ids = {
                str(fid).strip()
                for fid in (data.get("flow_ids") or [])
                if str(fid).strip()
            }
            matched = [
                row for row in capped
                if str(row.get("flow_id") or "").strip() in selected_ids
            ]
            if matched:
                _logger.debug(
                    "ControlSemanticsGrounder: LLM semantic match for %r → %s",
                    target.semantic_name,
                    [r.get("flow_id") for r in matched],
                )
            return matched
        except Exception as exc:
            _logger.warning(
                "ControlSemanticsGrounder: LLM semantic match failed for %r: %s",
                target.semantic_name,
                exc,
            )
            return []


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
