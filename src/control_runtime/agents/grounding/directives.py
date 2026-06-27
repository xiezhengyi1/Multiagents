from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .common import normalize_domain_evidence, normalize_requested_domains


class MainDirectiveExtractor:
    @staticmethod
    def _normalize_candidate_flows(values: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(values, list):
            return normalized
        for item in values:
            if not isinstance(item, dict):
                continue
            flow_id = str(item.get("flow_id") or "").strip()
            app_id = str(item.get("app_id") or "").strip()
            flow_name = str(item.get("flow_name") or "").strip()
            if not (flow_id or flow_name or app_id):
                continue
            normalized.append(dict(item))
        return normalized

    def extract_main_directives(self, context: str) -> Dict[str, Any]:
        text = str(context or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            payload = self._extract_markdown_payload(text)
        if not isinstance(payload, dict):
            return {}

        main_intent = payload.get("main_intent")
        if not isinstance(main_intent, dict):
            return {}

        requested_domains = normalize_requested_domains(main_intent.get("requested_domains"))
        domain_evidence = normalize_domain_evidence(main_intent.get("domain_evidence"))
        snapshot_summary = payload.get("snapshot_summary") if isinstance(payload.get("snapshot_summary"), dict) else {}
        candidate_flows = self._normalize_candidate_flows(main_intent.get("candidate_flows"))
        if not candidate_flows:
            candidate_flows = self._normalize_candidate_flows(payload.get("candidate_flows"))
        if not candidate_flows:
            candidate_flows = self._normalize_candidate_flows(snapshot_summary.get("candidate_flows"))
        return {
            "requested_domains": requested_domains,
            "domain_evidence": domain_evidence,
            "control_semantics": main_intent.get("control_semantics") if isinstance(main_intent.get("control_semantics"), dict) else {},
            "supi": str(main_intent.get("supi") or "").strip(),
            "retry_scope": str(main_intent.get("retry_scope") or "").strip(),
            "routing_decision": str(main_intent.get("routing_decision") or "").strip(),
            "routing_rationale": str(main_intent.get("routing_rationale") or "").strip(),
            "reuse_contract": main_intent.get("reuse_contract") if isinstance(main_intent.get("reuse_contract"), dict) else {},
            "intent_encoding_guidance": str(main_intent.get("intent_encoding_guidance") or "").strip(),
            "candidate_flows": candidate_flows,
        }

    @staticmethod
    def _extract_markdown_payload(text: str) -> Dict[str, Any]:
        def extract_json_line(label: str) -> Any:
            pattern = rf"(?im)^\s*-\s*{re.escape(label)}:\s*(.+?)\s*$"
            match = re.search(pattern, text)
            if not match:
                return {}
            raw = match.group(1).strip()
            try:
                return json.loads(raw)
            except Exception:
                return {}

        main_intent = extract_json_line("main_intent")
        snapshot_summary = extract_json_line("snapshot_summary")
        if not isinstance(main_intent, dict):
            main_intent = {}
        if not isinstance(snapshot_summary, dict):
            snapshot_summary = {}
        return {
            "main_intent": main_intent,
            "snapshot_summary": snapshot_summary,
        }
