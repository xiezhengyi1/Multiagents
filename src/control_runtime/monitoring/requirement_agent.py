from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from shared.agents import BaseAgent
from shared.runtime import RuntimeCache

from .contracts import MonitorAlert, RequirementDraft


def _metric_phrase(metric_name: str) -> str:
    mapping = {
        "telemetry.latency": "时延明显超出目标",
        "telemetry.jitter": "抖动已经不稳定",
        "telemetry.loss_rate": "丢包率偏高",
        "telemetry.throughput_ul": "上行吞吐不足",
        "telemetry.throughput_dl": "下行吞吐不足",
    }
    return mapping.get(metric_name, metric_name)


class AutonomousRequirementAgent(BaseAgent):
    agent_name = "autonomous_requirement_agent"

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        *,
        llm: Any = None,
    ) -> None:
        if llm is None:
            super().__init__(model_name=model_name, use_local_model=use_local_model)
        else:
            self._cache = RuntimeCache()
            self.model_name = model_name
            self.temperature = 0
            self.llm = llm

    def generate_requirement(
        self,
        alert: MonitorAlert,
        *,
        previous_user_intent: str = "",
        extra_context: Dict[str, Any] | None = None,
    ) -> RequirementDraft:
        if self.llm is not None:
            draft = self._generate_with_llm(
                alert,
                previous_user_intent=previous_user_intent,
                extra_context=extra_context,
            )
            if draft is not None:
                return draft
        return self._generate_template_requirement(
            alert,
            previous_user_intent=previous_user_intent,
            extra_context=extra_context,
        )

    def _generate_with_llm(
        self,
        alert: MonitorAlert,
        *,
        previous_user_intent: str = "",
        extra_context: Dict[str, Any] | None = None,
    ) -> RequirementDraft | None:
        prompt = self._build_prompt(
            alert,
            previous_user_intent=previous_user_intent,
            extra_context=extra_context,
        )
        try:
            response = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        except Exception:
            return None
        user_input = self._normalize_natural_language_response(response)
        if not user_input:
            return None
        routing_hint = self._build_routing_hint(
            alert,
            user_input=user_input,
            extra_context=extra_context,
            llm_rewritten=True,
        )
        return RequirementDraft(
            source_alert_id=alert.alert_id,
            snapshot_id=alert.snapshot_id,
            user_input=user_input,
            routing_hint=routing_hint,
        )

    @staticmethod
    def _normalize_natural_language_response(response: Any) -> str:
        text = str(getattr(response, "content", response) or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = AutonomousRequirementAgent._extract_user_input_from_json(text)
        if parsed:
            return parsed
        for prefix in ("user_input:", "requirement:", "request:", "需求：", "需求:"):
            if text.lower().startswith(prefix.lower()):
                return text[len(prefix) :].strip()
        return text.strip()

    @staticmethod
    def _extract_user_input_from_json(text: str) -> str:
        try:
            payload = json.loads(text)
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        for key in ("user_input", "requirement", "request", "natural_language_request"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _infer_fuzzy_semantics(alert: MonitorAlert, user_input: str) -> list[str]:
        semantics = ["用户体验明显变差", "恢复 SLA", "修正上一轮意图"]
        text = user_input.lower()
        if "持续" in user_input or "反复" in user_input:
            semantics.append("持续异常")
        if "低时延" in user_input or "latency" in text or "时延" in user_input:
            semantics.append("低时延保障")
        if "吞吐" in user_input or "throughput" in text:
            semantics.append("吞吐保障")
        if alert.severity == "critical":
            semantics.append("紧急纠偏")
        return list(dict.fromkeys(semantics))

    def _build_routing_hint(
        self,
        alert: MonitorAlert,
        *,
        user_input: str,
        extra_context: Dict[str, Any] | None,
        llm_rewritten: bool,
    ) -> Dict[str, Any]:
        return {
            "source": self.agent_name,
            "source_alert_id": alert.alert_id,
            "requested_domains": list(alert.suggested_domains or ["qos"]),
            "reuse_binding": alert.reuse_binding,
            "severity": alert.severity,
            "monitor_binding": {
                "supi": alert.supi,
                "app_id": alert.app_id,
                "app_name": alert.app_name,
                "flow_id": alert.flow_id,
                "flow_name": alert.flow_name,
            },
            "violated_metrics": list(alert.violated_metrics),
            "metric_deltas": dict(alert.metric_deltas),
            "context_policy": [
                "preserve_monitor_binding_when_reuse_binding_is_true",
                "treat_metric_deltas_as_runtime_evidence_not_user_claim",
                "prefer_qos_parameter_revision_before_regrounding_when_binding_is_stable",
            ],
            "fuzzy_semantics": self._infer_fuzzy_semantics(alert, user_input),
            "extra_context": self._compact_extra_context(extra_context or {}),
            "llm_rewritten": llm_rewritten,
            "llm_output_contract": "natural_language",
            "data_flywheel": {
                "trace_type": "monitor_alert_to_synthetic_user_requirement",
                "source_observation": "network_graph_flow_telemetry",
                "training_signal": "autonomous_reentry",
            },
        }

    @staticmethod
    def _compact_extra_context(extra_context: Dict[str, Any]) -> Dict[str, Any]:
        compacted: Dict[str, Any] = {}
        for key, value in extra_context.items():
            if key == "previous_control_context":
                text = str(value or "")
                if text:
                    compacted["previous_control_context_chars"] = len(text)
                    compacted["previous_control_context_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                continue
            compacted[key] = value
        return compacted

    def _build_prompt(
        self,
        alert: MonitorAlert,
        *,
        previous_user_intent: str = "",
        extra_context: Dict[str, Any] | None = None,
    ) -> str:
        return (
            "You are an autonomous requirement rewriting agent for a 6G control loop.\n"
            "Rewrite the monitor observation into a natural-language user request in Chinese.\n"
            "The request must sound like a user correcting or refining intent, not like a database record.\n"
            "Preserve explicit SUPI, flow/app names, and the monitor evidence. Do not invent identifiers.\n"
            "If reuse_binding is true, phrase the request so downstream agents preserve the monitored SUPI/app/flow binding unless runtime evidence proves it stale.\n"
            "Mention the violated user-facing experience or SLA metric, but keep raw metric tables out of the user sentence.\n"
            "Use fuzzy user-facing semantics when appropriate, such as experience degradation, unstable service, or restore SLA.\n"
            "This agent works with the original multi-agent control system to create a 数据飞轮/data flywheel: monitor alert -> synthetic user requirement -> autonomous reentry trace.\n"
            "不要输出 JSON，不要输出字段名，不要输出解释。只输出一段用户会说的自然语言需求。\n\n"
            f"Previous user intent: {previous_user_intent or 'N/A'}\n"
            f"Monitor alert: {json.dumps(alert.to_dict(), ensure_ascii=False)}\n"
            f"Extra context: {json.dumps(extra_context or {}, ensure_ascii=False)}"
        )

    def _generate_template_requirement(
        self,
        alert: MonitorAlert,
        *,
        previous_user_intent: str = "",
        extra_context: Dict[str, Any] | None = None,
    ) -> RequirementDraft:
        metric_text = "、".join(_metric_phrase(item) for item in alert.violated_metrics) or "业务体验异常"
        prior = str(previous_user_intent or "").strip()
        prior_clause = f"原始意图是“{prior}”，" if prior else ""
        flow_label = alert.flow_name or alert.flow_id or "目标业务流"
        supi_clause = f"（supi: {alert.supi}）" if alert.supi else ""
        binding_clause = "不要凭空更换业务对象，优先沿用已确认的 UE、app 和 flow 绑定，" if alert.reuse_binding else "请重新确认 UE、app 和 flow 绑定，"
        user_input = (
            f"{prior_clause}监控发现 {flow_label} {supi_clause} 的用户体验明显变差：{metric_text}。"
            f"请修正上一轮意图，{binding_clause}"
            "重新生成能让该业务流恢复 SLA 的自然语言控制需求；如果只是参数不合适，请优先重调 QoS，"
            "如果证据显示绑定已失效，再回到意图解析重新确认目标。"
        )
        routing_hint = self._build_routing_hint(
            alert,
            user_input=user_input,
            extra_context=extra_context,
            llm_rewritten=False,
        )
        return RequirementDraft(
            source_alert_id=alert.alert_id,
            snapshot_id=alert.snapshot_id,
            user_input=user_input,
            routing_hint=routing_hint,
        )
