from __future__ import annotations

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
        routing_hint = {
            "source": self.agent_name,
            "source_alert_id": alert.alert_id,
            "requested_domains": list(alert.suggested_domains or ["qos"]),
            "reuse_binding": alert.reuse_binding,
            "severity": alert.severity,
            "fuzzy_semantics": self._infer_fuzzy_semantics(alert, user_input),
            "extra_context": dict(extra_context or {}),
            "llm_rewritten": True,
            "llm_output_contract": "natural_language",
            "data_flywheel": {
                "trace_type": "monitor_alert_to_synthetic_user_requirement",
                "source_observation": "network_graph_flow_telemetry",
                "training_signal": "autonomous_reentry",
            },
        }
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
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1].strip()
        return text.strip()

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
        routing_hint = {
            "source": self.agent_name,
            "source_alert_id": alert.alert_id,
            "requested_domains": list(alert.suggested_domains or ["qos"]),
            "reuse_binding": alert.reuse_binding,
            "severity": alert.severity,
            "fuzzy_semantics": [
                "用户体验明显变差",
                "恢复 SLA",
                "修正上一轮意图",
            ],
            "extra_context": dict(extra_context or {}),
            "llm_rewritten": False,
            "llm_output_contract": "natural_language",
            "data_flywheel": {
                "trace_type": "monitor_alert_to_synthetic_user_requirement",
                "source_observation": "network_graph_flow_telemetry",
                "training_signal": "autonomous_reentry",
            },
        }
        return RequirementDraft(
            source_alert_id=alert.alert_id,
            snapshot_id=alert.snapshot_id,
            user_input=user_input,
            routing_hint=routing_hint,
        )
