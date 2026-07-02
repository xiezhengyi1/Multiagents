from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from shared.agents import BaseAgent, coerce_structured_response
from shared.runtime import ArtifactWorkerMixin, ContextPolicy, ToolLoopExecutionError, extract_tool_calls, extract_tool_results
from shared.logging import log_event, log_timing

from ...context.prompts import SinglePromptBuilder
from ...domain.collaboration import PlanningContext, PlanningRequest
from ...domain.control_plane import ControlDomain, GlobalControlIntent, MainRoundStrategy, ObjectiveProfile
from ...domain.policy_plan import FlowSelector, OperationIntent, PlanningRationale, PolicyDraft, PolicyPlanDraft
from ...integrations.storage import get_latest_snapshot_metadata
from ..planning.planning_validation import normalize_policy_plan_draft
from ..planning.compiler import OptimizationStrategyCompiler
from ..planning.tool_result_adapter import extract_planning_tool_evidence
from .contracts import SingleAgentRoundDecision
from .tools import build_single_agent_tools


class _SingleAgentChatDeepSeekMixin:
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # DeepSeek thinking mode requires replaying the assistant reasoning_content
        # verbatim on subsequent turns. langchain_deepseek stores it on the
        # AIMessage, but the inherited OpenAI payload conversion drops it.
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload
        for source_message, payload_message in zip(messages, payload_messages):
            if not isinstance(source_message, AIMessage):
                continue
            if payload_message.get("role") != "assistant":
                continue
            reasoning_content = source_message.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                payload_message["reasoning_content"] = reasoning_content
        return payload


class _SingleAgentOutputNormalizingLlm:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def __getattr__(self, item: str) -> Any:
        return getattr(self._llm, item)

    def bind_tools(self, tools: Any) -> "_SingleAgentOutputNormalizingLlm":
        return _SingleAgentOutputNormalizingLlm(self._llm.bind_tools(tools))

    def invoke(self, conversation: Any, **kwargs: Any) -> Any:
        message = self._llm.invoke(conversation, **kwargs)
        if isinstance(message, AIMessage):
            return SingleControlAgent._normalize_final_ai_message(message, list(conversation or []))
        return message

    async def ainvoke(self, conversation: Any, **kwargs: Any) -> Any:
        message = await self._llm.ainvoke(conversation, **kwargs)
        if isinstance(message, AIMessage):
            return SingleControlAgent._normalize_final_ai_message(message, list(conversation or []))
        return message


class SingleControlAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "single_control"
    DEEPSEEK_MODEL = "deepseek-v4-flash"
    LARGE_MODEL_TOOL_RESULT_LIMITS = {
        "preview_qos_optimizer": 32000,
        "get_sm_ue_context": 8000,
        "get_sm_ue_flow_catalog": 32000,
        "get_am_policy_context": 8000,
        "fetch_qos_network_status": 32000,
        "inspect_mobility_ue_policies": 8000,
        "search_semantic_knowledge": 8000,
        "get_knowledge_by_key": 8000,
    }
    SMALL_MODEL_TOOL_RESULT_LIMITS = {
        "preview_qos_optimizer": 12000,
        "get_sm_ue_context": 4000,
        "get_sm_ue_flow_catalog": 12000,
        "get_am_policy_context": 4000,
        "fetch_qos_network_status": 8000,
        "inspect_mobility_ue_policies": 6000,
        "search_semantic_knowledge": 3000,
        "get_knowledge_by_key": 3000,
    }
    FINAL_PRODUCT_FIELDS = {
        "supi",
        "requested_domains",
        "objective_profile_hint",
        "sm_policies",
        "am_policy",
        "ursp_policies",
    }
    KNOWLEDGE_HINT_TOKENS = (
        "3gpp",
        "29.5",
        "29.507",
        "29.512",
        "npcf_",
        "smpolicydecision",
        "policyassociationupdaterequest",
        "ursp",
        "requesttrigger",
    )

    def __init__(
        self,
        model_name: str = DEEPSEEK_MODEL,
        *,
        use_local_model: bool = False,
        rag_enabled: bool = True,
    ) -> None:
        use_deepseek_model = str(model_name or "").strip().lower().startswith("deepseek")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") if use_deepseek_model else None
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL") if use_deepseek_model else None
        if not use_local_model and use_deepseek_model:
            if not deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required for SingleControlAgent")
            if not deepseek_base_url:
                raise RuntimeError("DEEPSEEK_BASE_URL is required for SingleControlAgent")
        super().__init__(
            model_name=model_name,
            use_local_model=use_local_model,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
        )
        if not use_local_model and use_deepseek_model:
            from langchain_deepseek import ChatDeepSeek
            class _SingleAgentChatDeepSeek(_SingleAgentChatDeepSeekMixin, ChatDeepSeek):
                pass
            raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120")
            raw_max_retries = os.getenv("OPENAI_MAX_RETRIES", "2")
            self.llm = _SingleAgentChatDeepSeek(
                model=model_name,
                temperature=0,
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                timeout=float(raw_timeout),
                max_retries=int(raw_max_retries),
            )
        self.agent_name = "single_control"
        self.rag_enabled = rag_enabled
        self.plan_compiler = OptimizationStrategyCompiler()
        self.initialize_agent_runtime(logger_color="\033[96m")

    def create_json_agent(self, **kwargs: Any) -> Any:
        original_llm = self.llm
        self.llm = _SingleAgentOutputNormalizingLlm(original_llm)
        try:
            return super().create_json_agent(**kwargs)
        finally:
            self.llm = original_llm

    @staticmethod
    def _uses_small_model_context(model_name: str) -> bool:
        normalized = str(model_name or "").strip().lower()
        return not normalized.startswith("deepseek")

    @classmethod
    def _tool_result_limits_for_model(cls, model_name: str) -> Dict[str, int]:
        if cls._uses_small_model_context(model_name):
            return dict(cls.SMALL_MODEL_TOOL_RESULT_LIMITS)
        return dict(cls.LARGE_MODEL_TOOL_RESULT_LIMITS)

    @classmethod
    def _tool_context_policy_for_model(cls, model_name: str) -> ContextPolicy:
        small_model = cls._uses_small_model_context(model_name)
        limits = cls._tool_result_limits_for_model(model_name)
        return ContextPolicy(
            default_tool_result_chars=4000 if small_model else 8000,
            default_tool_result_tokens=1000 if small_model else 2000,
            tool_result_char_limits=limits,
            tool_result_token_limits={
                name: 1000 if small_model else 2000
                for name in limits
            },
            recent_tool_results_per_tool=1,
        )

    def plan_round(
        self,
        *,
        user_input: str,
        context: str = "",
        session_id: str = "",
        snapshot_id: str = "",
        round_index: int = 1,
        feedback_context: str = "",
        round_traces: Optional[List[Dict[str, Any]]] = None,
        previous_mediator_decision: Optional[Dict[str, Any]] = None,
        allow_user_interaction: bool = False,
    ) -> tuple[GlobalControlIntent, OperationIntent, PolicyPlanDraft]:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(self.logger, "single_control_round_plan_start", session_id=session_id, round_index=round_index)

        supi_match = re.search(r"(?i)(imsi-\d{5,})", str(user_input or ""))
        supi = supi_match.group(1) if supi_match else ""
        evidence = {
            "user_input": str(user_input or "").strip(),
            "supi": supi,
        }
        token_budget, token_counter = self._resolve_token_context()
        context_policy = self._tool_context_policy_for_model(self.model_name)
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            supi=supi or None,
            thread_id=session_id,
            allow_user_interaction=allow_user_interaction,
            token_budget=token_budget,
            token_counter=token_counter,
        )
        requested_domains = self._hint_requested_domains(user_input)
        allow_knowledge_tools = self._should_allow_knowledge_tools(user_input)
        advisor_agent = self.create_json_agent(
            tools=build_single_agent_tools(
                rag_enabled=self.rag_enabled,
                requested_domains=requested_domains,
                allow_knowledge_tools=allow_knowledge_tools,
            ),
            system_prompt=SinglePromptBuilder().system_prompt(),
            response_model=SingleAgentRoundDecision,
            max_iterations=12,
            tool_error_mode="raise",
            forbid_duplicate_tool_calls=True,
            max_tool_result_chars=context_policy.default_tool_result_chars,
            tool_result_limits=context_policy.tool_result_char_limits,
            max_tool_result_tokens=context_policy.default_tool_result_tokens,
            tool_result_token_limits=context_policy.tool_result_token_limits,
            context_policy=context_policy,
        )
        prompt = self._build_round_prompt(
            user_input=user_input,
            evidence=evidence,
            context=context,
            feedback_context=feedback_context,
        )
        result = self._invoke_round_advisor(
            advisor_agent=advisor_agent,
            prompt=prompt,
            runtime_context=runtime_context,
        )
        decision = coerce_structured_response(
            result,
            SingleAgentRoundDecision,
            error_message="Single control agent returned no unified round decision",
        )

        planning_tool_evidence = extract_planning_tool_evidence(advisor_result=result)
        operation_intent = self._build_compat_operation_intent_from_final_product(
            decision=decision,
            advisor_result=result,
            planning_tool_evidence=planning_tool_evidence,
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
        global_intent = self._build_global_intent(
            decision=decision,
            operation_intent=operation_intent,
            session_id=session_id,
            snapshot_id=snapshot_id,
            user_input=user_input,
        )
        planning_request = self._build_planning_request(
            global_intent=global_intent,
            operation_intent=operation_intent,
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=round_index,
            feedback_context=feedback_context,
            round_traces=round_traces or [],
            previous_mediator_decision=previous_mediator_decision or {},
        )
        policy_plan = self._assemble_policy_plan_from_decision(
            decision=decision,
            planning_request=planning_request,
            planning_tool_evidence=planning_tool_evidence,
        )
        log_timing(self.logger, "single_control_round_plan_total", time.perf_counter() - total_start, status="success")
        return global_intent, operation_intent, policy_plan

    def _build_compat_operation_intent_from_final_product(
        self,
        *,
        decision: SingleAgentRoundDecision,
        advisor_result: Dict[str, Any],
        planning_tool_evidence: Dict[str, Any],
        user_input: str,
        session_id: str,
        snapshot_id: str,
    ) -> OperationIntent:
        requested_domains = list(decision.requested_domains or [])
        optimizer_preview = self.plan_compiler.advisor_validator._latest_optimizer_preview(planning_tool_evidence)
        optimizer_flow_ids = self._optimizer_flow_ids(optimizer_preview)
        policy_flow_ids: List[str] = []
        for policy_details in decision.sm_policies or []:
            normalized = self._normalize_sm_policy_payload(policy_details)
            inferred = self._extract_sm_policy_flow_id(normalized)
            if inferred and inferred in optimizer_flow_ids:
                candidate = inferred
            elif len(optimizer_flow_ids) == 1:
                candidate = optimizer_flow_ids[0]
            else:
                candidate = inferred
            if candidate and candidate not in policy_flow_ids:
                policy_flow_ids.append(candidate)
        for flow_id in optimizer_flow_ids:
            if flow_id and flow_id not in policy_flow_ids:
                policy_flow_ids.append(flow_id)

        catalog_rows = self._catalog_rows_by_flow_id(advisor_result)
        flows: List[FlowSelector] = []
        for flow_id in policy_flow_ids:
            row = catalog_rows.get(flow_id, {})
            payload = self._flow_selector_payload(supi=decision.supi, flow_id=flow_id, row=row)
            if not payload.get("app_id"):
                payload["app_id"] = self._extract_app_id_for_flow_from_policies(decision.sm_policies, flow_id=flow_id)
            flows.append(FlowSelector.model_validate(payload))

        app_id = next((str(flow.app_id or "").strip() for flow in flows if str(flow.app_id or "").strip()), "")
        domain_evidence = {
            domain: [f"single_agent_final_{domain}_product"]
            for domain in requested_domains
        }
        return OperationIntent(
            session_id=str(session_id or "").strip(),
            snapshot_id=str(snapshot_id or "").strip(),
            supi=str(decision.supi or "").strip(),
            app_id=app_id,
            raw_input=str(user_input or "").strip(),
            resolution_status="final_product",
            requested_domains=requested_domains,
            grounded_requested_domains=requested_domains,
            domain_evidence=domain_evidence,
            flows=flows,
        )

    @classmethod
    def _catalog_rows_by_flow_id(cls, advisor_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        rows_by_flow_id: Dict[str, Dict[str, Any]] = {}
        for result in extract_tool_results((advisor_result or {}).get("messages") or []):
            if str(result.get("name") or "").strip() != "get_sm_ue_flow_catalog":
                continue
            payload = cls._parse_json_content(result.get("content"))
            if not isinstance(payload, dict):
                continue
            rows = payload.get("flow_catalog") if isinstance(payload.get("flow_catalog"), list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                flow_id = str(row.get("flow_id") or "").strip()
                if flow_id:
                    rows_by_flow_id[flow_id] = row
        return rows_by_flow_id

    @classmethod
    def _extract_app_id_for_flow_from_policies(cls, policies: List[Dict[str, Any]], *, flow_id: str) -> str:
        for policy_details in policies or []:
            normalized = cls._normalize_sm_policy_payload(policy_details)
            policy_flow_id = cls._extract_sm_policy_flow_id(normalized)
            if policy_flow_id and policy_flow_id != flow_id:
                continue
            app_id = cls._extract_sm_policy_app_id(normalized)
            if app_id:
                return app_id
        return ""

    @staticmethod
    def _build_round_prompt(*, user_input: str, evidence: Dict[str, Any], context: str, feedback_context: str) -> str:
        feedback_block = ""
        if feedback_context and feedback_context not in context:
            feedback_block = (
                "Feedback context:\n"
                f"{feedback_context}\n\n"
            )
        return (
            "User request:\n"
            f"{user_input}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
            "Round context:\n"
            f"{context or 'N/A'}\n\n"
            f"{feedback_block}"
            "Task:\n"
            "- Stage 1: infer requested_domains and ground the required identifiers.\n"
            "- Stage 2: use planning tools in the same loop and return executable policy payloads.\n"
            "- If qos is active, flows and sm_policies must refer to the same grounded flow_ids.\n"
            "- If mobility is active, am_policy must be backed by grounded UE mobility evidence.\n"
            "- Return one final policy product JSON object only.\n"
        )

    @classmethod
    def _normalize_final_ai_message(cls, message: AIMessage, conversation: List[Any]) -> AIMessage:
        if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None):
            return message
        payload = cls._parse_json_content(message.content)
        if isinstance(payload, dict):
            normalized = cls._normalize_final_dict_payload(payload, conversation)
        elif isinstance(payload, list) and payload:
            normalized = cls._normalize_final_list_payload(payload, conversation)
        else:
            return message
        if normalized is None:
            return message
        return AIMessage(
            content=json.dumps(normalized, ensure_ascii=False),
            additional_kwargs=dict(message.additional_kwargs or {}),
            tool_calls=list(getattr(message, "tool_calls", None) or []),
            invalid_tool_calls=list(getattr(message, "invalid_tool_calls", None) or []),
            response_metadata=dict(message.response_metadata or {}),
            id=message.id,
        )

    @classmethod
    def _normalize_final_dict_payload(cls, payload: Dict[str, Any], conversation: List[Any]) -> Optional[Dict[str, Any]]:
        candidate = cls._sanitize_round_decision_payload(payload)
        try:
            return cls._dump_final_product(SingleAgentRoundDecision.model_validate(candidate))
        except Exception:
            if cls._latest_successful_optimizer_call(conversation) is None:
                return None
            return cls._normalize_final_list_payload([payload], conversation)

    @classmethod
    def _dump_final_product(cls, decision: SingleAgentRoundDecision) -> Dict[str, Any]:
        return decision.model_dump(
            mode="json",
            include=cls.FINAL_PRODUCT_FIELDS,
            exclude_defaults=True,
            exclude_none=True,
        )

    @classmethod
    def _sanitize_round_decision_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate = dict(payload)
        for envelope_key in ("single_agent_decision", "single_agent_round_decision", "round_decision", "response"):
            wrapped = candidate.get(envelope_key)
            if isinstance(wrapped, dict):
                candidate = dict(wrapped)
                break
        if "intent" not in candidate and candidate.get("intent_classification") is not None:
            candidate["intent"] = str(candidate.get("intent_classification") or "").strip()
        if "sm_policies" not in candidate and isinstance(candidate.get("smPolicies"), list):
            candidate["sm_policies"] = candidate.get("smPolicies")
        if "am_policy" not in candidate and isinstance(candidate.get("amPolicy"), dict):
            candidate["am_policy"] = candidate.get("amPolicy")
        if "ursp_policies" not in candidate and "ursp_rules" in candidate:
            candidate["ursp_policies"] = candidate.get("ursp_rules")
        if isinstance(candidate.get("sm_policies"), list):
            candidate["sm_policies"] = [
                normalized
                for item in candidate.get("sm_policies") or []
                if isinstance(item, dict)
                for normalized in [cls._normalize_sm_policy_payload(item)]
                if normalized
            ]
        allowed = set(SingleAgentRoundDecision.model_fields)
        return {key: value for key, value in candidate.items() if key in allowed}

    @staticmethod
    def _parse_json_content(content: Any) -> Any:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                    parts.append(str(block.get("text") or ""))
            text = "".join(parts).strip()
        else:
            text = str(content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if not text:
            return None
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
            fenced = match.group(1).strip()
            if not fenced:
                continue
            try:
                return json.loads(fenced)
            except Exception:
                continue
        try:
            return json.loads(text)
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                return parsed
            except Exception:
                continue
        return None

    @classmethod
    def _normalize_final_list_payload(cls, payload: List[Any], conversation: List[Any]) -> Optional[Dict[str, Any]]:
        if len(payload) == 1 and isinstance(payload[0], dict):
            normalized_single = cls._normalize_final_dict_payload(payload[0], [])
            if normalized_single is not None:
                return normalized_single
        optimizer_call = cls._latest_successful_optimizer_call(conversation)
        if optimizer_call is None:
            return None
        args = dict(optimizer_call.get("args") or {})
        requested_domains = [
            str(item or "").strip().lower()
            for item in (args.get("requested_domains") or ["qos"])
            if str(item or "").strip()
        ]
        if requested_domains != ["qos"]:
            return None
        sm_policies = cls._extract_sm_policy_list(payload)
        if not sm_policies:
            return None
        supi = str(args.get("supi") or "").strip()
        flow_ids = [str(item or "").strip() for item in (args.get("flow_ids") or []) if str(item or "").strip()]
        if not supi or not flow_ids:
            return None
        return {
            "requested_domains": ["qos"],
            "supi": supi,
            "objective_profile_hint": str(args.get("objective_profile") or "balanced").strip().lower() or "balanced",
            "sm_policies": sm_policies,
        }

    @staticmethod
    def _extract_sm_policy_list(payload: List[Any]) -> List[Dict[str, Any]]:
        policies: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                return []
            if isinstance(item.get("sm_policies"), list):
                policies.extend(
                    SingleControlAgent._normalize_sm_policy_payload(policy)
                    for policy in item.get("sm_policies") or []
                    if isinstance(policy, dict)
                )
                continue
            if isinstance(item.get("smPolicies"), list):
                policies.extend(
                    SingleControlAgent._normalize_sm_policy_payload(policy)
                    for policy in item.get("smPolicies") or []
                    if isinstance(policy, dict)
                )
                continue
            policies.append(SingleControlAgent._normalize_sm_policy_payload(item))
        return [policy for policy in policies if policy]

    @classmethod
    def _normalize_sm_policy_payload(cls, policy_details: Dict[str, Any]) -> Dict[str, Any]:
        data = copy.deepcopy(policy_details)
        for wrapper_key in ("policy_decision", "policyDecision", "decision", "smPolicyDecision", "sm_policy_decision"):
            wrapped = data.get(wrapper_key)
            if not isinstance(wrapped, dict):
                continue
            if not any(key in wrapped for key in ("pccRules", "pcc_rules", "qosDecs", "qos_decs")):
                continue
            merged = copy.deepcopy(wrapped)
            for carry_key in ("supi", "flow_id", "flowId", "app_id", "appId"):
                if carry_key in data and carry_key not in merged:
                    merged[carry_key] = data[carry_key]
            data = merged
            break

        pcc_rules = cls._normalize_pcc_rules(data.get("pccRules") or data.get("pcc_rules"))
        if pcc_rules:
            data["pccRules"] = pcc_rules
        qos_decs = cls._normalize_qos_decs(data.get("qosDecs") or data.get("qos_decs"))
        if qos_decs:
            data["qosDecs"] = qos_decs
        if isinstance(data.get("policyCtrlReqTriggers"), list):
            data["policyCtrlReqTriggers"] = cls._normalize_sm_policy_triggers(data.get("policyCtrlReqTriggers"))
        sess_rules = data.get("sessRules") or data.get("sess_rules")
        if not isinstance(sess_rules, dict) and isinstance(data.get("sessionRule"), dict):
            sess_rule = copy.deepcopy(data["sessionRule"])
            sess_rule_id = str(sess_rule.get("sessRuleId") or sess_rule.get("sessionRuleId") or "sess-rule-1").strip()
            sess_rule["sessRuleId"] = sess_rule_id
            data["sessRules"] = {sess_rule_id: sess_rule}
        return data

    @staticmethod
    def _normalize_pcc_rules(value: Any) -> Dict[str, Any]:
        if isinstance(value, list):
            rules: Dict[str, Any] = {}
            for index, item in enumerate(value, start=1):
                if isinstance(item, str) and item.strip():
                    rules[item.strip()] = {"pccRuleId": item.strip()}
                    continue
                if not isinstance(item, dict):
                    continue
                rule_id = str(item.get("pccRuleId") or item.get("id") or f"pcc-rule-{index}").strip()
                rules[rule_id] = item
            value = rules
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for rule_key, rule_value in value.items():
            if not isinstance(rule_value, dict):
                continue
            rule = copy.deepcopy(rule_value)
            rule_id = str(rule.get("pccRuleId") or rule_key or "").strip()
            if rule_id:
                rule["pccRuleId"] = rule_id
            if "flowInfos" not in rule and isinstance(rule.get("flowDescriptions"), list):
                rule["flowInfos"] = [
                    {"flowDescription": str(item)}
                    for item in rule.get("flowDescriptions") or []
                    if str(item or "").strip()
                ]
            if "flowInfos" in rule and isinstance(rule["flowInfos"], list):
                flow_infos: List[Dict[str, Any]] = []
                for item in rule["flowInfos"]:
                    if not isinstance(item, dict):
                        continue
                    flow_info = copy.deepcopy(item)
                    if "flowDescription" not in flow_info and flow_info.get("flowId"):
                        flow_info["flowDescription"] = str(flow_info.get("flowId") or "").strip()
                    flow_infos.append(flow_info)
                rule["flowInfos"] = flow_infos
            if "refQosData" not in rule:
                refs = rule.get("qosDataRefs") or rule.get("qosData") or rule.get("qosDecs")
                if isinstance(refs, list):
                    rule["refQosData"] = [
                        str(item.get("qosId") if isinstance(item, dict) else item).strip()
                        for item in refs
                        if str(item.get("qosId") if isinstance(item, dict) else item).strip()
                    ]
                elif isinstance(refs, str) and refs.strip():
                    rule["refQosData"] = [refs.strip()]
                elif isinstance(refs, dict):
                    qos_id = str(refs.get("qosId") or "").strip()
                    if qos_id:
                        rule["refQosData"] = [qos_id]
            normalized[str(rule_key)] = rule
        return normalized

    @staticmethod
    def _normalize_qos_decs(value: Any) -> Dict[str, Any]:
        if isinstance(value, list):
            normalized: Dict[str, Any] = {}
            for index, item in enumerate(value, start=1):
                if isinstance(item, str) and item.strip():
                    normalized[item.strip()] = {"qosId": item.strip()}
                    continue
                if not isinstance(item, dict):
                    continue
                qos = copy.deepcopy(item)
                qos_id = str(qos.get("qosId") or qos.get("id") or f"qos-{index}").strip()
                qos["qosId"] = qos_id
                normalized[qos_id] = qos
            return normalized
        if not isinstance(value, dict):
            return {}
        normalized = {}
        for qos_key, qos_value in value.items():
            if not isinstance(qos_value, dict):
                continue
            qos = copy.deepcopy(qos_value)
            qos_id = str(qos.get("qosId") or qos_key or "").strip()
            if qos_id:
                qos["qosId"] = qos_id
            normalized[str(qos_key)] = qos
        return normalized

    @staticmethod
    def _normalize_sm_policy_triggers(value: Any) -> List[str]:
        from model.PolicyTrigger import PolicyTrigger

        aliases = {
            "QOS_NOTIF_CONTROL": "QOS_NOTIF",
            "QOS_NOTIFICATION": "QOS_NOTIF",
        }
        allowed = {item.value for item in PolicyTrigger}
        normalized: List[str] = []
        for item in value or []:
            trigger = aliases.get(str(item or "").strip(), str(item or "").strip())
            if trigger in allowed and trigger not in normalized:
                normalized.append(trigger)
        return normalized

    @classmethod
    def _extract_sm_policy_flow_id(cls, policy_details: Any) -> str:
        from ...domain.policy_compiler import PolicyCompiler

        data = cls._normalize_sm_policy_payload(policy_details) if isinstance(policy_details, dict) else {}
        explicit = str(PolicyCompiler.extract_flow_id(data) or "").strip()
        if explicit:
            return explicit
        pcc_rules = data.get("pccRules")
        if isinstance(pcc_rules, dict):
            for rule in pcc_rules.values():
                if not isinstance(rule, dict):
                    continue
                for key in ("flow_id", "flowId"):
                    flow_id = str(rule.get(key) or "").strip()
                    if flow_id:
                        return flow_id
                flow_infos = rule.get("flowInfos")
                if isinstance(flow_infos, list):
                    for info in flow_infos:
                        if not isinstance(info, dict):
                            continue
                        flow_id = str(info.get("flow_id") or info.get("flowId") or "").strip()
                        if flow_id:
                            return flow_id
        return ""

    @classmethod
    def _extract_sm_policy_app_id(cls, policy_details: Any) -> str:
        data = cls._normalize_sm_policy_payload(policy_details) if isinstance(policy_details, dict) else {}
        for key in ("app_id", "appId"):
            app_id = str(data.get(key) or "").strip()
            if app_id:
                return app_id
        pcc_rules = data.get("pccRules")
        if isinstance(pcc_rules, dict):
            for rule in pcc_rules.values():
                if isinstance(rule, dict):
                    app_id = str(rule.get("app_id") or rule.get("appId") or "").strip()
                    if app_id:
                        return app_id
        return ""

    @staticmethod
    def _latest_successful_optimizer_call(conversation: List[Any]) -> Optional[Dict[str, Any]]:
        calls_by_id = {
            str(call.get("id") or "").strip(): call
            for call in extract_tool_calls(conversation)
            if str(call.get("name") or "").strip() == "preview_qos_optimizer"
        }
        successful_ids = {
            str(result.get("tool_call_id") or "").strip()
            for result in extract_tool_results(conversation)
            if str(result.get("name") or "").strip() == "preview_qos_optimizer"
            and str(result.get("status") or "success").strip().lower() != "error"
        }
        for call_id, call in reversed(list(calls_by_id.items())):
            if call_id in successful_ids:
                return call
        return None

    @staticmethod
    def _flow_selector_payload(*, supi: str, flow_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        service = row.get("service") if isinstance(row.get("service"), dict) else {}
        sla = row.get("sla") if isinstance(row.get("sla"), dict) else {}
        traffic = row.get("traffic") if isinstance(row.get("traffic"), dict) else {}
        flow_name = str(row.get("flow_name") or row.get("name") or flow_id).strip()
        return {
            "supi": supi,
            "app_id": str(row.get("app_id") or "").strip(),
            "app_name": str(row.get("app_name") or "").strip() or None,
            "flow_id": flow_id,
            "target_type": "flow",
            "name": flow_name,
            "service_type": str(service.get("service_type") or "").strip() or None,
            "service_type_id": service.get("service_type_id"),
            "bw_ul": sla.get("bandwidth_ul"),
            "bw_dl": sla.get("bandwidth_dl"),
            "gbr_ul": sla.get("guaranteed_bandwidth_ul"),
            "gbr_dl": sla.get("guaranteed_bandwidth_dl"),
            "lat": sla.get("latency"),
            "loss_req": sla.get("loss_rate"),
            "jitter_req": sla.get("jitter"),
            "priority": sla.get("priority"),
            "description": flow_name,
            "five_tuple": list(traffic.get("five_tuple")) if isinstance(traffic.get("five_tuple"), (list, tuple)) else None,
            "resolution_status": "resolved",
        }

    @classmethod
    def _hint_requested_domains(cls, user_input: str) -> Optional[List[str]]:
        # Keep single-agent domain routing model-driven.  The prompt owns the
        # domain boundary rules; pre-filtering tools here made natural-language
        # exclusions brittle and could expose the wrong tool subset.
        del user_input
        return None

    @classmethod
    def _should_allow_knowledge_tools(cls, user_input: str) -> bool:
        lowered = str(user_input or "").lower()
        return any(token in lowered for token in cls.KNOWLEDGE_HINT_TOKENS)

    @staticmethod
    def _invoke_round_advisor(*, advisor_agent: Any, prompt: str, runtime_context: Any) -> Dict[str, Any]:
        try:
            return advisor_agent.invoke({"messages": [{"role": "user", "content": prompt}]}, context=runtime_context)
        except Exception as exc:
            if isinstance(exc, ToolLoopExecutionError):
                message = str(exc)
                if "max iterations" in message.lower():
                    raise RuntimeError(f"Single control round advisor did not converge: {message}") from exc
                if "SingleAgentRoundDecision" in message:
                    raise RuntimeError(
                        "Single control round advisor returned an invalid unified round decision. "
                        f"Validator error: {message}"
                    ) from exc
            raise RuntimeError(f"Single control round advisor invocation failed: {exc}") from exc

    @staticmethod
    def _validate_policy_presence(*, decision: SingleAgentRoundDecision, planning_request: PlanningRequest) -> None:
        requested_domains = {
            str(domain or "").strip().lower()
            for domain in (planning_request.operation_intent.requested_domains or [])
            if str(domain or "").strip()
        }
        has_sm = bool(decision.sm_policies)
        has_am = decision.am_policy is not None
        has_ursp = bool(decision.ursp_policies)

        if not has_sm and not has_am and not has_ursp:
            raise RuntimeError("Single control round advisor returned empty policy output")
        if "qos" in requested_domains and not has_sm:
            raise RuntimeError("Single control round advisor omitted sm_policies for a qos request")
        if "qos" not in requested_domains and has_sm:
            raise RuntimeError("Single control round advisor returned sm_policies for a non-qos request")
        if "mobility" in requested_domains and not has_am:
            raise RuntimeError("Single control round advisor omitted am_policy for a mobility request")
        if "mobility" not in requested_domains and has_am:
            raise RuntimeError("Single control round advisor returned am_policy for a non-mobility request")

    def _assemble_policy_plan_from_decision(
        self,
        *,
        decision: SingleAgentRoundDecision,
        planning_request: PlanningRequest,
        planning_tool_evidence: Dict[str, Any],
    ) -> PolicyPlanDraft:
        self._validate_policy_presence(decision=decision, planning_request=planning_request)
        optimizer_preview = self.plan_compiler.advisor_validator._latest_optimizer_preview(planning_tool_evidence)
        mobility_context = self.plan_compiler.advisor_validator._latest_mobility_context(planning_tool_evidence)

        active_domains = {
            str(item).strip().lower()
            for item in (planning_request.context.active_domains or [])
            if str(item).strip()
        }
        if ControlDomain.QOS.value in active_domains:
            self.plan_compiler.advisor_validator._validate_optimizer_preview(optimizer_preview)
        if ControlDomain.MOBILITY.value in active_domains and decision.am_policy is not None and not mobility_context:
            raise ValueError("am_policy compilation requires grounded mobility context")

        planning_metadata = {
            "planning_mode": "single_agent_direct_pcf",
            "requested_domains": list(planning_request.context.active_domains or []),
            "main_retry_scope": str(planning_request.context.main_retry_scope or "").strip(),
            "objective_breakdown": dict(optimizer_preview.get("objective_breakdown") or {}) if isinstance(optimizer_preview, dict) else {},
            "revision_requests": planning_request.context.revision_requests or [],
            "unified_constraints": planning_request.context.unified_constraints or {},
            "optimizer_cross_domain_verdicts": [
                item
                for item in ((optimizer_preview.get("cross_domain_verdicts") if isinstance(optimizer_preview, dict) else []) or [])
            ],
            "snapshot_writeback_patch": self.plan_compiler.artifact_compiler._build_snapshot_writeback_patch(optimizer_preview),
        }

        plan = PolicyPlanDraft(
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_metadata=planning_metadata,
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.objective_profile.get("profile_name")
                    or decision.objective_profile_hint
                    or ""
                ).strip(),
                objective_tradeoff_summary=str(planning_metadata["objective_breakdown"] or "").strip(),
                decisive_evidence=[
                    item
                    for item in [
                        "tool:preview_qos_optimizer" if decision.sm_policies else "",
                        "tool:inspect_mobility_ue_policies" if decision.am_policy is not None else "",
                        "mediator_constraints" if planning_request.context.unified_constraints else "",
                        "revision_requests" if planning_request.context.revision_requests else "",
                    ]
                    if item
                ],
                active_constraints=[
                    str(item)
                    for item in self.plan_compiler.artifact_compiler._normalized_hard_constraints(
                        planning_request.context.unified_constraints
                    )
                    if str(item).strip()
                ],
                explanation="Single-agent final policy product compiled from grounded tool evidence.",
                rejected_alternatives=[],
            ),
            all_policies=[],
        )

        for index, policy_details in enumerate(decision.sm_policies, start=1):
            normalized_policy_details = self._normalize_sm_policy_payload(policy_details)
            flow_id = self._resolve_pcflow_id(
                normalized_policy_details,
                decision=decision,
                planning_request=planning_request,
                optimizer_preview=optimizer_preview,
            )
            flow_ctx = self._resolve_single_agent_flow_context(
                planning_request=planning_request,
                decision=decision,
                flow_id=flow_id,
                policy_details=normalized_policy_details,
            )
            app_id = str(
                flow_ctx.app_id
                or self._extract_sm_policy_app_id(normalized_policy_details)
                or planning_request.operation_intent.app_id
                or ""
            ).strip()
            resource_keys = self.plan_compiler.advisor_validator._extract_qos_resource_keys(optimizer_preview, flow_id=flow_id)
            if not resource_keys:
                raise ValueError(f"optimizer preview does not contain a grounded QoS assignment for flow_id={flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[f"Apply PCF SM policy for {flow_id}"],
                    supi=str(flow_ctx.supi or planning_request.operation_intent.supi or "").strip(),
                    app_id=app_id,
                    flow_id=flow_id,
                    target_type="flow",
                    policy_id=f"smp-{app_id or 'app'}-{flow_id or index}",
                    policy_type="SmPolicyDecision",
                    resource_keys=resource_keys,
                    policy_details=normalized_policy_details,
                )
            )

        if decision.am_policy is not None:
            normalized_am_policy = self._normalize_am_policy_payload(
                decision.am_policy,
                planning_request=planning_request,
                mobility_context=mobility_context,
            )
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id="",
                    flow_id=None,
                    target_type="ue",
                    policy_id=self.plan_compiler.artifact_compiler._resolve_am_association_id(
                        planning_request=planning_request,
                        mobility_context=mobility_context,
                    ),
                    policy_type="PcfAmPolicyControlPolicyAssociation",
                    policy_details=normalized_am_policy,
                )
            )

        for index, policy_details in enumerate(decision.ursp_policies, start=1):
            flow_id = self._resolve_pcflow_id(
                policy_details,
                decision=decision,
                planning_request=planning_request,
                allow_missing=True,
            )
            app_id = self._resolve_decision_app_id(
                decision,
                flow_id=flow_id,
                policy_details=policy_details,
                planning_request=planning_request,
            )
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id=app_id,
                    flow_id=flow_id,
                    target_type="flow" if flow_id else "app",
                    policy_id=f"ursp-{app_id or 'app'}-{flow_id or index}",
                    policy_type="UrspRuleRequest",
                    policy_details=dict(policy_details),
                )
            )

        normalized = normalize_policy_plan_draft(plan)
        self.plan_compiler.plan_validator.validate_compiled_plan(normalized, planning_request)
        return normalized

    def _resolve_single_agent_flow_context(
        self,
        *,
        planning_request: PlanningRequest,
        decision: SingleAgentRoundDecision,
        flow_id: str,
        policy_details: Dict[str, Any],
    ) -> FlowSelector:
        try:
            return self.plan_compiler.artifact_compiler._resolve_flow(planning_request, flow_id)
        except Exception:
            pass
        return FlowSelector(
            supi=str(decision.supi or planning_request.operation_intent.supi or "").strip(),
            app_id=str(
                self._resolve_decision_app_id(
                    decision,
                    flow_id=flow_id,
                    policy_details=policy_details,
                    planning_request=planning_request,
                )
                or planning_request.operation_intent.app_id
                or ""
            ).strip(),
            flow_id=str(flow_id or "").strip(),
            name=str(flow_id or "").strip(),
            target_type="flow",
            resolution_status="resolved",
        )

    @classmethod
    def _resolve_decision_app_id(
        cls,
        decision: SingleAgentRoundDecision,
        *,
        flow_id: str | None = None,
        policy_details: Optional[Dict[str, Any]] = None,
        planning_request: Optional[PlanningRequest] = None,
    ) -> str:
        target_flow_id = str(flow_id or "").strip()
        request_flows = planning_request.operation_intent.flows if planning_request is not None else []
        for flow in request_flows or []:
            if target_flow_id and str(flow.flow_id or "").strip() != target_flow_id:
                continue
            app_id = str(flow.app_id or "").strip()
            if app_id:
                return app_id
        if policy_details:
            app_id = str(cls._extract_sm_policy_app_id(policy_details) or "").strip()
            if app_id:
                return app_id
        for policy in decision.sm_policies or []:
            app_id = str(cls._extract_sm_policy_app_id(policy) or "").strip()
            if app_id:
                return app_id
        if planning_request is not None:
            return str(planning_request.operation_intent.app_id or "").strip()
        return ""

    @classmethod
    def _normalize_am_policy_payload(
        cls,
        policy_details: Dict[str, Any],
        *,
        planning_request: PlanningRequest,
        mobility_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = copy.deepcopy(policy_details)
        if isinstance(payload.get("request"), dict):
            request = copy.deepcopy(payload["request"])
            request["supi"] = str(planning_request.operation_intent.supi or request.get("supi") or "").strip()
            request.setdefault("suppFeat", str(payload.get("suppFeat") or "1"))
            normalized = {
                key: copy.deepcopy(value)
                for key, value in payload.items()
                if key in cls._am_policy_allowed_fields()
            }
            normalized["request"] = request
            normalized.setdefault("suppFeat", str(payload.get("suppFeat") or request.get("suppFeat") or "1"))
            return normalized

        association_id = str(payload.get("associationId") or payload.get("association_id") or "").strip()
        mobility_summary = mobility_context.get("mobilitySummary") if isinstance(mobility_context, dict) else {}
        if not association_id:
            association_id = str((mobility_summary or {}).get("currentAssociationId") or "").strip()
        am_policy_context = mobility_context.get("amPolicyContext") if isinstance(mobility_context, dict) else {}
        associations = am_policy_context.get("associations") if isinstance(am_policy_context, dict) else {}
        association_payload = associations.get(association_id) if isinstance(associations, dict) else {}
        base_policy = copy.deepcopy(association_payload) if isinstance(association_payload, dict) else {}
        base_request = base_policy.get("request") if isinstance(base_policy.get("request"), dict) else {}
        request = copy.deepcopy(base_request)
        request["supi"] = str(planning_request.operation_intent.supi or payload.get("supi") or request.get("supi") or "").strip()
        request["notificationUri"] = str(
            payload.get("notificationUri")
            or payload.get("notification_uri")
            or request.get("notificationUri")
            or f"http://localhost:8000/notify/{request['supi']}"
        ).strip()
        for key in (
            "accessType",
            "accessTypes",
            "ratType",
            "ratTypes",
            "servingPlmn",
            "allowedSnssais",
            "targetSnssais",
            "mappingSnssais",
            "rfsp",
            "ueAmbr",
            "servAreaRes",
            "guami",
            "userLoc",
        ):
            if payload.get(key) is not None:
                request[key] = copy.deepcopy(payload[key])
        request["suppFeat"] = str(payload.get("suppFeat") or request.get("suppFeat") or base_policy.get("suppFeat") or "1")

        normalized = {
            key: copy.deepcopy(value)
            for key, value in base_policy.items()
            if key in cls._am_policy_allowed_fields() and key != "request"
        }
        normalized["request"] = request
        if payload.get("triggers") is not None:
            normalized["triggers"] = list(payload.get("triggers") or [])
        elif normalized.get("triggers") is None:
            normalized["triggers"] = []
        normalized["rfsp"] = int(payload.get("rfsp") or request.get("rfsp") or normalized.get("rfsp") or 1)
        normalized["suppFeat"] = str(payload.get("suppFeat") or normalized.get("suppFeat") or request.get("suppFeat") or "1")
        return normalized

    @staticmethod
    def _am_policy_allowed_fields() -> set[str]:
        return {
            "request",
            "triggers",
            "servAreaRes",
            "wlServAreaRes",
            "rfsp",
            "targetRfsp",
            "smfSelInfo",
            "ueAmbr",
            "ueSliceMbrs",
            "pras",
            "suppFeat",
            "pcfUeInfo",
            "matchPdus",
            "asTimeDisParam",
        }

    @staticmethod
    def _optimizer_flow_ids(optimizer_preview: Any) -> List[str]:
        if not isinstance(optimizer_preview, dict):
            return []
        flow_ids: List[str] = []
        assignments = optimizer_preview.get("qos_flow_assignments")
        if isinstance(assignments, list):
            for item in assignments:
                if not isinstance(item, dict):
                    continue
                flow_id = str(item.get("flow_id") or item.get("id") or "").strip()
                if flow_id and flow_id not in flow_ids:
                    flow_ids.append(flow_id)
        qos_plan = optimizer_preview.get("qos_plan")
        flow_sets: List[Any] = []
        if isinstance(qos_plan, dict):
            target_apps = qos_plan.get("target_apps")
            if isinstance(target_apps, list):
                for app in target_apps:
                    if isinstance(app, dict):
                        flow_sets.append(app.get("flows"))
            target_app = qos_plan.get("target_app")
            if isinstance(target_app, dict):
                flow_sets.append(target_app.get("flows"))
        for flows in flow_sets:
            if not isinstance(flows, list):
                continue
            for item in flows:
                if not isinstance(item, dict):
                    continue
                flow_id = str(item.get("id") or item.get("flow_id") or "").strip()
                if flow_id and flow_id not in flow_ids:
                    flow_ids.append(flow_id)
        return flow_ids

    @classmethod
    def _resolve_pcflow_id(
        cls,
        policy_details: Dict[str, Any],
        *,
        decision: SingleAgentRoundDecision,
        planning_request: Optional[PlanningRequest] = None,
        optimizer_preview: Any = None,
        allow_missing: bool = False,
    ) -> str:
        inferred_flow_id = cls._extract_sm_policy_flow_id(policy_details)
        request_flows = planning_request.operation_intent.flows if planning_request is not None else []
        grounded_flow_ids = [
            str(flow.flow_id or "").strip()
            for flow in request_flows or []
            if str(flow.flow_id or "").strip()
        ]
        optimizer_flow_ids = cls._optimizer_flow_ids(optimizer_preview)
        authoritative_flow_ids = list(dict.fromkeys(optimizer_flow_ids + grounded_flow_ids))

        if inferred_flow_id and inferred_flow_id in authoritative_flow_ids:
            return inferred_flow_id
        if len(authoritative_flow_ids) == 1:
            return authoritative_flow_ids[0]
        if inferred_flow_id:
            return inferred_flow_id
        if not allow_missing:
            raise ValueError("PCF-style policy payload does not contain a grounded flow_id")
        return ""

    @staticmethod
    def _required_evidence_from_domains(domains: List[str]) -> List[str]:
        required: List[str] = []
        normalized = {str(item or "").strip().lower() for item in domains if str(item or "").strip()}
        if "qos" in normalized:
            required.append("qos_runtime_evidence")
        if "mobility" in normalized:
            required.append("mobility_policy_context")
        return required

    def _build_global_intent(
        self,
        *,
        decision: SingleAgentRoundDecision,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        user_input: str,
    ) -> GlobalControlIntent:
        requested_domains = [ControlDomain(item) for item in decision.requested_domains]
        return GlobalControlIntent(
            session_id=session_id,
            snapshot_id=snapshot_id,
            raw_input=user_input,
            supi=str(operation_intent.supi or "").strip(),
            round_strategy=MainRoundStrategy.INITIAL_GROUNDING,
            next_agent="optimization_strategy",
            requested_domains=requested_domains,
            domain_evidence=operation_intent.domain_evidence,
            control_semantics=operation_intent.control_semantics,
            objective_profile=ObjectiveProfile(
                profile_name=str(decision.objective_profile_hint or "").strip()
            ),
            investigation_targets=[],
            uncertainty_flags=[],
            required_evidence=self._required_evidence_from_domains(decision.requested_domains),
            forbidden_assumptions=[],
        )

    @staticmethod
    def _build_planning_request(
        *,
        global_intent: GlobalControlIntent,
        operation_intent: OperationIntent,
        session_id: str,
        snapshot_id: str,
        round_index: int,
        feedback_context: str,
        round_traces: List[Dict[str, Any]],
        previous_mediator_decision: Dict[str, Any],
    ) -> PlanningRequest:
        return PlanningRequest(
            operation_intent=operation_intent,
            context=PlanningContext(
                round_index=round_index,
                session_id=session_id,
                snapshot_id=snapshot_id,
                snapshot_metadata=get_latest_snapshot_metadata() or {},
                feedback_context=feedback_context,
                handoff_history=round_traces[-2:],
                active_domains=[item.value for item in global_intent.requested_domains],
                main_round_strategy=global_intent.round_strategy.value,
                objective_profile=global_intent.objective_profile.model_dump(mode="json"),
                forbidden_assumptions=list(global_intent.forbidden_assumptions or []),
                required_evidence=list(global_intent.required_evidence or []),
                revision_requests=list((previous_mediator_decision or {}).get("revision_requests") or []),
                unified_constraints=dict((previous_mediator_decision or {}).get("unified_constraints") or {}),
            ),
        )


__all__ = ["SingleControlAgent"]
