from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

from knowledge_runtime.retrieval.raw import warmup_knowledge_tool_models
from shared.memory import MemoryManager
from agent_runtime.core.token_budget import TokenCounter, TokenBudget
from agent_runtime.core.context_policy import ContextPolicy
from ..agents.common import project_global_intent_for_prompt, project_memory_payload

from ..agents.dispatch import PolicyDispatchAgent
from ..agents.grounding import IntentEncodingAgent
from ..agents.main import MainControlAgent
from ..agents.planning import OptimizationStrategyAgent
from ..diagnostics.diagnosis import AssuranceDiagnosisTool
from ..diagnostics.mediation import ConflictResolutionTool
from ..domain.collaboration import (
    DomainNegotiationRequest,
    ExecutionReentryRequest,
    PlanningBlockerReport,
    PlanningRequest,
)
from ..domain.control_plane import GlobalControlIntent
from ..domain.policy_plan import OperationIntent
from ..integrations.storage import get_snapshot_data_by_id
from .main_control_support import (
    ControlRoundResult,
    build_feedback_context_from_snapshots,
    build_round_feedback_block,
    _build_snapshot_summary,
    build_main_context,
    build_planning_context,
)
from .loop_state import OrchestratorLoopState, append_round_trace, finish_control_session, start_control_session
from .round_execution import execute_planned_round
from shared.logging import log_event


class MainControlOrchestrator:
    _DEEPSEEK_MODEL = "deepseek-v4-flash"

    def __init__(
        self,
        *,
        main_agent: Optional[MainControlAgent] = None,
        ie_agent: Optional[IntentEncodingAgent] = None,
        os_agent: Optional[OptimizationStrategyAgent] = None,
        pd_agent: Optional[PolicyDispatchAgent] = None,
        cr_tool: Optional[ConflictResolutionTool] = None,
        ad_tool: Optional[AssuranceDiagnosisTool] = None,
        memory_manager: Optional[MemoryManager] = None,
        max_rounds: int = 3,
        use_local_model: bool = False,
        use_deepseek: bool = False,
        preload_models: bool = True,
        rag_enabled: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        _deepseek_kwargs = {"model_name": self._DEEPSEEK_MODEL} if use_deepseek else {}
        self.main_agent = main_agent or MainControlAgent(use_local_model=use_local_model, **_deepseek_kwargs)
        self.ie_agent = ie_agent or IntentEncodingAgent(use_local_model=use_local_model, rag_enabled=rag_enabled, **_deepseek_kwargs)
        self.os_agent = os_agent or OptimizationStrategyAgent(use_local_model=use_local_model, rag_enabled=rag_enabled, **_deepseek_kwargs)
        self.pd_agent = pd_agent or PolicyDispatchAgent(use_local_model=use_local_model, **_deepseek_kwargs)
        self.cr_tool = cr_tool or ConflictResolutionTool()
        self.ad_tool = ad_tool or AssuranceDiagnosisTool()
        self.memory_manager = memory_manager or MemoryManager(
            short_term_limit=max(20, max_rounds * 8),
            enable_llm_summarization=True,
        )
        self.max_rounds = max_rounds
        self.rag_enabled = rag_enabled
        self._token_counter = TokenCounter()
        self._context_policy = ContextPolicy()
        self.preloaded_models: Dict[str, Any] = {}
        if preload_models:
            self.preloaded_models = self._preload_runtime_models()

    def _preload_runtime_models(self) -> Dict[str, Any]:
        try:
            llm_models: List[str] = []
            preloaded_llm_agents: List[str] = []
            for agent in (self.main_agent, self.ie_agent, self.os_agent, self.pd_agent):
                model_name = str(getattr(agent, "model_name", "") or "").strip()
                if model_name:
                    llm_models.append(model_name)

                get_llm = getattr(agent, "get_llm", None)
                if callable(get_llm):
                    get_llm()
                    preloaded_llm_agents.append(str(getattr(agent, "agent_name", agent.__class__.__name__) or agent.__class__.__name__))

            knowledge_models = warmup_knowledge_tool_models() if self.rag_enabled else {}
            log_event(
                self.main_agent.logger,
                "runtime_model_preload_complete",
                llm_models=",".join(model for model in llm_models if model) or "<none>",
                llm_agents=",".join(preloaded_llm_agents) or "<none>",
                rerankers=",".join(knowledge_models.get("rerankers") or []) or "<none>",
                vectorstores=",".join(knowledge_models.get("vectorstores") or []) or "<none>",
            )
            return {
                "llm_models": llm_models,
                "knowledge_tool": knowledge_models,
            }
        except Exception as exc:
            raise RuntimeError(f"failed to preload runtime models: {exc}") from exc

    @staticmethod
    def _trace_metadata(*, scenario_id: str = "", scenario_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "scenario_id": str(scenario_id or "").strip(),
            "scenario_tags": [str(item).strip() for item in (scenario_tags or []) if str(item).strip()],
        }

    def _inject_token_context(self) -> None:
        for agent in (self.main_agent, self.ie_agent, self.os_agent, self.pd_agent):
            agent._token_budget = self._token_budget
            agent._token_counter = self._token_counter

    def _remember(self, role: str, payload: Any) -> None:
        if isinstance(payload, str):
            content = payload
        else:
            content = json.dumps(project_memory_payload(role, payload), ensure_ascii=False)
        if role == "IEA":
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                supi = str(parsed.get("supi") or "").strip()
                if supi:
                    self.memory_manager.bind_supi(supi)
        self.memory_manager.add_memory(role, content)

    def _build_memory_context(self, user_input: str, *, diagnosis_hint: str = "", routing_hint: str = "") -> str:
        bundle = self.memory_manager.retrieve(user_input)
        short_term = bundle.get("short_term", []) if isinstance(bundle, dict) else []
        long_term = bundle.get("long_term", []) if isinstance(bundle, dict) else []
        short_term = self._rerank_by_hint(short_term, diagnosis_hint, routing_hint)[:5]
        long_term = self._rerank_by_hint(long_term, diagnosis_hint, routing_hint)[:5]
        blocks: List[str] = []
        if short_term:
            blocks.append("[Memory][Short-Term]\n" + "\n".join(f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in short_term if isinstance(item, dict)))
        if long_term:
            blocks.append("[Memory][Long-Term]\n" + "\n".join(str(item) for item in long_term))
        rendered = "\n\n".join(block for block in blocks if block.strip())
        policy = getattr(self, "_context_policy", None) or ContextPolicy()
        counter = getattr(self, "_token_counter", None)
        return policy.compact_text(
            rendered,
            max_chars=6000,
            max_tokens=1500,
            token_counter=counter,
        )

    @staticmethod
    def _rerank_by_hint(items: List[Any], diagnosis_hint: str = "", routing_hint: str = "") -> List[Any]:
        hint_text = f"{diagnosis_hint} {routing_hint}".lower()
        hint_tokens = {
            token
            for token in re.split(r"[^a-zA-Z0-9_]+", hint_text)
            if len(token) >= 3
        }
        if not hint_tokens:
            return list(items)

        def score_item(index_item: tuple[int, Any]) -> tuple[int, int]:
            index, item = index_item
            if isinstance(item, dict):
                text = json.dumps(item, ensure_ascii=False).lower()
            else:
                text = str(item or "").lower()
            item_tokens = {
                token
                for token in re.split(r"[^a-zA-Z0-9_]+", text)
                if len(token) >= 3
            }
            return (len(hint_tokens & item_tokens), -index)

        ranked = sorted(enumerate(items), key=score_item, reverse=True)
        return [item for _, item in ranked]

    def _build_ie_context(
        self,
        *,
        global_intent: Dict[str, Any],
        snapshot_id: str,
        round_index: int,
        diagnosis: Dict[str, Any],
        feedback_context: str,
    ) -> str:
        snapshot = get_snapshot_data_by_id(snapshot_id) or {}
        projected_global_intent = project_global_intent_for_prompt(global_intent)
        return (
            "## Guidance\n"
            f"- round_index: {round_index}\n"
            f"- intent_encoding_guidance: {global_intent.get('intent_encoding_guidance', '') or 'N/A'}\n"
            f"- retry_scope: {global_intent.get('retry_scope') or 'N/A'}\n"
            f"- routing_decision: {global_intent.get('routing_decision') or 'N/A'}\n"
            f"- routing_rationale: {global_intent.get('routing_rationale') or 'N/A'}\n\n"
            "## Evidence\n"
            f"- main_intent: {json.dumps(projected_global_intent, ensure_ascii=False)}\n"
            f"- snapshot_summary: {json.dumps(_build_snapshot_summary(snapshot) if snapshot else {}, ensure_ascii=False)}\n"
            "\n"
            "## Previous Diagnosis\n"
            f"{json.dumps(diagnosis or {}, ensure_ascii=False)}\n\n"
            "## Feedback\n"
            f"{feedback_context or 'N/A'}"
        )

    @staticmethod
    def _scope_global_intent_for_ie(
        *,
        global_intent: GlobalControlIntent,
        round_index: int,
    ) -> GlobalControlIntent:
        semantics = global_intent.control_semantics
        stages = list(semantics.stages or [])
        if not stages:
            return global_intent

        ordered_stages = sorted(stages, key=lambda stage: int(stage.stage_index or 0))
        scoped_position = min(max(1, int(round_index or 1)), len(ordered_stages)) - 1
        scoped_stage = ordered_stages[scoped_position].model_copy(deep=True)
        scoped_semantics = semantics.model_copy(
            update={
                "current_stage": 1,
                "stages": [
                    scoped_stage.model_copy(
                        update={"stage_index": 1},
                        deep=True,
                    )
                ],
            },
            deep=True,
        )
        return global_intent.model_copy(
            update={"control_semantics": scoped_semantics},
            deep=True,
        )

    @staticmethod
    def _build_planning_failure_payload(
        exc: Exception,
        *,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        message = str(exc or "").strip() or exc.__class__.__name__
        report_payload = {
            "execution_status": "Failed",
            "violation_details": message,
            "feedback_payload": {
                "phase": "planning",
                "error": message,
                "target_bindings_at_risk": [],
                "policy_objects_at_risk": [],
                "reason_by_domain": {},
                "debug_context": dict(debug_context or {}),
            },
        }
        diagnosis = {
            "root_cause_category": "planning_failure",
            "root_cause": message,
            "reason_summary": message,
            "recommended_actions": [],
            "debug_context": dict(debug_context or {}),
        }
        return report_payload, diagnosis

    @staticmethod
    def _build_negotiation_request(operation_intent: OperationIntent, *, round_index: int) -> DomainNegotiationRequest:
        issues = []
        for question in operation_intent.open_questions:
            payload = question.model_dump(mode="json") if hasattr(question, "model_dump") else dict(question)
            issues.append(
                {
                    "source_agent": "intent_encoding",
                    "issue_type": "domain_boundary",
                    "domain": str((payload.get("related_domains") or [""])[0] or ""),
                    "binding_keys": [],
                    "policy_objects": [],
                    "missing_evidence": [str(payload.get("question") or "").strip()] if str(payload.get("question") or "").strip() else [],
                    "rationale": str(payload.get("question") or "").strip(),
                }
            )
        return DomainNegotiationRequest(
            round_index=round_index,
            source_agent="intent_encoding",
            main_requested_domains=list(operation_intent.main_requested_domains or []),
            grounded_requested_domains=list(operation_intent.grounded_requested_domains or operation_intent.requested_domains or []),
            domain_resolution=str(operation_intent.domain_resolution or "cannot_confirm"),
            domain_revision_needed=bool(operation_intent.domain_revision_needed),
            issues=issues,
            recommended_consumers=["main_control", "intent_encoding"],
            summary=str(operation_intent.domain_revision_rationale or "").strip(),
        )

    @staticmethod
    def _build_negotiation_diagnosis(request: DomainNegotiationRequest) -> Dict[str, Any]:
        issue_reasons = [
            str(item.rationale or "").strip()
            for item in (request.issues or [])
            if str(item.rationale or "").strip()
        ]
        return {
            "root_cause_category": "domain_negotiation_required",
            "root_cause": request.summary,
            "reason_summary": request.summary,
            "recommended_actions": issue_reasons,
        }

    @staticmethod
    def _build_reentry_report_payload(request: ExecutionReentryRequest) -> Dict[str, Any]:
        return {
            "execution_status": "Failed",
            "violation_details": request.summary,
            "feedback_payload": {
                "failure_scope": request.failure_scope,
                "target_bindings_at_risk": list(request.target_bindings_at_risk or []),
                "policy_objects_at_risk": list(request.policy_objects_at_risk or []),
                "reason_by_domain": dict(request.reason_by_domain or {}),
                "failures": list(request.failures or []),
            },
        }

    @staticmethod
    def _should_reuse_operation_intent(
        *,
        global_intent: GlobalControlIntent,
        previous_operation_intent: Optional[OperationIntent],
        previous_report_payload: Dict[str, Any],
        previous_mediator_decision: Optional[Dict[str, Any]],
    ) -> bool:
        if previous_operation_intent is None:
            return False
        if str(global_intent.next_agent or "").strip().lower() != "optimization_strategy":
            return False
        contract = global_intent.reuse_contract
        if not contract.allowed:
            return False
        if contract.preserve_bindings and str(previous_operation_intent.supi or "").strip() != str(global_intent.supi or "").strip():
            return False
        previous_domains = {str(item or "").strip().lower() for item in (previous_operation_intent.requested_domains or []) if str(item or "").strip()}
        current_domains = {item.value for item in global_intent.requested_domains}
        if contract.preserve_domains and previous_domains != current_domains:
            return False
        if contract.preserve_stage_scope:
            stages = previous_operation_intent.control_semantics.stages or []
            active_stage = next(
                (
                    stage
                    for stage in stages
                    if int(stage.stage_index or 0) == int(previous_operation_intent.control_semantics.current_stage or 1)
                ),
                None,
            )
            if active_stage is None:
                return False
            if not [flow_id for flow_id in (active_stage.active_flow_ids or []) if str(flow_id or "").strip()]:
                return False
        invalidate_text = json.dumps(
            {
                "report": previous_report_payload or {},
                "mediator": previous_mediator_decision or {},
                "open_questions": previous_operation_intent.open_questions,
            },
            ensure_ascii=False,
        ).lower()
        for token in contract.invalidate_on:
            normalized = str(token or "").strip().lower()
            if normalized and normalized in invalidate_text:
                return False
        if previous_operation_intent.open_questions:
            return False
        return True

    def _plan_round(
        self,
        *,
        user_input: str,
        session_id: str,
        snapshot_id: str,
        round_index: int,
        scenario_id: str,
        scenario_tags: Optional[List[str]],
        memory_context: str,
        feedback_context: str,
        previous_diagnosis: Dict[str, Any],
        previous_report_payload: Dict[str, Any],
        previous_mediator_decision: Optional[Dict[str, Any]],
        previous_operation_intent: Optional[OperationIntent],
        previous_negotiation_request: Dict[str, Any],
        previous_planning_blocker: Dict[str, Any],
        previous_execution_reentry: Dict[str, Any],
        round_traces: List[Dict[str, Any]],
    ) -> tuple[GlobalControlIntent, Optional[OperationIntent], Optional[Any], Optional[DomainNegotiationRequest]]:
        trace_metadata = self._trace_metadata(scenario_id=scenario_id, scenario_tags=scenario_tags)
        global_intent = self.main_agent.analyze_global_intent(
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
            context=build_main_context(
                snapshot_id,
                round_index=round_index,
                memory_context=memory_context,
                feedback_context=feedback_context,
                previous_diagnosis=previous_diagnosis,
                previous_execution_feedback=previous_report_payload,
                previous_operation_intent=(
                    previous_operation_intent.model_dump(mode="json")
                    if previous_operation_intent is not None
                    else {}
                ),
                previous_negotiation_request=previous_negotiation_request,
                previous_planning_blocker=previous_planning_blocker,
                previous_execution_reentry=previous_execution_reentry,
            ),
            trace_metadata=trace_metadata,
        )
        if not global_intent.requested_domains:
            raise RuntimeError("Main Agent returned no requested_domains; refusing to infer domains outside the agent.")
        if not str(global_intent.supi or "").strip():
            raise RuntimeError("Main Agent returned no SUPI; refusing to patch identifiers outside the agent.")
        self._remember("MAIN", global_intent)

        selected_next_agent = str(global_intent.next_agent or "").strip().lower()
        reuse_operation_intent = (
            round_index > 1
            and self._should_reuse_operation_intent(
                global_intent=global_intent,
                previous_operation_intent=previous_operation_intent,
                previous_report_payload=previous_report_payload,
                previous_mediator_decision=previous_mediator_decision,
            )
        )

        if reuse_operation_intent:
            operation_intent = previous_operation_intent.model_copy(deep=True)
            log_event(
                self.main_agent.logger,
                "control_round_resume",
                session_id=session_id,
                round_index=round_index,
                entrypoint="optimization_strategy",
                selected_next_agent=selected_next_agent,
            )
        else:
            ie_scoped_intent = self._scope_global_intent_for_ie(
                global_intent=global_intent,
                round_index=round_index,
            )
            operation_intent = self.ie_agent.analyze_operation_intent(
                user_input=user_input,
                context=self._build_ie_context(
                    global_intent=ie_scoped_intent.model_dump(mode="json"),
                    snapshot_id=snapshot_id,
                    round_index=round_index,
                    diagnosis=previous_diagnosis,
                    feedback_context=feedback_context,
                ),
                session_id=session_id,
                snapshot_id=snapshot_id,
                trace_metadata=trace_metadata,
            )
            self._remember("IEA", operation_intent)
            if str(operation_intent.domain_resolution or "").strip().lower() == "cannot_confirm":
                return global_intent, operation_intent, None, self._build_negotiation_request(
                    operation_intent,
                    round_index=round_index,
                )

        operation_intent = self._activate_control_stage(
            operation_intent=operation_intent,
            round_index=round_index,
        )

        planning_request = PlanningRequest(
            operation_intent=operation_intent,
            context=build_planning_context(
                global_intent,
                session_id,
                snapshot_id,
                active_domains=list(operation_intent.requested_domains or []),
                round_index=round_index,
                memory_context=memory_context,
                feedback_context=feedback_context,
                handoff_history=round_traces,
                revision_requests=(previous_mediator_decision or {}).get("revision_requests") if isinstance(previous_mediator_decision, dict) else None,
                unified_constraints=(previous_mediator_decision or {}).get("unified_constraints") if isinstance(previous_mediator_decision, dict) else None,
            ),
        )
        policy_plan = self.os_agent.generate_strategy(planning_request, trace_metadata=trace_metadata)
        self._remember("OSA", policy_plan)
        return global_intent, operation_intent, policy_plan, None

    @staticmethod
    def _activate_control_stage(
        *,
        operation_intent: OperationIntent,
        round_index: int,
    ) -> OperationIntent:
        semantics = operation_intent.control_semantics
        if not semantics.stages:
            return operation_intent
        activated = operation_intent.model_copy(deep=True)
        max_stage = max(stage.stage_index for stage in semantics.stages)
        activated.control_semantics.current_stage = min(max(1, round_index), max_stage)
        return activated

    def run(
        self,
        user_input: str,
        *,
        scenario_id: str = "",
        scenario_tags: Optional[List[str]] = None,
        snapshot_id: str = "",
    ) -> ControlRoundResult:
        session_id, snapshot_id = start_control_session(step_name="main_control", user_input=user_input, snapshot_id=snapshot_id)
        self.memory_manager.bind_thread(session_id)
        self._token_budget = TokenBudget()
        self._inject_token_context()
        state = OrchestratorLoopState()
        previous_operation_intent: Optional[OperationIntent] = None

        for round_index in range(1, self.max_rounds + 1):
            log_event(
                self.main_agent.logger,
                "control_round_start",
                session_id=session_id,
                round_index=round_index,
                retry_count=max(0, round_index - 1),
                previous_root_cause=str(state.previous_diagnosis.get("root_cause_category") or "").strip() or "<none>",
            )
            memory_context = self._build_memory_context(
                user_input,
                diagnosis_hint=str(state.previous_diagnosis.get("root_cause_category") or ""),
                routing_hint=str((state.previous_mediator_decision or {}).get("status") or ""),
            )
            feedback_context = build_feedback_context_from_snapshots(
                state.rounds,
                token_counter=self._token_counter,
                summarizer_llm=getattr(self.memory_manager, "summarizer_llm", None),
            )
            trace_metadata = self._trace_metadata(scenario_id=scenario_id, scenario_tags=scenario_tags)
            try:
                global_intent, operation_intent, policy_plan, negotiation_request = self._plan_round(
                    user_input=user_input,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    round_index=round_index,
                    scenario_id=scenario_id,
                    scenario_tags=scenario_tags,
                    memory_context=memory_context,
                    feedback_context=feedback_context,
                    previous_diagnosis=state.previous_diagnosis,
                    previous_report_payload=state.previous_report_payload,
                    previous_mediator_decision=state.previous_mediator_decision,
                    previous_operation_intent=previous_operation_intent,
                    previous_negotiation_request=state.previous_negotiation_request,
                    previous_planning_blocker=state.previous_planning_blocker,
                    previous_execution_reentry=state.previous_execution_reentry,
                    round_traces=state.round_traces,
                )
            except Exception as exc:
                state.completed = False
                debug_context = {
                    "intent_encoding": getattr(self.ie_agent, "last_failure_debug", {}) or {},
                    "optimization_strategy": getattr(self.os_agent, "last_failure_debug", {}) or {},
                }
                debug_context = {key: value for key, value in debug_context.items() if value}
                report_payload, diagnosis = self._build_planning_failure_payload(
                    exc,
                    debug_context=debug_context,
                )
                trace_payload = {
                    "round_index": round_index,
                    "global_intent": {},
                    "operation_intent": {},
                    "policy_plan": {},
                    "domain_verdicts": [],
                    "pda_feedback": report_payload,
                    "qos_feedback": {},
                    "mobility_feedback": {},
                    "diagnosis": diagnosis,
                    "negotiation_request": {},
                    "planning_blocker": {},
                    "execution_reentry": {},
                }
                feedback_added = build_round_feedback_block(
                    pda_feedback=report_payload,
                    diagnosis=diagnosis,
                    domain_verdicts=[],
                    mediator_decision={},
                    negotiation_request={},
                    planning_blocker={},
                    execution_reentry={},
                    round_index=round_index,
                )
                append_round_trace(state, trace_payload=trace_payload, feedback_added=feedback_added)
                state.latest_result = ControlRoundResult(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    completed=False,
                    global_intent={},
                    unified_plan={},
                    qos_feedback={},
                    mobility_feedback={},
                    diagnosis=diagnosis,
                    negotiation_request={},
                    planning_blocker={},
                    execution_reentry={},
                    round_count=round_index,
                    retry_count=max(0, round_index - 1),
                    round_traces=state.round_traces,
                )
                log_event(
                    self.main_agent.logger,
                    "control_round_complete",
                    session_id=session_id,
                    round_index=round_index,
                    completed=False,
                    requested_domains="<planning_failed>",
                    diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                    execution_status="Failed",
                )
                if round_index >= self.max_rounds:
                    break
                log_event(
                    self.main_agent.logger,
                    "control_round_retry_scheduled",
                    session_id=session_id,
                    next_round=round_index + 1,
                    diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                    recommended_consumers="<none>",
                )
                continue
            if negotiation_request is not None:
                diagnosis = self._build_negotiation_diagnosis(negotiation_request)
                negotiation_payload = negotiation_request.model_dump(mode="json")
                trace_payload = {
                    "round_index": round_index,
                    "global_intent": global_intent.model_dump(mode="json"),
                    "operation_intent": operation_intent.model_dump(mode="json") if operation_intent is not None else {},
                    "policy_plan": {},
                    "domain_verdicts": [],
                    "pda_feedback": {},
                    "qos_feedback": {},
                    "mobility_feedback": {},
                    "diagnosis": diagnosis,
                    "negotiation_request": negotiation_payload,
                    "planning_blocker": {},
                    "execution_reentry": {},
                }
                feedback_added = build_round_feedback_block(
                    diagnosis=diagnosis,
                    negotiation_request=negotiation_payload,
                    round_index=round_index,
                )
                append_round_trace(state, trace_payload=trace_payload, feedback_added=feedback_added)
                state.latest_result = ControlRoundResult(
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    completed=False,
                    global_intent=global_intent.model_dump(mode="json"),
                    unified_plan={},
                    qos_feedback={},
                    mobility_feedback={},
                    diagnosis=diagnosis,
                    negotiation_request=negotiation_payload,
                    planning_blocker={},
                    execution_reentry={},
                    round_count=round_index,
                    retry_count=max(0, round_index - 1),
                    round_traces=state.round_traces,
                )
                continue

            if operation_intent is None or policy_plan is None:
                raise RuntimeError("main control planning round produced no executable planning artifacts")
            previous_operation_intent = operation_intent.model_copy(deep=True)

            round_execution = execute_planned_round(
                session_id=session_id,
                snapshot_id=snapshot_id,
                round_index=round_index,
                global_intent=global_intent,
                operation_intent=operation_intent,
                policy_plan=policy_plan,
                cr_tool=self.cr_tool,
                pd_agent=self.pd_agent,
                ad_tool=self.ad_tool,
                trace_metadata=trace_metadata,
            )
            state.completed = round_execution.completed
            report = round_execution.report
            qos_feedback = round_execution.qos_feedback
            mobility_feedback = round_execution.mobility_feedback
            diagnosis = round_execution.diagnosis
            self._remember("AD", diagnosis)
            state.latest_result = ControlRoundResult(
                session_id=session_id,
                snapshot_id=snapshot_id,
                completed=state.completed,
                global_intent=global_intent.model_dump(mode="json"),
                unified_plan=round_execution.unified_plan.model_dump(mode="json"),
                qos_feedback=qos_feedback,
                mobility_feedback=mobility_feedback,
                diagnosis=diagnosis,
                negotiation_request={},
                planning_blocker=(
                    round_execution.planning_blocker.model_dump(mode="json")
                    if round_execution.planning_blocker is not None
                    else {}
                ),
                execution_reentry=(
                    round_execution.execution_reentry.model_dump(mode="json")
                    if round_execution.execution_reentry is not None
                    else {}
                ),
                round_count=round_index,
                retry_count=max(0, round_index - 1),
                round_traces=state.round_traces,
            )
            log_event(
                self.main_agent.logger,
                "control_round_complete",
                session_id=session_id,
                round_index=round_index,
                completed=state.completed,
                requested_domains=",".join(item.value for item in global_intent.requested_domains) or "<empty>",
                diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                execution_status=report.execution_status if report is not None else "blocked",
            )
            if state.completed:
                append_round_trace(
                    state,
                    trace_payload=json.loads(json.dumps(round_execution.trace, default=lambda obj: obj.__dict__, ensure_ascii=False)),
                )
                break

            if round_execution.execution_reentry is not None:
                report_payload = self._build_reentry_report_payload(round_execution.execution_reentry)
            elif round_execution.planning_blocker is not None:
                report_payload = {
                    "execution_status": "Failed",
                    "violation_details": round_execution.planning_blocker.summary,
                    "feedback_payload": {
                        "missing_evidence": list(round_execution.planning_blocker.missing_evidence or []),
                        "blocked_targets": list(round_execution.planning_blocker.blocked_targets or []),
                        "upstream_requests": list(round_execution.planning_blocker.upstream_requests or []),
                        "planner_conflicts": list(round_execution.planning_blocker.planner_conflicts or []),
                    },
                }
            elif report is not None:
                report_payload = report.model_dump(mode="json")
            else:
                report_payload = {
                    "execution_status": "Failed",
                    "violation_details": diagnosis.get("reason_summary") or "round execution failed",
                }
            previous_planning_blocker = (
                round_execution.planning_blocker.model_dump(mode="json")
                if round_execution.planning_blocker is not None
                else {}
            )
            previous_execution_reentry = (
                round_execution.execution_reentry.model_dump(mode="json")
                if round_execution.execution_reentry is not None
                else {}
            )
            trace_payload = json.loads(json.dumps(round_execution.trace, default=lambda obj: obj.__dict__, ensure_ascii=False))
            trace_payload["pda_feedback"] = dict(report_payload)
            trace_payload["mediator_decision"] = dict(round_execution.mediator_decision_payload)
            trace_payload["planning_blocker"] = previous_planning_blocker
            trace_payload["execution_reentry"] = previous_execution_reentry
            feedback_added = build_round_feedback_block(
                pda_feedback=report_payload,
                diagnosis=diagnosis,
                domain_verdicts=round_execution.domain_verdict_payloads,
                mediator_decision=round_execution.mediator_decision_payload,
                planning_blocker=previous_planning_blocker,
                execution_reentry=previous_execution_reentry,
                round_index=round_index,
            )
            append_round_trace(state, trace_payload=trace_payload, feedback_added=feedback_added)
            log_event(
                self.main_agent.logger,
                "control_round_retry_scheduled",
                session_id=session_id,
                next_round=round_index + 1,
                diagnosis_category=str(diagnosis.get("root_cause_category") or "").strip() or "<none>",
                recommended_consumers="<none>",
            )

        finish_control_session(session_id=session_id, snapshot_id=snapshot_id, state=state)
        if state.latest_result is None:
            raise RuntimeError("main control orchestrator produced no result")
        return state.latest_result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main control orchestrator end-to-end.",
    )
    parser.add_argument(
        "user_input",
        nargs="?",
        help="Natural-language control request. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--scenario-id",
        dest="scenario_id",
        default="",
        help="Optional trace scenario identifier.",
    )
    parser.add_argument(
        "--scenario-tag",
        dest="scenario_tags",
        action="append",
        default=[],
        help="Optional trace scenario tag. Repeat to provide multiple tags.",
    )
    parser.add_argument(
        "--snapshot-id",
        dest="snapshot_id",
        default="",
        help="Existing live network graph snapshot id to bind this run.",
    )
    parser.add_argument(
        "--max-rounds",
        dest="max_rounds",
        type=int,
        default=3,
        help="Maximum control-loop rounds to run.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result.",
    )
    parser.add_argument(
        "--deepseek",
        action="store_true",
        dest="use_deepseek",
        help="Use deepseek-v4-flash for all agents instead of the default models.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run the autonomous guard loop instead of a single request.",
    )
    parser.add_argument(
        "--watch-interval",
        dest="watch_interval",
        type=float,
        default=1.0,
        help="Seconds between autonomous guard-loop ticks.",
    )
    parser.add_argument(
        "--watch-iterations",
        dest="watch_iterations",
        type=int,
        default=0,
        help="Maximum guard-loop ticks. Use 0 for continuous guarding.",
    )
    parser.add_argument(
        "--monitor-context-chars",
        dest="monitor_context_chars",
        type=int,
        default=4000,
        help="Maximum previous-control context characters passed to monitor reentry.",
    )
    return parser.parse_args(argv)


def _resolve_user_input(cli_value: Optional[str]) -> str:
    text = str(cli_value or "").strip()
    if text:
        return text
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    raise ValueError("user_input is required either as a positional argument or via stdin")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        orchestrator = MainControlOrchestrator(max_rounds=args.max_rounds, use_local_model=True, use_deepseek=args.use_deepseek)
        if args.watch:
            from control_runtime.monitoring import ConsoleUserInputSource, build_default_autonomous_watch_loop

            console_source = ConsoleUserInputSource(prompt="control> ")
            initial_user_input = str(args.user_input or "").strip()

            def user_input_source() -> str:
                nonlocal initial_user_input
                if initial_user_input:
                    text = initial_user_input
                    initial_user_input = ""
                    return text
                return console_source()

            watch_loop = build_default_autonomous_watch_loop(
                orchestrator=orchestrator,
                user_input_source=user_input_source,
                previous_context_max_chars=args.monitor_context_chars,
                poll_interval_seconds=args.watch_interval,
            )
            results = watch_loop.run_forever(
                max_iterations=args.watch_iterations if args.watch_iterations > 0 else None,
                snapshot_id=args.snapshot_id,
                scenario_id=args.scenario_id,
                scenario_tags=args.scenario_tags,
            )
            payload = [result.to_dict() for result in results]
            print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
            return 0

        user_input = _resolve_user_input(args.user_input)
        result = orchestrator.run(
            user_input,
            scenario_id=args.scenario_id,
            scenario_tags=args.scenario_tags,
            snapshot_id=args.snapshot_id,
        )
        payload = result.__dict__
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0
    except Exception as exc:
        error_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2 if args.pretty else None), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
