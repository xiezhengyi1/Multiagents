from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logger import setup_logger, log_event, log_timing
from .basemodel import BaseAgent
from .Prompt import (
    PDA_SYSTEM_PROMPT,
    PDA_USER_FEEDBACK_PROMPT,
    PDA_EXECUTION_TOOL_SYSTEM_PROMPT,
    PDA_COMMIT_TOOL_SYSTEM_PROMPT,
)
from tools.pcf_tools import dispatch_policy_to_pcf, get_network_feedback
from tools.db_tool import upsert_ue_context, get_latest_snapshot_data, get_ue_flow_catalog_by_supi


def _json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

# --- Tool Wrappers ---
@tool
def tool_dispatch_policy(policy_type: str, policy_json: str) -> str:
    """
    HTTP POST: 下发策略到 PCF。
    Args:
        policy_details: 具体的策略内容 JSON 字符串 (policy_details)。
    """
    return dispatch_policy_to_pcf(policy_type, policy_json)

@tool
def tool_evaluate_sla(supi: str, flow_id: str, k: float=0.2) -> str:
    """Evaluate whether the performance metrics of a specific flow meet the SLA requirements.
    Args:
        supi: 用户唯一标识 SUPI。
        flow_id: 业务流 ID。
        k: 评估阈值，默认 0.2，表示允许的性能偏离程度。
    Returns:
        "satisfied" if all key performance indicators meet the SLA requirements within the threshold k, otherwise "violated".
    """
    snapshot = get_latest_snapshot_data() or {}
    app_data = snapshot.get("apps", []) if isinstance(snapshot, dict) else []
    target_supi = str(supi or "").strip()
    for app in app_data:
        app_supi = ""
        if isinstance(app, dict):
            app_supi = str(app.get("supi") or "").strip()
        flows = app.get("flows", [])
        for flow in flows:
            flow_supi = app_supi or str(flow.get("supi") or "").strip()
            if flow_supi == target_supi and flow.get("flow_id") == flow_id:
                # 这里可以根据 feedback_json 中的性能指标与 flow 中的要求进行对比，生成评估结果
                lat_req = float(flow.get("lat") or 0.0)
                jitter_req = float(flow.get("jitter_req") or 0.0)
                loss_req = float(flow.get("loss_req") or 0.0)
                gbr_ul = float(flow.get("gbr_ul") or 0.0)
                gbr_dl = float(flow.get("gbr_dl") or 0.0)
                sim_latency = float(flow.get("sim_latency") or 0.0)
                sim_jitter = float(flow.get("sim_jitter") or 0.0)
                sim_loss_rate = float(flow.get("sim_loss_rate") or 0.0)
                sim_throughput_ul = float(flow.get("sim_throughput_ul") or 0.0)
                sim_throughput_dl = float(flow.get("sim_throughput_dl") or 0.0)

                k_lat = (sim_latency - lat_req) / (lat_req if lat_req > 0 else 1.0)
                k_jitter = (sim_jitter - jitter_req) / (jitter_req if jitter_req > 0 else 1.0)
                # k_loss = (sim_loss_rate - loss_req) / (loss_req if loss_req > 0 else 1.0)
                k_ul = (gbr_ul - sim_throughput_ul) / (gbr_ul if gbr_ul > 0 else 1.0)
                k_dl = (gbr_dl - sim_throughput_dl) / (gbr_dl if gbr_dl > 0 else 1.0)
                if all(k_i < 0 for k_i in [k_lat, k_jitter, k_ul, k_dl]):
                    return "satisfied"
                if all(k_i <= k for k_i in [k_lat, k_jitter, k_ul, k_dl]):
                    return "satisfied"
                return f"{flow.get('flow_id')} violated"
    return "violated"

@tool
def tool_update_db_after_success(supi:str, policy: str = "") -> str:
    """
    Commit: 在所有策略成功后，调用此工具更新数据库中的UeContext。
    """
    try:
        if not supi:
            return "数据库更新失败: supi 为空"

        parsed_policy: Dict[str, Any] = {}
        if isinstance(policy, str) and policy.strip():
            try:
                parsed_policy = json.loads(policy)
            except Exception:
                parsed_policy = {}
        elif isinstance(policy, dict):
            parsed_policy = policy

        # 关键步骤：兼容不同来源的字段命名，统一为 upsert_ue_context 参数
        sm_policy_data = parsed_policy.get("sm_policy_data", parsed_policy.get("smPolicyData"))
        pcc_rules = parsed_policy.get("pcc_rules", parsed_policy.get("pccRules"))
        qos_decs = parsed_policy.get("qos_decs", parsed_policy.get("qosDecs"))
        sess_rules = parsed_policy.get("sess_rules", parsed_policy.get("sessRules"))
        traff_cont_decs = parsed_policy.get("traff_cont_decs", parsed_policy.get("traffContDecs"))
        chg_decs = parsed_policy.get("chg_decs", parsed_policy.get("chgDecs"))
        catalog = get_ue_flow_catalog_by_supi(supi)

        ok = upsert_ue_context(
            supi=supi,
            sm_policy_data=sm_policy_data,
            pcc_rules=pcc_rules,
            qos_decs=qos_decs,
            sess_rules=sess_rules,
            traff_cont_decs=traff_cont_decs,
            chg_decs=chg_decs,
            app_catalog=catalog.get("app_catalog") or [],
            flow_catalog=catalog.get("flow_catalog") or [],
        )

        return "数据库更新成功" if ok else "数据库更新失败"
    except Exception as e:
        return f"数据库更新失败: {str(e)}"

# --- Output Structure ---
class FeedbackReport(BaseModel):
    execution_status: str = Field(description="整体执行状态: 'Success', 'Partial Success', 'Failed'")
    performance_metrics: str = Field(description="从网元获取的关键性能指标摘要")
    violation_details: str = Field(description="如果有违规(如时延超标)，详细描述；否则为 'None'")
    correction_suggestion: str = Field(description="给意图识别Agent的修正建议。例如：'当前拥塞，建议降级为1080P' 或 'None'")

class PolicyDispatchAgent(BaseAgent):
    def __init__(self, model_name="qwen3-30b-a3b-instruct-2507"):
        super().__init__(model_name=model_name)
        self.logger = setup_logger(self.__class__.__name__, default_msg_color="\033[92m") # 绿色日志
        
        # Tools
        self.tools = [tool_dispatch_policy, tool_update_db_after_success, tool_evaluate_sla]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Parser
        self.output_parser = PydanticOutputParser(pydantic_object=FeedbackReport)

    def execute_and_evaluate(self, strategy_output) -> FeedbackReport:
        # Redirect to new Fail-Fast implementation
        return self._execute_fail_fast(strategy_output)

    @staticmethod
    def _strip_rule_prefix(candidate: Any) -> Optional[str]:
        if candidate is None:
            return None
        text = str(candidate).strip()
        if not text:
            return None
        for prefix in ("pcc-", "qos-", "sess-", "smp-", "ursp-"):
            if text.startswith(prefix) and len(text) > len(prefix):
                return text[len(prefix):]
        return text

    @staticmethod
    def _contains_failed_marker(text: str) -> bool:
        normalized = (text or "").lower()
        return ("failed" in normalized) or ("失败" in normalized)

    @staticmethod
    def _parse_feedback_status(feedback_text: Any) -> str:
        if isinstance(feedback_text, dict):
            completed = feedback_text.get("completed")
            expected = feedback_text.get("expected")
            received = feedback_text.get("received")
            if completed is True:
                return "success"
            if isinstance(expected, int) and isinstance(received, int):
                if received <= 0:
                    return "failed"
                if received < expected:
                    return "partial"
                if received >= expected:
                    return "success"
            return "unknown"

        text = str(feedback_text or "")
        if "Status: Success" in text:
            return "success"
        if "Status: Partial Success" in text:
            return "partial"
        if "Status: Failed" in text or "失败" in text:
            return "failed"
        return "unknown"

    @staticmethod
    def _extract_json_payload_from_dispatch_output(dispatch_output: str) -> Optional[Dict[str, Any]]:
        text = str(dispatch_output or "").strip()
        json_start = text.find("{")
        if json_start < 0:
            return None
        try:
            payload = json.loads(text[json_start:])
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _extract_feedback_outputs(cls, dispatch_output: str) -> List[Dict[str, Any]]:
        payload = cls._extract_json_payload_from_dispatch_output(dispatch_output)
        if not isinstance(payload, dict):
            return []
        ack = payload.get("ack")
        if isinstance(ack, dict):
            return [ack]
        return []

    @staticmethod
    def _summarize_feedback_outputs(feedback_outputs: List[Dict[str, Any]]) -> str:
        if not feedback_outputs:
            return "No ack feedback received."
        ack = feedback_outputs[-1]
        summary = {
            "request_id": ack.get("request_id"),
            "expected": ack.get("expected"),
            "received": ack.get("received"),
            "completed": ack.get("completed"),
            "result_count": len(ack.get("results", [])) if isinstance(ack.get("results"), list) else 0,
        }
        return json.dumps(summary, ensure_ascii=False)

    @staticmethod
    def _build_feedback_report_from_ack(feedback_outputs: List[Dict[str, Any]], *, aborted: bool) -> FeedbackReport:
        if not feedback_outputs:
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="No ack feedback received.",
                violation_details="Missing ack in PDA dispatch response.",
                correction_suggestion="Check PDA downstream ack path and PCF response format.",
            )

        ack = feedback_outputs[-1]
        expected = ack.get("expected")
        received = ack.get("received")
        completed = ack.get("completed")
        results = ack.get("results", []) if isinstance(ack.get("results"), list) else []

        if completed is True:
            execution_status = "Success"
            violation_details = "None"
            correction_suggestion = "None"
        elif isinstance(expected, int) and isinstance(received, int) and received > 0:
            execution_status = "Partial Success"
            violation_details = f"Ack incomplete: expected={expected}, received={received}, completed={completed}"
            correction_suggestion = "Retry policy dispatch or inspect missing ack results."
        else:
            execution_status = "Failed"
            violation_details = f"Ack failed: expected={expected}, received={received}, completed={completed}"
            correction_suggestion = "Check policy dispatch pipeline and downstream executor status."

        if aborted and execution_status == "Success":
            execution_status = "Partial Success"
            correction_suggestion = "Check execution logs for downstream interruption."

        return FeedbackReport(
            execution_status=execution_status,
            performance_metrics=json.dumps(
                {
                    "request_id": ack.get("request_id"),
                    "expected": expected,
                    "received": received,
                    "completed": completed,
                    "results": results,
                },
                ensure_ascii=False,
            ),
            violation_details=violation_details,
            correction_suggestion=correction_suggestion,
        )

    @staticmethod
    def _extract_flow_id(policy_details: Any) -> Optional[str]:
        data: Dict[str, Any] = {}
        if isinstance(policy_details, dict):
            data = policy_details
        elif hasattr(policy_details, "model_dump"):
            try:
                dumped = policy_details.model_dump(mode="json")
                if isinstance(dumped, dict):
                    data = dumped
            except Exception:
                data = {}

        flow_id = PolicyDispatchAgent._strip_rule_prefix(data.get("flow_id") or data.get("flowId"))
        if flow_id:
            return str(flow_id)

        # 关键步骤：优先适配当前策略数据习惯，flow_id 通常位于 qosDecs 的 key。
        qos_decs = data.get("qos_decs", data.get("qosDecs"))
        if isinstance(qos_decs, dict) and qos_decs:
            first_qos = next(iter(qos_decs.values()))
            if isinstance(first_qos, dict):
                qos_id = PolicyDispatchAgent._strip_rule_prefix(first_qos.get("qosId") or first_qos.get("qos_id"))
                if qos_id:
                    return str(qos_id)
            first_qos_key = PolicyDispatchAgent._strip_rule_prefix(next(iter(qos_decs.keys())))
            if first_qos_key:
                return str(first_qos_key)

        pcc_rules = data.get("pcc_rules", data.get("pccRules"))
        if isinstance(pcc_rules, dict) and pcc_rules:
            first_rule = next(iter(pcc_rules.values()))
            if isinstance(first_rule, dict):
                rid = PolicyDispatchAgent._strip_rule_prefix(
                    first_rule.get("flow_id") or first_rule.get("flowId") or first_rule.get("pccRuleId")
                )
                if rid:
                    return str(rid)
            first_key = PolicyDispatchAgent._strip_rule_prefix(next(iter(pcc_rules.keys())))
            if first_key:
                return str(first_key)

        return None

    def _run_tool_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_rounds: int = 4,
        policy_index: Optional[int] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """让 LLM 自主选择工具并执行，返回最终消息与工具调用轨迹。"""
        import time

        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        tool_traces: List[Dict[str, Any]] = []

        for _ in range(max_rounds):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                return ai_msg, tool_traces

            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("args", {}) or {}
                start = time.perf_counter()
                tool = self.tool_map.get(tool_name)

                if tool is None:
                    tool_output = f"Unknown tool: {tool_name}"
                else:
                    try:
                        tool_output = tool.invoke(tool_args)
                    except Exception as exc:
                        tool_output = f"Tool execution error: {str(exc)}"

                elapsed = time.perf_counter() - start
                if tool_name == "tool_dispatch_policy":
                    log_timing(
                        self.logger,
                        "pda_dispatch_call",
                        elapsed,
                        policy_index=policy_index,
                        policy_type=tool_args.get("policy_type", "unknown"),
                    )
                elif tool_name == "tool_get_feedback":
                    log_timing(
                        self.logger,
                        "pda_feedback_call",
                        elapsed,
                        policy_index=policy_index,
                    )
                elif tool_name == "tool_update_db_after_success":
                    log_timing(self.logger, "pda_commit_call", elapsed)

                tool_output_str = str(tool_output)
                tool_traces.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "output": tool_output_str,
                    }
                )
                messages.append(
                    ToolMessage(
                        content=tool_output_str,
                        tool_call_id=call.get("id", ""),
                    )
                )

        return messages[-1], tool_traces

    def _execute_fail_fast(self, strategy_output) -> FeedbackReport:
        """
        New Fail-Fast Implementation
        """
        import time

        total_start = time.perf_counter()
        log_event(self.logger, "pda_execute_start")

        # 1. 解析策略列表
        policies_to_dispatch = []
        supi = ""
        if isinstance(strategy_output, dict):
            policies_to_dispatch = strategy_output.get("all_policies", [])
            supi = str(strategy_output.get("supi", "")).strip()
        elif hasattr(strategy_output, "all_policies"):
            policies_to_dispatch = strategy_output.all_policies
            supi = str(getattr(strategy_output, "supi", "")).strip()
            
        if not policies_to_dispatch:
            log_timing(self.logger, "pda_total", time.perf_counter() - total_start, status="no_policies")
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="N/A",
                violation_details="输出未包含策略 (No policies found)",
                correction_suggestion="请检查 OSA 优化输出是否为空"
            )

        execution_logs = []
        aborted = False
        successful_policy_payloads: List[str] = []
        cleaned_feedback_outputs: List[Dict[str, Any]] = []

        # 2. 逐个执行
        for idx, p in enumerate(policies_to_dispatch):
            policy_start = time.perf_counter()
            p_type = p.get("policy_type") if isinstance(p, dict) else p.policy_type
            p_details = p.get("policy_details") if isinstance(p, dict) else p.policy_details
            policy_flow_id = str(p.get("flow_id") or "").strip() if isinstance(p, dict) else str(getattr(p, "flow_id", "") or "").strip()
            policy_id = str(p.get("policy_id") or "").strip() if isinstance(p, dict) else str(getattr(p, "policy_id", "") or "").strip()
            policy_app_id = str(p.get("app_id") or "").strip() if isinstance(p, dict) else str(getattr(p, "app_id", "") or "").strip()

            details_payload = _json_friendly(p_details)
            if policy_id and isinstance(details_payload, dict):
                details_payload.setdefault("policy_id", policy_id)
            details_str = json.dumps(details_payload, ensure_ascii=False)

            step_log = f"--- 策略 #{idx+1} ({p_type}) ---"
            
            try:
                # 关键步骤：由 Agent 自主决定调用哪个绑定工具
                flow_id = policy_flow_id or self._extract_flow_id(details_payload)
                tool_system_prompt = PDA_EXECUTION_TOOL_SYSTEM_PROMPT
                tool_user_prompt = (
                    f"策略编号: {idx+1}\n"
                    f"supi: {supi}\n"
                    f"app_id: {policy_app_id}\n"
                    f"policy_id: {policy_id}\n"
                    f"flow_id: {flow_id or ''}\n"
                    f"policy_type: {p_type}\n"
                    f"policy_details_json: {details_str}\n\n"
                    "请你自己判断并调用合适的工具完成执行。"
                    "完成后用一句话总结。"
                )

                _, tool_traces = self._run_tool_loop(
                    tool_system_prompt,
                    tool_user_prompt,
                    max_rounds=4,
                    policy_index=idx + 1,
                )

                dispatch_outputs = [
                    t["output"] for t in tool_traces if t["name"] == "tool_dispatch_policy"
                ]
                sla_outputs = [
                    t["output"] for t in tool_traces if t["name"] == "tool_evaluate_sla"
                ]

                if not dispatch_outputs:
                    self.logger.error("未调用下发工具。终止后续流程。")
                    step_log += "\n[下发]: 未执行 tool_dispatch_policy"
                    step_log += "\n[结果]: 下发阶段缺失。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

                dispatch_res = dispatch_outputs[-1]
                self.logger.info(f"[下发结果]: {dispatch_res}")
                step_log += f"\n[下发]: {dispatch_res}"

                if self._contains_failed_marker(dispatch_res):
                    self.logger.error("策略下发失败。终止后续流程。")
                    step_log += "\n[结果]: 下发失败。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

                feedback_outputs = self._extract_feedback_outputs(dispatch_res)
                if not feedback_outputs:
                    self.logger.error(f"策略 #{idx+1} 未获取 ack 反馈。终止流程。")
                    step_log += "\n[反馈]: missing ack in dispatch response"
                    step_log += "\n[结果]: 缺少执行反馈。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

                feedback_res = feedback_outputs[-1]
                self.logger.info(f"[反馈结果]: {feedback_res}")
                step_log += f"\n[反馈]: {self._summarize_feedback_outputs(feedback_outputs)}"
                cleaned_feedback_outputs = feedback_outputs

                feedback_status = self._parse_feedback_status(feedback_res)
                if feedback_status == "success":
                    self.logger.info(f"策略 #{idx+1} 执行成功。")
                    step_log += "\n[结果]: 执行正常。继续下一条。"
                elif feedback_status == "partial":
                    self.logger.warning(f"策略 #{idx+1} 执行部分成功。")
                    step_log += "\n[结果]: 执行部分成功。继续下一条。"
                else:
                    self.logger.error(f"策略 #{idx+1} 执行失败 (反馈异常)。终止流程。")
                    step_log += "\n[结果]: 执行失败。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

                if flow_id:
                    if not sla_outputs:
                        self.logger.error(f"策略 #{idx+1} 未执行 SLA 评估。终止流程。")
                        step_log += "\n[SLA评估]: 未执行 tool_evaluate_sla"
                        step_log += "\n[结果]: 缺少 SLA 评估。终止流程。"
                        execution_logs.append(step_log)
                        aborted = True
                        break

                    sla_res = str(sla_outputs[-1]).strip().lower()
                    step_log += f"\n[SLA评估]: {sla_outputs[-1]}"
                    if "violated" in sla_res:
                        self.logger.error(f"策略 #{idx+1} SLA 违约。终止流程。")
                        step_log += "\n[结果]: SLA violated。终止流程。"
                        execution_logs.append(step_log)
                        aborted = True
                        break

                successful_policy_payloads.append(details_str)

            except Exception as e:
                step_log += f"\n[异常]: {str(e)}"
                execution_logs.append(step_log)
                aborted = True
                break
            
            execution_logs.append(step_log)
            log_timing(self.logger, "pda_policy_total", time.perf_counter() - policy_start, policy_index=idx + 1)

        # [新增] 只有在未中止的情况下才提交所有结果到数据库
        if not aborted and execution_logs:
            self.logger.info("所有策略执行完毕且状态正常，正在更新数据库...")
            commit_system_prompt = PDA_COMMIT_TOOL_SYSTEM_PROMPT
            commit_policy = successful_policy_payloads[-1] if successful_policy_payloads else ""
            commit_user_prompt = (
                "执行流程已完成且未中止。\n"
                f"supi: {supi}\n"
                f"policy_json: {commit_policy}\n"
                "请判断是否调用数据库更新工具并执行。"
            )
            _, commit_traces = self._run_tool_loop(
                commit_system_prompt,
                commit_user_prompt,
                max_rounds=2,
            )
            commit_outputs = [
                t["output"] for t in commit_traces if t["name"] == "tool_update_db_after_success"
            ]
            if commit_outputs:
                commit_res = commit_outputs[-1]
                self.logger.info(f"[数据库更新]: {commit_res}")
                execution_logs.append(f"\n[数据库更新]: {commit_res}")
            else:
                self.logger.warning("Agent 未调用数据库更新工具。")
                execution_logs.append("\n[数据库更新]: 跳过 (Agent 判定无需调用)")
        elif aborted:
            self.logger.warning("流程被中止，跳过数据库更新。")
            execution_logs.append("\n[数据库更新]: 跳过 (因流程中止)")

        # 3. Report
        full_log = "\n".join(execution_logs)

        try:
            result = self._build_feedback_report_from_ack(
                cleaned_feedback_outputs,
                aborted=aborted,
            )
            log_timing(self.logger, "pda_total", time.perf_counter() - total_start, status="success")
            return result
        except Exception as e:
            self.logger.error(f"Report Gen Error: {e}")
            log_timing(self.logger, "pda_total", time.perf_counter() - total_start, status="error")
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="See logs",
                violation_details="Report Gen Error",
                correction_suggestion=f"Log: {full_log[:100]}"
            )
