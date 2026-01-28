from typing import List, Optional
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logger import setup_logger
from .basemodel import BaseAgent
from .Prompt import PDA_SYSTEM_PROMPT, PDA_USER_FEEDBACK_PROMPT
from tools.pcf_tools import dispatch_policy_to_pcf, get_network_feedback

# --- Tool Wrappers ---
@tool
def tool_dispatch_policy(policy_type: str, policy_json: str) -> str:
    """
    HTTP POST: 下发策略到 PCF。
    Args:
        policy_type: 策略类型，例如 'UrspRuleRequest' 或 'SmPolicyDecision'
        policy_json: 具体的策略内容 JSON 字符串 (policy_details)。
    """
    return dispatch_policy_to_pcf(policy_type, policy_json)

@tool
def tool_get_feedback(policy_id: str) -> str:
    """
    Query: 获取网元对该策略的执行反馈数据。
    Args:
        policy_id: 策略 ID。
    """
    return get_network_feedback(policy_id)

@tool
def tool_update_db_after_success() -> str:
    """
    Commit: 在所有策略成功后，调用此工具更新数据库中的网络状态。
    """
    from tools.commit_tool import commit_optimization_result_to_db
    # 传入空字符串作为触发信号
    return commit_optimization_result_to_db("")

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
        self.tools = [tool_dispatch_policy, tool_get_feedback, tool_update_db_after_success]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Parser
        self.output_parser = PydanticOutputParser(pydantic_object=FeedbackReport)

    def execute_and_evaluate(self, strategy_output) -> FeedbackReport:
        # Redirect to new Fail-Fast implementation
        return self._execute_fail_fast(strategy_output)

    def _execute_fail_fast(self, strategy_output) -> FeedbackReport:
        """
        New Fail-Fast Implementation
        """
        import time
        import json
        from tools.pcf_tools import dispatch_policy_to_pcf, get_network_feedback, commit_optimization_result
        from langchain_core.prompts import ChatPromptTemplate
        from multi_agents.Prompt import PDA_SYSTEM_PROMPT

        # 1. 解析策略列表
        policies_to_dispatch = []
        if isinstance(strategy_output, dict):
            policies_to_dispatch = strategy_output.get("all_policies", [])
        elif hasattr(strategy_output, "all_policies"):
            policies_to_dispatch = strategy_output.all_policies
            
        if not policies_to_dispatch:
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="N/A",
                violation_details="输出未包含策略 (No policies found)",
                correction_suggestion="请检查 OSA 优化输出是否为空"
            )

        execution_logs = []
        aborted = False

        # 2. 逐个执行
        for idx, p in enumerate(policies_to_dispatch):
            p_type = p.get("policy_type") if isinstance(p, dict) else p.policy_type
            p_details = p.get("policy_details") if isinstance(p, dict) else p.policy_details
            
            if hasattr(p_details, "model_dump_json"):
                details_str = p_details.model_dump_json()
            elif isinstance(p_details, dict):
                details_str = json.dumps(p_details, ensure_ascii=False)
            else:
                details_str = str(p_details)

            step_log = f"--- 策略 #{idx+1} ({p_type}) ---"
            
            try:
                # Dispatch
                dispatch_res = dispatch_policy_to_pcf(p_type, details_str)
                self.logger.info(f"[下发结果]: {dispatch_res}")
                step_log += f"\n[下发]: {dispatch_res}"
                
                if "Failed" in dispatch_res:
                    self.logger.error("策略下发失败。终止后续流程。")
                    step_log += "\n[结果]: 下发失败。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

                # Monitor
                mock_policy_id = f"pol-{idx}-{int(time.time())}"
                feedback_res = get_network_feedback(mock_policy_id)
                self.logger.info(f"[反馈结果]: {feedback_res}")
                step_log += f"\n[反馈]: {feedback_res}"

                if "Status: Success" in feedback_res:
                    self.logger.info(f"策略 #{idx+1} 执行成功。")
                    step_log += "\n[结果]: 执行正常。继续下一条。"
                    # 注意：不在循环内提交 DB，改为在最后统一提交
                    
                elif "Status: Partial Success" in feedback_res:
                     self.logger.warning(f"策略 #{idx+1} 执行部分成功。")
                     step_log += "\n[结果]: 执行部分成功。继续下一条。"
                else:
                    self.logger.error(f"策略 #{idx+1} 执行失败 (反馈异常)。终止流程。")
                    step_log += "\n[结果]: 执行失败。终止流程。"
                    execution_logs.append(step_log)
                    aborted = True
                    break

            except Exception as e:
                step_log += f"\n[异常]: {str(e)}"
                execution_logs.append(step_log)
                aborted = True
                break
            
            execution_logs.append(step_log)

        # [新增] 只有在未中止的情况下才提交所有结果到数据库
        if not aborted and execution_logs:
            self.logger.info("所有策略执行完毕且状态正常，正在更新数据库...")
            commit_res = commit_optimization_result("") # 使用空字符串触发默认行为
            self.logger.info(f"[数据库更新]: {commit_res}")
            execution_logs.append(f"\n[数据库更新]: {commit_res}")
        elif aborted:
            self.logger.warning("流程被中止，跳过数据库更新。")
            execution_logs.append("\n[数据库更新]: 跳过 (因流程中止)")

        # 3. Report
        full_log = "\n".join(execution_logs)
        status_hint = "Partial Success or Failed" if aborted else "Success"

        prompt = ChatPromptTemplate.from_messages([
            ("system", PDA_SYSTEM_PROMPT),
            ("user", PDA_USER_FEEDBACK_PROMPT)
        ])
        
        # Combine all inputs into a single format_messages call
        messages = prompt.format_messages(
            full_log=full_log,
            status_hint=status_hint,
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        try:
            response = self.llm.invoke(messages)
            return self.output_parser.parse(response.content)
        except Exception as e:
            self.logger.error(f"Report Gen Error: {e}")
            return FeedbackReport(
                execution_status="Failed",
                performance_metrics="See logs",
                violation_details="Report Gen Error",
                correction_suggestion=f"Log: {full_log[:100]}"
            )