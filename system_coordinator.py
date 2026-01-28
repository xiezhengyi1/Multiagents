from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import uuid

# 引入数据库与模型
from database.connection import get_db
from database.models import SessionContext, EpisodicExperience
from database import engine

# 引入 Agents
from multi_agents.IntentEncodingAgent import IntentEncodingAgent
from multi_agents.OptimizationStrategyAgent import OptimizationStrategyAgent, OutputStrategy
from multi_agents.PolicyDispatchAgent import PolicyDispatchAgent, FeedbackReport
from multi_agents.MemoryManager import MemoryManager

# 初始化 Logger
logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """
    集成 三大 Agent + 记忆系统 + 数据库 的核心协调器
    """
    def __init__(self):
        self.ie_agent = IntentEncodingAgent()
        self.os_agent = OptimizationStrategyAgent()
        self.pd_agent = PolicyDispatchAgent()
        
        # 初始化 Memory Manager (文本记忆)
        self.memory_manager = MemoryManager(short_term_limit=5)
        
        # 数据库 Session 在方法内动态创建，避免长连接问题
        
    def _create_session(self, initial_intent) -> str:
        """创建新的任务会话 (Step A)"""
        db = next(get_db())
        try:
            session = SessionContext(
                current_step="intent",
                intent_data=initial_intent if isinstance(initial_intent, dict) else {"raw": str(initial_intent)},
                status="active"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.session_id
        except Exception as e:
            logger.error(f"创建 Session 失败: {e}")
            return str(uuid.uuid4()) # Fallback
        finally:
            db.close()
            
    def _update_session(self, session_id: str, step: str, policy_data=None, status="active"):
        """更新任务会话状态"""
        db = next(get_db())
        try:
            session = db.query(SessionContext).filter(SessionContext.session_id == session_id).first()
            if session:
                session.current_step = step
                session.status = status
                if policy_data:
                    # 确保 policy_data 转为 json 兼容格式
                    if hasattr(policy_data, "model_dump"): # Pydantic v2
                         session.policy_data = policy_data.model_dump()
                    elif hasattr(policy_data, "dict"): # Pydantic v1
                         session.policy_data = policy_data.dict()
                    else:
                         session.policy_data = policy_data
                db.commit()
        except Exception as e:
            logger.error(f"更新 Session 失败: {e}")
        finally:
            db.close()

    def _archive_experience(self, session_id: str, feedback: FeedbackReport):
        """将成功完成的任务归档到长期经验库 (Step B)"""
        db = next(get_db())
        try:
            # 1. 读取 Session 完整数据
            session = db.query(SessionContext).filter(SessionContext.session_id == session_id).first()
            if not session:
                return

            # 2. 计算 Reward Score (简单示例)
            reward = 0.0
            if feedback.execution_status == "Success":
                 reward = 1.0
            elif feedback.execution_status == "Partial Success":
                 reward = 0.5
            
            # 3. 生成嵌入向量 (如果 intent 是复杂的，这里应该调用 Embedding 模型)
            # 为了演示，我们暂时尝试用 MemoryManager 的 encoder 来生成向量
            raw_intent_str = str(session.intent_data)
            vector_data = self.memory_manager.encoder.encode(raw_intent_str)
            intent_vector = vector_data.get("vector") if vector_data else None
            
            # 维度截断/填充检查 (防止 1536 vs 1024 不匹配)
            if intent_vector and len(intent_vector) != 1024:
                # 简单处理：如果多了截断，如果是1536转1024可以保留前1024
                # 但最好是重新训练或者使用 PCA，这里仅做兼容性截断
                if len(intent_vector) > 1024:
                    intent_vector = intent_vector[:1024]
                else: 
                     # 填充0
                    intent_vector = intent_vector + [0.0] * (1024 - len(intent_vector))

            # 4. 存入 EpisodicExperience 表
            experience = EpisodicExperience(
                raw_intent=raw_intent_str,
                intent_vector=intent_vector, # 需要 pgvector 支持
                applied_policy=session.policy_data,
                environment_state={}, # TODO: 可以从 OSA 获取环境上下文
                feedback_metrics={"status": feedback.execution_status, "detail": feedback.performance_metrics},
                reward_score=reward
            )
            db.add(experience)
            db.commit()
            logger.info(f"闭环任务已归档至长期经验库, ID: {session_id}, Reward: {reward}")
            
        except Exception as e:
            logger.error(f"归档经验失败: {e}")
        finally:
            db.close()

    def run_loop(self, user_input: str) -> str:
        """
        运行完整的三智能体闭环流程 (含自动纠错闭环)
        User Input -> IEA -> (DB Session) -> OSA (RAG) -> (DB Session) -> PDA -> Feedback -> DB Archive
           ^                                                                        |
           |---------------------------(Feedback Loop)------------------------------|
        """
        logger.info(f"=== 收到用户请求: {user_input} ===")
        
        # 0. 记忆上下文检索
        memory_ctx = self.memory_manager.retrieve(user_input)
        short_term_context = [f"{m['role']}: {m['content']}" for m in memory_ctx['short_term']]
        long_term_context = memory_ctx['long_term']
        
        # 基础上下文
        base_context = "\n".join(short_term_context + long_term_context)
        current_context = base_context
        
        max_retries = 3
        last_response = "任务初始化..."

        for attempt in range(max_retries):
            logger.info(f"--- Round {attempt+1} ---")

            # 1. 意图解析 (IEA)
            # 这里的 Context 可能包含了上一轮的失败反馈
            intent_obj = self.ie_agent.analyze_intent(user_input, context=current_context)
            
            if not intent_obj:
                if attempt == 0:
                     return "抱歉，我无法理解您的意图，请重新描述。"
                else: 
                     break # 已经尝试过但无法解析有效意图
            
            # 第一轮时记录用户输入到 Memory (避免重复)
            if attempt == 0:
                self.memory_manager.add_memory("user", user_input)
                self.memory_manager.add_memory("system", f"Intent Identified: {intent_obj.raw_intent_summary}")
            else:
                logger.info(f"Re-adjusted Intent: {intent_obj.raw_intent_summary}")

            # 创建数据库 Session (每次 Loop 都是一次独立的尝试)
            session_id = self._create_session(intent_obj.model_dump())
            
            # 2. 策略生成 (OSA)
            # TODO: 这里应该先从 episodic_experience 表中检索相似案例 (RAG)
            strategy_output = self.os_agent.generate_strategy(intent_obj.model_dump())
            
            # 更新数据库
            self._update_session(session_id, "generation", policy_data=strategy_output)
            
            if not strategy_output or not strategy_output.all_policies:
                self._update_session(session_id, "generation", status="failed")
                # OSA 失败也视为一种 Feedback
                fail_reason = "Optimization Strategy Generation Failed."
                current_context += f"\n[System Feedback]: {fail_reason} Please try to relax constraints."
                last_response = fail_reason
                continue

            # 3. 策略下发与反馈 (PDA)
            feedback = self.pd_agent.execute_and_evaluate(strategy_output)
            
            # 更新数据库状态
            final_status = "completed" if feedback.execution_status == "Success" else "failed"
            self._update_session(session_id, "execution", status=final_status)
    
            # 4. 经验归档
            self._archive_experience(session_id, feedback)
            
            # 成功则直接返回
            if feedback.execution_status == "Success":
                response_msg = f"策略执行成功 (Round {attempt+1})。详情: {feedback.performance_metrics}"
                self.memory_manager.add_memory("system", response_msg)
                return response_msg
            
            # 失败则准备下一轮 Loop
            logger.warning(f"Round {attempt+1} Failed: {feedback.performance_metrics}")
            correction = feedback.correction_suggestion
            
            # 将 Feedback 添加到 Context 供 IEA 修正
            current_context += f"\n[Previous Attempt Failed]:\nStatus: {feedback.execution_status}\nMetrics: {feedback.performance_metrics}\nSuggestion: {correction}\nInstruction: Please adjust the intent to satisfy the suggestion."
            last_response = f"策略执行失败。建议: {correction}"
        
        # 超过最大重试次数
        final_msg = f"经过 {max_retries} 次尝试后任务失败。最后反馈: {last_response}"
        self.memory_manager.add_memory("system", final_msg)
        return final_msg

if __name__ == "__main__":
    # 配置日志
    logger = logging.getLogger("MultiAgentSystem")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 初始化并运行
    system = MultiAgentSystem()
    
    # 模拟一次对话
    print(system.run_loop("我需要为新的云游戏应用'GameNow'申请一个低延迟切片，要求延迟小于10ms。supi:imsi-46001"))
