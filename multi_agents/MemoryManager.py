import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from .basemodel import BaseAgent
from .Embedding import NaturalLanguageEncoder
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryManager:
    """
    记忆管理器
    实现功能：
    1. 短期记忆 (Short-term Memory): 保存最近的 N 轮对话。
    2. 长期记忆 (Long-term Memory): 基于向量检索的历史摘要存储。
    3. 记忆整合 (Consolidation): 使用 LLM 将过期的短期记忆总结并存入长期记忆。
    """
    def __init__(self, 
                 short_term_limit: int = 10, 
                 long_term_file: str = "long_term_memory.json",
                 similarity_threshold: float = 0.5):
        """
        :param short_term_limit: 短期记忆保留的消息条数上限，超过触发总结
        :param long_term_file: 长期记忆存储文件路径
        :param similarity_threshold: 检索长期记忆的相似度阈值
        """
        # 1. Short-term Memory (Deque)
        self.short_term_memory = deque()
        self.short_term_limit = short_term_limit
        
        # 2. Long-term Memory (Vector Store)
        self.long_term_file = long_term_file
        self.long_term_memory = self._load_long_term_memory()
        
        # 3. Components
        # 使用现有的 Embedding Encoder
        self.encoder = NaturalLanguageEncoder()
        # 使用现有的 Agent 基类进行总结
        self.agent = BaseAgent()
        self.similarity_threshold = similarity_threshold

    def _load_long_term_memory(self) -> List[Dict]:
        if os.path.exists(self.long_term_file):
            try:
                with open(self.long_term_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"无法加载长期记忆: {e}")
                return []
        return []

    def _save_long_term_memory(self):
        try:
            with open(self.long_term_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"无法保存长期记忆: {e}")

    def add_memory(self, role: str, content: str):
        """
        添加新的交互到短期记忆。
        如果超出限制，则触发记忆整合（移动旧记忆到长期记忆）。
        """
        memory_item = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.short_term_memory.append(memory_item)
        
        # 简单策略：当短期记忆超过限制时，整合最早的一半记忆
        if len(self.short_term_memory) > self.short_term_limit:
            self.consolidate_memory()

    def consolidate_memory(self):
        """
        将旧的短期记忆总结并存入长期记忆。
        策略：取出最早的 n/2 条消息进行总结。
        """
        items_to_summarize = []
        # 保留最近的一半，归档旧的一半
        keep_count = self.short_term_limit // 2
        
        while len(self.short_term_memory) > keep_count:
            items_to_summarize.append(self.short_term_memory.popleft())
            
        if not items_to_summarize:
            return

        # 格式化文本供 LLM 总结
        text_chunk = "\n".join([f"{item['role']}: {item['content']}" for item in items_to_summarize])
        
        logger.info(f"正在整合 {len(items_to_summarize)} 条记忆...")
        
        # LLM Summarization
        summary = self._summarize_text(text_chunk)
        
        if summary and summary != "Summary failed":
            # Embedding
            emb_result = self.encoder.encode(summary)
            vector = emb_result.get("vector")
            
            if vector:
                memory_entry = {
                    "content": summary,
                    "vector": vector,
                    "timestamp": time.time(),
                    "ref_start_time": items_to_summarize[0]['timestamp'],
                    "raw_snippet": text_chunk[:100] + "..." # Optional: keep glimpse of raw
                }
                self.long_term_memory.append(memory_entry)
                self._save_long_term_memory()
                logger.info(f"长期记忆已更新: {summary[:30]}...")
            else:
                 logger.warning("Embedding 生成失败，跳过长期记忆存储")

    def _summarize_text(self, text: str) -> str:
        """调用 LLM 总结文本"""
        prompt = f"""
        请将以下对话片段总结为简洁的知识点或事实陈述。忽略客套话，保留关键信息（如用户偏好、任务目标、重要参数）。
        
        对话片段:
        {text}
        
        总结:
        """
        try:
            llm = self.agent.get_llm()
            # Langchain invoke
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"总结失败: {e}")
            return "Summary failed"

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        根据当前 Query 检索相关上下文。
        返回：
        {
            "short_term": [List of recent messages],
            "long_term": [List of relevant summaries strings]
        }
        """
        # 1. 获取 Query 向量
        query_emb = self.encoder.encode(query).get("vector")
        
        relevant_long_term = []
        if query_emb and self.long_term_memory:
            # 简单的余弦相似度计算
            q_vec = np.array(query_emb)
            norm_q = np.linalg.norm(q_vec)
            
            scores = []
            for item in self.long_term_memory:
                d_vec = np.array(item['vector'])
                norm_d = np.linalg.norm(d_vec)
                
                if norm_q > 0 and norm_d > 0:
                    score = np.dot(q_vec, d_vec) / (norm_q * norm_d)
                else:
                    score = 0.0
                scores.append((score, item))
            
            # 按相似度降序排列
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # 筛选 Top K 且 高于阈值
            for score, item in scores:
                if len(relevant_long_term) >= top_k:
                    break
                if score >= self.similarity_threshold:
                    relevant_long_term.append(item['content'])

        return {
            "short_term": list(self.short_term_memory),
            "long_term": relevant_long_term
        }
