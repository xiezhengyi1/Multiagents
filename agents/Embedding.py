import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from openai import OpenAI
from utils.logger import setup_logger

logger = setup_logger(__name__)


# 中文标注：全局共享 OpenAI 客户端，避免重复初始化
def _get_shared_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not api_key:
        raise RuntimeError("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


_SHARED_CLIENT: Optional[OpenAI] = None


def get_embedding_client() -> OpenAI:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = _get_shared_openai_client()
    return _SHARED_CLIENT

class BaseEncoder(ABC):
    """编码器基类"""
    @abstractmethod
    def encode(self, input_data: Any) -> Any:
        pass

class NaturalLanguageEncoder(BaseEncoder):
    """
    自然语言编码器 (NL Encoder)
    """
    EMBEDDING_MODEL_NAME = "text-embedding-v4"

    def __init__(self, embedding_model=None):
        # 中文标注：共享客户端，初始化失败直接抛出
        self.embedding_model = embedding_model or get_embedding_client()

    def encode(self, text: str) -> Dict[str, Any]:
        """
        输入：用户自然语言字符串
        输出：语义特征字典 (包含 vector)
        """
        if not text:
            return {"raw_text": "", "detected_keywords": [], "vector": []}

        # 1. 基础清理
        text_clean = text.strip()
        # 2. 简单关键词提取（可扩展）
        keyword_map = {
            "带宽": "bandwidth",
            "速率": "throughput",
            "延迟": "latency",
            "时延": "latency",
            "卡顿": "jitter",
            "丢包": "packet_loss",
            "视频": "video",
            "通话": "voice_call",
            "游戏": "gaming",
            "断线": "disconnect",
        }
        detected_keywords = []
        for k, v in keyword_map.items():
            if k in text_clean:
                detected_keywords.append(v)

        # 2. 向量化 - 中文标注：失败直接抛出，不再静默返回空向量
        vector = []
        if text_clean:
            completion = self.embedding_model.embeddings.create(
                model=self.EMBEDDING_MODEL_NAME,
                input=text_clean,
            )
            vector = completion.data[0].embedding

        # 4. 构造输出
        return {
            "raw_text": text_clean,
            "detected_keywords": detected_keywords,
            "vector": vector
        }

class UserDataEncoder(BaseEncoder):
    """用户数据编码器 (Data Encoder)"""
    EMBEDDING_MODEL_NAME = "text-embedding-v4"

    def __init__(self, embedding_model=None):
        # 中文标注：共享客户端，初始化失败直接抛出
        self.embedding_model = embedding_model or get_embedding_client()

    def encode(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入：UeContext 字典
        输出：特征字典 (包含 vector)
        """
        if not user_context:
            return {"text_repr": "", "vector": [], "raw_data": {}}

        # 1. 数据序列化：提取关键信息并构造自然语言描述
        # 这种 "Text-Serialization" 是对齐模态的最简单方法
        text_parts = []
        
        supi = user_context.get("supi", "Unknown")
        text_parts.append(f"User {supi}.")

        sm_policy_map = user_context.get("smPolicyData") or {}
        active_slices = []
        
        for p_id, ue_sm_data in sm_policy_map.items():
            if not isinstance(ue_sm_data, dict):
                continue
            
            policy_ctx = ue_sm_data.get("policyContext") or {}
            slice_info = policy_ctx.get("sliceInfo") or {}
            sst = slice_info.get("sst")
            
            # 使用自然语言描述切片属性
            slice_desc = "Unknown Slice"
            if sst == 1: slice_desc = "high bandwidth eMBB slice"
            elif sst == 2: slice_desc = "low latency URLLC slice"
            elif sst == 3: slice_desc = "massive IoT slice"
            
            dnn = policy_ctx.get("dnn", "internet")
            rem_dl = ue_sm_data.get("remainGbrDL", 0)
            
            active_slices.append(f"Session {p_id} on {slice_desc} (SST {sst}) for service {dnn} with {rem_dl}Mbps DL remaining.")

        if active_slices:
            text_parts.append("Has active sessions: " + " ".join(active_slices))
        else:
            text_parts.append("No active sessions currently.")

        text_repr = " ".join(text_parts)

        # 2. 向量化 - 中文标注：失败直接抛出
        vector = []
        if text_repr:
            completion = self.embedding_model.embeddings.create(
                model=self.EMBEDDING_MODEL_NAME,
                input=text_repr
            )
            vector = completion.data[0].embedding

        return {
            "text_repr": text_repr,
            "vector": vector,
            "raw_data": user_context # 保留原始数据供规则检查
        }

class FeatureFusionLayer:
    """
    特征融合层
    功能：
    1. 接收 NL Encoder 和 Data Encoder 的输出。
    2. 计算意图向量与用户状态向量的相似度 (Dot Product / Cosine)。
    3. 构造包含语义相似度信息的增强 Prompt。
    """
    def __init__(self, alpha: float = 0.5, target_dim: Optional[int] = None):
        self.alpha = alpha
        self.target_dim = target_dim

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        if not vec:
            return []
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def _fold_to_dim(self, vec: List[float], dim: int) -> List[float]:
        if not vec or dim <= 0:
            return []
        if len(vec) <= dim:
            return vec
        folded = [0.0 for _ in range(dim)]
        for i, v in enumerate(vec):
            folded[i % dim] += v
        return folded

    def fuse(self, nl_features: Dict[str, Any], data_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        输出：融合后的向量特征，作为 LLM 的输入向量
        """
        nl_vec = nl_features.get("vector", [])
        data_vec = data_features.get("vector", [])

        # 相同维度：加权映射；不同维度：拼接映射
        if nl_vec and data_vec and len(nl_vec) == len(data_vec):
            fused_vec = [self.alpha * a + (1 - self.alpha) * b for a, b in zip(nl_vec, data_vec)]
        else:
            fused_vec = list(nl_vec) + list(data_vec)

        # 可选降维映射
        if self.target_dim is not None:
            fused_vec = self._fold_to_dim(fused_vec, self.target_dim)

        fused_vec = self._l2_normalize(fused_vec)

        return {
            "raw_text": nl_features.get("raw_text", ""),
            "text_repr": data_features.get("text_repr", ""),
            "intent_vector": nl_vec,
            "context_vector": data_vec,
            "fused_vector": fused_vec,
            "detected_keywords": nl_features.get("detected_keywords", []),
        }
