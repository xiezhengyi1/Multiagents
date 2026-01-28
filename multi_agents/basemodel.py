import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class BaseAgent:
    def __init__(self, model_name="qwen-plus", temperature=0):
        """
        初始化基础Agent
        :param model_name: 模型名称，默认 qwen-plus (阿里云百炼)
        :param temperature: 温度系数，0表示最确定的输出
        """
        # 优先使用 OPENAI_API_KEY，如果没有则尝试 DASHSCOPE_API_KEY
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 优先使用 OPENAI_BASE_URL，默认 fallback 到阿里云 DashScope 兼容端点
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
             # 为了避免在导入时报错（如果没有配置key），这里可以打印警告或者在运行时抛出
             print("警告: 未检测到 API Key (OPENAI_API_KEY 或 DASHSCOPE_API_KEY)")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

    def get_llm(self):
        return self.llm
