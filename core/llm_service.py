"""模块5：LLM服务（统一接口）"""
from typing import List, Dict
from openai import OpenAI

from config.settings import settings
from utils.logger import logger
from utils.exceptions import LLMError


class LLMService:
    """大模型问答（统一入参出参）"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        self.model = settings.LLM_MODEL

    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """
        调用LLM生成回答
        messages: [{"role": "system/user/assistant", "content": "..."}]
        """
        try:
            logger.info(f"调用LLM: model={self.model}, messages数={len(messages)}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            logger.info(f"LLM回答完成: {len(answer)}字符")
            return answer

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise LLMError(f"LLM调用异常: {e}")

    def chat_stream(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2048):
        """流式调用LLM"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            raise LLMError(f"LLM流式调用异常: {e}")
