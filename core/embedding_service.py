"""嵌入服务（本地 Ollama API）"""
from typing import List
import httpx

from config.settings import settings
from utils.logger import logger


class OllamaEmbeddingService:
    """本地 Ollama 文本嵌入服务"""

    def __init__(self):
        self.base_url = settings.EMBEDDING_BASE_URL.rstrip("/")
        self.model = settings.EMBEDDING_MODEL

    def embed(self, text: str) -> List[float]:
        """生成单个文本的向量"""
        return self._call_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量（ollama不支持批量，逐个调用）"""
        all_embeddings = []
        
        for i, text in enumerate(texts):
            logger.info(f"[Embed] 处理 {i+1}/{len(texts)}")
            embedding = self._call_embedding(text)
            all_embeddings.append(embedding)
        
        logger.info(f"[Embed] 批量嵌入完成, 总数={len(all_embeddings)}")
        return all_embeddings

    def _call_embedding(self, text: str) -> List[float]:
        """调用Ollama嵌入API"""
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model,
            "prompt": text,
        }
        
        try:
            response = httpx.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
            
            return data["embedding"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"嵌入API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"嵌入API调用失败: {e}")
            raise
