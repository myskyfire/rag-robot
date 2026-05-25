"""模块3：向量知识库构建（FAISS）"""
import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import faiss

from config.settings import settings
from core.chunker import Chunk
from utils.logger import logger
from utils.exceptions import VectorStoreError


@dataclass
class ChunkMetadata:
    """chunk元数据"""
    chunk_id: str
    content: str
    content_type: str
    page_num: int
    source_file: str = ""
    confidence: float = 1.0
    is_complete: bool = True  # 是否完整（残缺检测标记）
    risk_level: str = "low"  # 风险等级: low/medium/high


class VectorStore:
    """FAISS向量检索库"""

    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.persist_dir = settings.CACHE_DIR / "faiss_index"
        self.persist_dir.mkdir(exist_ok=True)

        self.index = None  # FAISS索引
        self.metadata_store: Dict[int, ChunkMetadata] = {}  # id -> metadata
        self.content_store: Dict[str, str] = {}  # chunk_id -> content
        self.dim = settings.EMBEDDING_DIM
        self._next_id = 0

    def build(self, chunks: List[Chunk], embeddings: List[List[float]], source_file: str = ""):
        """
        构建向量知识库
        chunks: 分块列表
        embeddings: 对应的向量列表
        source_file: 源文件名
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(f"chunks数量({len(chunks)})与embeddings数量({len(embeddings)})不匹配")

        if len(chunks) == 0:
            logger.warning(f"无有效chunks，跳过向量库构建: {source_file}")
            return

        logger.info(f"开始构建FAISS向量库: {self.index_name}, 共{len(chunks)}个chunks")

        # 转换为numpy数组
        vectors = np.array(embeddings, dtype=np.float32)

        # 创建或加载索引
        if self.index is None:
            base_index = faiss.IndexFlatIP(self.dim)  # 内积相似度（余弦）
            self.index = faiss.IndexIDMap(base_index)  # 支持自定义ID

        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(vectors)

        # 添加到索引
        ids = np.arange(self._next_id, self._next_id + len(vectors), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)

        # 存储元数据
        for chunk, emb_id in zip(chunks, ids):
            chunk_id = chunk.chunk_id or f"chunk_{uuid.uuid4().hex[:8]}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                content=chunk.content,
                content_type=chunk.content_type,
                page_num=chunk.page_num,
                source_file=source_file,
                confidence=chunk.metadata.get("confidence", 1.0),
                is_complete=chunk.metadata.get("is_complete", True),
                risk_level=chunk.metadata.get("risk_level", "low"),
            )
            self.metadata_store[int(emb_id)] = metadata
            self.content_store[chunk_id] = chunk.content

        self._next_id += len(vectors)
        logger.info(f"FAISS向量库更新完成: 总向量数={self.index.ntotal}")

    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """
        向量相似度检索
        返回: [{"id": str, "content": str, "score": float, "metadata": dict}]
        """
        if self.index is None or self.index.ntotal == 0:
            raise VectorStoreError("向量库为空，请先调用build()")

        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        # 限制top_k不超过总向量数
        top_k = min(top_k, self.index.ntotal)

        # 归一化查询向量
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            metadata = self.metadata_store.get(int(idx))
            if metadata:
                results.append({
                    "id": metadata.chunk_id,
                    "content": metadata.content,
                    "score": float(score),
                    "metadata": {
                        "content_type": metadata.content_type,
                        "page_num": metadata.page_num,
                        "source_file": metadata.source_file,
                        "confidence": metadata.confidence,
                        "is_complete": metadata.is_complete,
                        "risk_level": metadata.risk_level,
                    },
                })

        return results

    def search_by_keyword(self, query: str, top_k: int = None) -> List[Dict]:
        """
        关键词匹配检索（遍历匹配）
        返回: [{"id": str, "content": str, "score": float, "metadata": dict}]
        """
        if top_k is None:
            top_k = settings.KEYWORD_TOP_K

        results = []
        for idx, metadata in self.metadata_store.items():
            if query.lower() in metadata.content.lower():
                results.append({
                    "id": metadata.chunk_id,
                    "content": metadata.content,
                    "score": 0.8,
                    "metadata": {
                        "content_type": metadata.content_type,
                        "page_num": metadata.page_num,
                        "source_file": metadata.source_file,
                        "confidence": metadata.confidence,
                        "is_complete": metadata.is_complete,
                        "risk_level": metadata.risk_level,
                    },
                })

        # 按内容相关性排序（简单按匹配次数）
        results.sort(key=lambda x: x["content"].lower().count(query.lower()), reverse=True)
        return results[:top_k]

    def get_content_by_id(self, chunk_id: str) -> Optional[str]:
        """根据chunk_id获取原始内容"""
        return self.content_store.get(chunk_id)

    def save(self):
        """持久化索引和元数据"""
        if self.index is None:
            return

        index_path = self.persist_dir / f"{self.index_name}.index"
        meta_path = self.persist_dir / f"{self.index_name}.meta.json"

        # 保存FAISS索引
        faiss.write_index(self.index, str(index_path))

        # 保存元数据
        meta_dict = {
            "next_id": self._next_id,
            "metadata_store": {
                str(k): {
                    "chunk_id": v.chunk_id,
                    "content": v.content,
                    "content_type": v.content_type,
                    "page_num": v.page_num,
                    "source_file": v.source_file,
                    "confidence": v.confidence,
                }
                for k, v in self.metadata_store.items()
            },
            "content_store": self.content_store,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS索引已保存: {index_path}")

    def load(self):
        """加载持久化的索引和元数据"""
        index_path = self.persist_dir / f"{self.index_name}.index"
        meta_path = self.persist_dir / f"{self.index_name}.meta.json"

        if not index_path.exists() or not meta_path.exists():
            logger.info(f"未找到持久化索引: {self.index_name}")
            return False

        # 加载FAISS索引
        self.index = faiss.read_index(str(index_path))

        # 加载元数据
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        self._next_id = meta_dict["next_id"]
        self.metadata_store = {
            int(k): ChunkMetadata(**v)
            for k, v in meta_dict["metadata_store"].items()
        }
        self.content_store = meta_dict["content_store"]

        logger.info(f"FAISS索引已加载: {self.index_name}, 向量数={self.index.ntotal}")
        return True

    def reset(self):
        """重置索引"""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata_store.clear()
        self.content_store.clear()
        self._next_id = 0
        logger.info(f"FAISS索引已重置: {self.index_name}")
