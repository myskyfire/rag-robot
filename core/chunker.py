"""模块3：文档切块策略"""
import re
from typing import List
from dataclasses import dataclass

from config.settings import settings
from utils.logger import logger


@dataclass
class Chunk:
    """文本块"""
    chunk_id: str
    content: str
    content_type: str  # "text" / "clause" / "table"
    page_num: int
    metadata: dict  # 额外元数据


class TextChunker:
    """通用滑动窗口分块策略"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_blocks(self, blocks) -> List[Chunk]:
        """
        将解析后的文本块切分为chunks
        blocks: List[TextBlock]
        """
        all_chunks = []
        chunk_counter = 0

        for block in blocks:
            if block.content_type == "table":
                # 表格独立存储，不拆分
                chunk = Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    content=block.content,
                    content_type="table",
                    page_num=block.page_num,
                    metadata={
                        "confidence": block.confidence,
                        "is_complete": block.is_complete,
                        "risk_level": block.risk_level,
                    },
                )
                all_chunks.append(chunk)
                chunk_counter += 1
            elif block.content_type == "clause":
                # 条款保持完整，不拆分
                chunk = Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    content=block.content,
                    content_type="clause",
                    page_num=block.page_num,
                    metadata={
                        "clause_level": block.clause_level,
                        "confidence": block.confidence,
                        "is_complete": block.is_complete,
                        "risk_level": block.risk_level,
                    },
                )
                all_chunks.append(chunk)
                chunk_counter += 1
            else:
                # 普通文本：滑动窗口分块
                text_chunks = self._split_text(block.content, block.page_num, block.is_complete, block.risk_level, block.confidence)
                for tc in text_chunks:
                    tc.chunk_id = f"chunk_{chunk_counter}"
                    all_chunks.append(tc)
                    chunk_counter += 1

        logger.info(f"分块完成: 共{len(all_chunks)}个chunks")
        return all_chunks

    def _split_text(self, text: str, page_num: int, is_complete: bool = True, risk_level: str = "low", confidence: float = 1.0) -> List[Chunk]:
        """滑动窗口分块，保证语义完整"""
        if len(text) <= self.chunk_size:
            return [Chunk(
                chunk_id="",
                content=text,
                content_type="text",
                page_num=page_num,
                metadata={
                    "is_complete": is_complete,
                    "risk_level": risk_level,
                    "confidence": confidence,
                },
            )]

        chunks = []
        # 按句子边界分割
        sentences = re.split(r'([。！？；\n])', text)
        # 重新组合句子和标点
        segments = []
        for i in range(0, len(sentences) - 1, 2):
            seg = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if seg.strip():
                segments.append(seg.strip())

        current_chunk = ""
        for seg in segments:
            if len(current_chunk) + len(seg) > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        chunk_id="",
                        content=current_chunk,
                        content_type="text",
                        page_num=page_num,
                        metadata={
                            "is_complete": is_complete,
                            "risk_level": risk_level,
                            "confidence": confidence,
                        },
                    ))
                # 重叠部分
                overlap = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap + seg
            else:
                current_chunk += seg

        if current_chunk:
            chunks.append(Chunk(
                chunk_id="",
                content=current_chunk,
                content_type="text",
                page_num=page_num,
                metadata={
                    "is_complete": is_complete,
                    "risk_level": risk_level,
                    "confidence": confidence,
                },
            ))

        return chunks
