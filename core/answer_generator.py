"""模块5：答案生成与溯源拼接"""
from typing import List, Dict
from dataclasses import dataclass

from config.settings import settings
from core.llm_service import LLMService
from core.ocr_postprocessor import add_risk_warning
from utils.logger import logger


@dataclass
class AnswerResult:
    """答案结果"""
    answer: str
    sources: List[Dict]  # 引用源列表
    confidence: float  # 答案置信度


class AnswerGenerator:
    """LLM答案生成+溯源拼接"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate(self, query: str, retrieved_docs: List[Dict]) -> AnswerResult:
        """
        生成答案并拼接溯源信息
        query: 用户问题
        retrieved_docs: 检索到的文档列表
        """
        if not retrieved_docs:
            logger.warning("无检索结果，生成拒答")
            return AnswerResult(
                answer="根据文档内容，未找到与您问题相关的信息。",
                sources=[],
                confidence=0.0,
            )

        # 构建prompt
        system_prompt = self._build_system_prompt(retrieved_docs)
        user_prompt = self._build_user_prompt(query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            answer = self.llm_service.chat(messages)
            sources = self._format_sources(retrieved_docs)
            
            # 添加OCR风险提示
            answer = add_risk_warning(answer, retrieved_docs)
            
            # 简单置信度计算（可根据检索分数平均值调整）
            avg_score = sum([doc.get("score", 0) for doc in retrieved_docs]) / len(retrieved_docs)
            confidence = min(1.0, avg_score)

            logger.info(f"答案生成完成: {len(answer)}字符")
            return AnswerResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return AnswerResult(
                answer="抱歉，答案生成过程中出现错误。",
                sources=[],
                confidence=0.0,
            )

    def _build_system_prompt(self, retrieved_docs: List[Dict]) -> str:
        """构建系统提示词"""
        context_text = ""
        for i, doc in enumerate(retrieved_docs):
            page_num = doc["metadata"].get("page_num", "未知")
            content_type = doc["metadata"].get("content_type", "文本")
            content = doc["content"]
            
            # 对表格类型内容做特殊标记
            if content_type == "table":
                context_text += f"\n【来源{i+1} - P{page_num} - {content_type}】\n（表格数据）\n{content}\n"
            else:
                context_text += f"\n【来源{i+1} - P{page_num} - {content_type}】\n{content}\n"

        return f"""你是一个专业的文档问答助手。

你的唯一知识来源是下方【相关内容】中提供的文档片段。你没有其他任何知识。

回答规则：
- 如果【相关内容】中包含回答问题的信息，请引用原文片段回答
- 如果【相关内容】中不包含回答问题的信息，请直接回复：根据文档内容，未找到相关信息。
- 不要编造、猜测或联想文档外的内容
- 保持客观、准确、简洁的回答风格
- 对于表格数据问题，请仔细查找表格内容中的对应数值
- 对于条款号查询（如"3.4条款"），请在【相关内容】中搜索包含该条款号的内容
- 对于表格数值比较问题（如"最大"、"最小"），请仔细对比表格中各项目的数值大小

输出格式要求（必须严格遵守）：
1. 首先给出直接回答
2. 然后标注【标准条款号与页码】，明确指出信息来自哪个条款和页码
3. 最后标注【原文片段】，引用相关的原文内容

示例格式：
【回答】
（你的直接回答）

【标准条款号与页码】
（例如：第3.2条，第3页）

【原文片段】
（引用相关原文）

重要提醒：
- 你不知道任何文档外的知识
- 你只能基于下方【相关内容】中的内容作答
- 不要解释、不要补充、不要联想
- 必须按照上述格式输出
- 如果问题涉及表格数值比较，请逐一分析表格中的数据

【相关内容】：
{context_text}"""

    def _build_user_prompt(self, query: str) -> str:
        """构建用户提示词"""
        return query

    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """格式化引用源（按页码去重）"""
        sources = []
        seen_pages = set()
        
        for doc in retrieved_docs:
            page_num = doc["metadata"].get("page_num", "未知")
            
            # 按页码去重，每页只保留一个来源
            if page_num not in seen_pages:
                seen_pages.add(page_num)
                sources.append({
                    "page_num": page_num,
                    "content_type": doc["metadata"].get("content_type", "文本"),
                    "content_preview": doc["content"][:200],
                    "score": doc.get("score", 0.0),
                })
        
        return sources
