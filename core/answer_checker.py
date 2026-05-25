"""模块6：答案自检与边界异常风控"""
import re
from typing import List, Dict
from dataclasses import dataclass

from config.settings import settings
from utils.logger import logger


@dataclass
class CheckResult:
    """自检结果"""
    is_valid: bool  # 答案是否有效
    check_type: str  # 检查类型
    message: str  # 提示信息
    confidence: float  # 置信度


class AnswerChecker:
    """答案可靠性自检+边界异常处理"""

    def __init__(self):
        self.confidence_threshold = settings.SELF_CHECK_THRESHOLD

    def check_answer(self, answer: str, sources: List[Dict], query: str) -> List[CheckResult]:
        """
        完整答案自检流程
        返回检查结果列表
        """
        logger.info(f"开始答案自检: query='{query[:50]}...'")
        results = []

        # 1. 无依据校验
        no_source_result = self._check_no_source(answer, sources)
        if no_source_result:
            results.append(no_source_result)

        # 2. 幻觉校验
        hallucination_result = self._check_hallucination(answer, sources)
        if hallucination_result:
            results.append(hallucination_result)

        # 3. OCR风险校验
        ocr_risk_result = self._check_ocr_risk(sources)
        if ocr_risk_result:
            results.append(ocr_risk_result)

        # 4. 边界场景处理
        boundary_result = self._check_boundary(query, answer, sources)
        if boundary_result:
            results.append(boundary_result)

        logger.info(f"自检完成: {len(results)}项检查")
        return results

    def _check_no_source(self, answer: str, sources: List[Dict]) -> CheckResult:
        """无依据校验：无匹配文档内容时拒答"""
        if not sources:
            return CheckResult(
                is_valid=False,
                check_type="no_source",
                message="未找到相关文档内容，无法回答",
                confidence=0.0,
            )
        return None

    def _check_hallucination(self, answer: str, sources: List[Dict]) -> CheckResult:
        """幻觉校验：比对答案与原文语义一致性"""
        if not sources:
            return None

        # 提取答案中的关键实体
        answer_entities = self._extract_entities(answer)
        
        # 检查这些实体是否在来源中出现
        source_text = " ".join([src.get("content_preview", "") for src in sources])
        
        # 计算实体匹配率
        matched = 0
        total = len(answer_entities)
        
        # 过滤掉通用词汇
        generic_words = {"根据", "文档", "内容", "信息", "来源", "回答", "问题", "标准", "规定", "要求", "技术", "条件"}
        filtered_entities = [e for e in answer_entities if e not in generic_words and len(e) > 1]
        
        for entity in filtered_entities:
            if entity in source_text:
                matched += 1
        
        if len(filtered_entities) > 0:
            match_rate = matched / len(filtered_entities)
            # 提高阈值，降低误报
            if match_rate < 0.3:  # 从0.5降低到0.3
                return CheckResult(
                    is_valid=False,
                    check_type="hallucination",
                    message=f"答案可能包含文档外信息（实体匹配率: {match_rate:.1%}）",
                    confidence=match_rate,
                )
        
        return None

    def _check_ocr_risk(self, sources: List[Dict]) -> CheckResult:
        """OCR风险校验：低置信内容标注"""
        low_confidence_sources = []
        
        for src in sources:
            # 检查是否有低置信度标记
            if src.get("confidence", 1.0) < settings.OCR_CONFIDENCE_THRESHOLD:
                low_confidence_sources.append(src)
        
        if low_confidence_sources:
            return CheckResult(
                is_valid=True,
                check_type="ocr_risk",
                message=f"答案基于OCR识别内容，可能存在识别误差（{len(low_confidence_sources)}个低置信来源）",
                confidence=0.7,
            )
        
        return None

    def _check_boundary(self, query: str, answer: str, sources: List[Dict]) -> CheckResult:
        """边界场景处理"""
        # 检查是否为模糊提问
        if self._is_vague_query(query):
            return CheckResult(
                is_valid=True,
                check_type="vague_query",
                message="问题表述较为模糊，已尽力回答，建议提供更具体的问题",
                confidence=0.6,
            )
        
        # 检查是否为无关问题
        if not sources and self._is_irrelevant_query(query):
            return CheckResult(
                is_valid=False,
                check_type="irrelevant_query",
                message="该问题与文档内容无关",
                confidence=0.0,
            )
        
        return None

    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的关键实体"""
        entities = []
        
        # 提取中文词汇（2-4字）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        entities.extend(chinese_words)
        
        # 提取数字+单位
        numbers = re.findall(r'\d+\.?\d*\s*[%%万亿千百万]?.*', text)
        entities.extend(numbers)
        
        # 去重
        return list(set(entities))

    def _is_vague_query(self, query: str) -> bool:
        """判断是否为模糊提问"""
        vague_patterns = [
            r'这个.*什么',
            r'那个.*怎么样',
            r'说一下.*',
            r'介绍一下.*',
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, query):
                return True
        return False

    def _is_irrelevant_query(self, query: str) -> bool:
        """判断是否为无关问题"""
        # 简单规则：如果问题完全不包含文档中的关键词
        irrelevant_keywords = [
            "天气", "新闻", "股票", "体育", "娱乐",
            "今天", "明天", "现在", "几点",
        ]
        
        for kw in irrelevant_keywords:
            if kw in query.lower():
                return True
        return False

    def format_result(self, answer: str, sources: List[Dict], check_results: List[CheckResult]) -> str:
        """格式化最终输出结果"""
        output = f"【回答】\n{answer}\n\n"
        
        # 添加溯源信息（显示页码+内容摘要）
        if sources:
            output += "【引用来源】\n"
            for i, src in enumerate(sources, 1):
                page_num = src.get("page_num", "未知")
                content_preview = src.get("content_preview", "")
                content_type = src.get("content_type", "文本")
                
                # 截取前100字符作为摘要
                if content_preview:
                    preview = content_preview[:100].strip()
                    if len(content_preview) > 100:
                        preview += "..."
                    output += f"  {i}. 第{page_num}页 [{content_type}]\n     {preview}\n"
                else:
                    output += f"  {i}. 第{page_num}页 [{content_type}]\n"
        
        # 添加自检结果
        if check_results:
            output += "\n【自检结果】\n"
            for cr in check_results:
                if cr.check_type == "no_source":
                    output += f"  [警告] {cr.message}\n"
                elif cr.check_type == "hallucination":
                    output += f"  [警告] {cr.message}\n"
                elif cr.check_type == "ocr_risk":
                    output += f"  [警告] {cr.message}\n"
                elif cr.check_type == "vague_query":
                    output += f"  [警告] {cr.message}\n"
                elif cr.check_type == "irrelevant_query":
                    output += f"  [警告] {cr.message}\n"
        
        return output
