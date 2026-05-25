"""模块4：问题理解与多路检索召回"""
import re
from typing import List, Dict

from config.settings import settings
from core.vectorstore import VectorStore
from utils.logger import logger


class Retriever:
    """用户问题理解+多路内容召回"""

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, query_embedding: List[float]) -> List[Dict]:
        """
        双路召回策略
        query: 用户问题
        query_embedding: 问题的向量表示
        返回: 合并去重后的检索结果
        """
        logger.info(f"开始多路召回: query='{query[:50]}...'")

        # 1. 向量相似度检索
        dense_results = self._vector_search(query_embedding)
        logger.info(f"向量检索召回: {len(dense_results)}条")

        # 2. 关键词精准匹配
        keyword_results = self._keyword_search(query)
        logger.info(f"关键词检索召回: {len(keyword_results)}条")

        # 3. 合并去重
        merged = self._merge_results(dense_results, keyword_results)
        logger.info(f"合并去重后: {len(merged)}条")

        # 4. 表格查询智能补充
        if self._is_table_related_query(query):
            merged = self._supplement_table_chunks(merged)
            logger.info(f"表格查询补充后: {len(merged)}条")

        return merged

    def _vector_search(self, query_embedding: List[float]) -> List[Dict]:
        """向量相似度检索"""
        try:
            results = self.vectorstore.search(
                query_embedding=query_embedding,
                top_k=settings.RETRIEVAL_TOP_K,
            )
            return results
        except Exception as e:
            logger.error(f"向量检索异常: {e}")
            return []

    def _keyword_search(self, query: str) -> List[Dict]:
        """关键词精准匹配"""
        try:
            # 提取关键实体
            keywords = self._extract_keywords(query)
            all_results = []

            for kw in keywords:
                results = self.vectorstore.search_by_keyword(
                    query=kw,
                    top_k=settings.KEYWORD_TOP_K,
                )
                all_results.extend(results)

            return all_results
        except Exception as e:
            logger.error(f"关键词检索异常: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词（通用实体提取，不依赖硬编码词典）"""
        keywords = []

        # 1. 提取专有名词（中文连续2-6字词）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,6}', query)
        keywords.extend(chinese_words)

        # 2. 提取数字+单位（如"3.5%"、"100万元"）
        number_units = re.findall(r'\d+\.?\d*\s*[%%万亿千百万]?', query)
        keywords.extend(number_units)

        # 3. 提取条款引用（如"第X条"、"第X章"）
        clauses = re.findall(r'第[一二三四五六七八九十百\d]+[条章节]', query)
        keywords.extend(clauses)

        # 4. 提取"X.Y"格式的条款号（如"3.4"、"4.2"）
        clause_numbers = re.findall(r'\d+\.\d+', query)
        keywords.extend(clause_numbers)

        # 5. 提取大写缩写、字母数字组合（如"AQL"、"GB/T"）
        alpha_numeric = re.findall(r'[A-Za-z0-9/]+[\u4e00-\u9fa5]*', query)
        keywords.extend(alpha_numeric)

        # 过滤无意义虚词
        stop_words = {
            "多少", "什么", "规定", "按照", "请问", "是否", "有没有",
            "哪些", "哪个", "哪一项", "分别", "对应", "要求",
            "标准", "内容", "执行", "说明", "同时",
        }
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2]

        # 去重
        return list(set(keywords))

    def _merge_results(self, dense_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """合并去重，向量结果优先"""
        seen_ids = set()
        merged = []

        # 先加向量结果
        for r in dense_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        # 再加关键词结果
        for r in keyword_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        # 按score排序
        merged.sort(key=lambda x: x["score"], reverse=True)

        return merged[:settings.RETRIEVAL_TOP_K]

    def _is_table_related_query(self, query: str) -> bool:
        """判断查询是否涉及表格数据（通用规则，不依赖硬编码词典）"""
        table_indicators = [
            "数值", "数据", "表格", "AQL", "合格", "验收", "检查项目",
            "最大", "最小", "对比", "比较", "一致", "区别", "差异",
            "等级", "水平", "参数", "指标",
        ]
        return any(indicator in query for indicator in table_indicators)

    def _supplement_table_chunks(self, results: List[Dict]) -> List[Dict]:
        """
        对表格相关查询，智能补充表格类型chunk
        当检索结果中没有表格chunk时，尝试从向量库中补充
        """
        # 检查是否已有表格类型内容
        has_table = any(r["metadata"].get("content_type") == "table" for r in results)
        
        if not has_table:
            # 尝试通过关键词"表"来检索表格内容
            table_results = self.vectorstore.search_by_keyword(query="表", top_k=3)
            table_chunks = [r for r in table_results if r["metadata"].get("content_type") == "table"]
            
            if table_chunks:
                # 补充表格chunk到结果中
                seen_ids = {r["id"] for r in results}
                for tc in table_chunks:
                    if tc["id"] not in seen_ids:
                        results.append(tc)
                        seen_ids.add(tc["id"])
                        if len(results) >= settings.RETRIEVAL_TOP_K:
                            break
                
                # 重新按score排序
                results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
