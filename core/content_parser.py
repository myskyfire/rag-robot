"""模块2：文本+条款+表格结构化解析"""
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# 必须在 import paddle/paddleocr 之前设置！
os.environ["FLAGS_use_mkldnn"] = "false"
os.environ["FLAGS_use_onednn"] = "false"
os.environ["FLAGS_use_mkldnn_bf16"] = "false"

from config.settings import settings
from utils.logger import logger
from utils.exceptions import OCRParseError
from core.ocr_postprocessor import (
    postprocess_ocr_text,
    process_table_data,
    OCRTextSegment,
)
from core.multimodal_fallback import MultimodalFallbackService

# PaddleOCR 为可选依赖
try:
    import paddle
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    logger.warning("PaddleOCR 未安装，扫描PDF页面将无法进行OCR识别")


@dataclass
class TableData:
    """表格数据"""
    page_num: int
    rows: List[List[str]]  # 二维表格数据
    confidence: float  # OCR置信度
    raw_text: str = ""  # 原始文本


@dataclass
class TextBlock:
    """文本块"""
    page_num: int
    content: str
    content_type: str  # "text" / "clause" / "table"
    clause_level: int = 0  # 条款层级（0表示非条款）
    confidence: float = 1.0  # 识别置信度
    is_complete: bool = True  # 是否完整（残缺检测标记）
    risk_level: str = "low"  # 风险等级: low/medium/high


class ContentParser:
    """多类型内容结构化解析"""

    def __init__(self):
        self._ocr = None
        self._multimodal_service = MultimodalFallbackService()

    def _get_ocr(self):
        """懒加载OCR模型"""
        if not HAS_PADDLEOCR:
            raise OCRParseError("PaddleOCR 未安装，无法进行OCR识别。请安装 paddlepaddle 和 paddleocr。")
        
        if self._ocr is None:
            logger.info("初始化PaddleOCR模型...")
            # PaddleOCR 3.x 版本API变化，只需传lang
            self._ocr = PaddleOCR(lang="ch")
            logger.info("PaddleOCR初始化完成")
        return self._ocr

    def parse_document(self, pdf_doc) -> List[TextBlock]:
        """
        解析整个PDF文档
        pdf_doc: PDFDocument对象
        """
        all_blocks = []

        for page in pdf_doc.pages:
            if page.is_scanned:
                # 扫描页：OCR识别
                blocks = self._parse_scanned_page(page)
            else:
                # 原生页：直接提取
                blocks = self._parse_native_page(page)

            all_blocks.extend(blocks)

        logger.info(f"文档解析完成，共提取{len(all_blocks)}个文本块")
        return all_blocks

    def _parse_native_page(self, page) -> List[TextBlock]:
        """解析原生PDF页面"""
        blocks = []

        # 1. 提取正文
        text = page.text.strip()
        if text:
            text_blocks = self._extract_text_blocks(text, page.page_num)
            blocks.extend(text_blocks)

        # 2. 提取表格
        tables = self._extract_tables_native(page)
        blocks.extend(tables)

        return blocks

    def _parse_scanned_page(self, page) -> List[TextBlock]:
        """解析扫描PDF页面（OCR）"""
        blocks = []

        if not page.image_path:
            logger.warning(f"第{page.page_num}页为扫描页但无图片路径")
            return blocks

        if not HAS_PADDLEOCR:
            logger.warning(f"第{page.page_num}页为扫描页，但PaddleOCR未安装，跳过OCR识别")
            return blocks

        try:
            ocr = self._get_ocr()
            # PaddleOCR 3.x API变化，直接传图片路径即可
            result = ocr.ocr(page.image_path)

            if not result or not result[0]:
                logger.warning(f"第{page.page_num}页OCR识别结果为空")
                return blocks

            # 构建OCR原始结果（带位置信息）
            raw_segments = []
            for line in result[0]:
                if line and len(line) >= 2:
                    box = line[0]
                    text_info = line[1]
                    raw_segments.append({
                        'text': text_info[0],
                        'confidence': text_info[1],
                        'box': box,
                    })

            # OCR后处理：断行合并、纠错、标准化、残缺检测、置信度分级
            processed_segments = postprocess_ocr_text(raw_segments, page.page_num)

            # 按风险等级分组
            low_risk = [s for s in processed_segments if s.risk_level == "low"]
            medium_risk = [s for s in processed_segments if s.risk_level == "medium"]
            high_risk = [s for s in processed_segments if s.risk_level == "high"]

            # 合并文本
            full_text = "\n".join([s.text for s in processed_segments])

            # 计算平均置信度
            avg_confidence = sum(s.confidence for s in processed_segments) / len(processed_segments) if processed_segments else 0

            # 多模态兜底识别（当OCR置信度过低时）
            if self._multimodal_service.should_fallback(avg_confidence):
                logger.info(
                    f"第{page.page_num}页OCR置信度较低({avg_confidence:.3f})，尝试多模态模型兜底识别"
                )
                multimodal_result = self._multimodal_service.recognize_page(page.image_path)
                if multimodal_result:
                    logger.info(
                        f"第{page.page_num}页多模态兜底识别成功，置信度: {multimodal_result.get('confidence', 0):.3f}"
                    )
                    # 使用多模态结果替换OCR结果
                    return self._build_blocks_from_multimodal(multimodal_result, page.page_num)

            # 判断整体完整性
            is_complete = all(s.is_complete for s in processed_segments)

            # 确定整体风险等级
            if high_risk:
                overall_risk = "high"
            elif medium_risk:
                overall_risk = "medium"
            else:
                overall_risk = "low"

            # 文本分块（携带后处理元数据）
            text_blocks = self._extract_text_blocks(
                full_text, page.page_num, avg_confidence, is_complete, overall_risk
            )
            blocks.extend(text_blocks)

            # 表格OCR识别（带容错处理）
            tables = self._extract_tables_ocr_with_postprocess(
                raw_segments, page.page_num, avg_confidence
            )
            blocks.extend(tables)

            logger.info(
                f"第{page.page_num}页OCR后处理完成: "
                f"平均置信度={avg_confidence:.3f}, "
                f"完整性={'完整' if is_complete else '残缺'}, "
                f"风险等级={overall_risk}"
            )

        except Exception as e:
            logger.error(f"第{page.page_num}页OCR识别异常: {e}")
            raise OCRParseError(f"OCR识别失败: {e}")

        return blocks

    def _build_blocks_from_multimodal(self, multimodal_result: Dict, page_num: int) -> List[TextBlock]:
        """
        从多模态模型结果构建TextBlock
        multimodal_result: 多模态模型识别结果
        page_num: 页码
        """
        blocks = []
        confidence = multimodal_result.get("confidence", 0.7)
        risk_level = "high" if confidence < 0.7 else ("medium" if confidence < 0.9 else "low")

        # 1. 处理文本内容
        text = multimodal_result.get("text", "").strip()
        if text:
            text_blocks = self._extract_text_blocks(
                text, page_num, confidence, is_complete=True, risk_level=risk_level
            )
            blocks.extend(text_blocks)

        # 2. 处理表格内容
        tables = multimodal_result.get("tables", [])
        for table_data in tables:
            if isinstance(table_data, dict) and "rows" in table_data:
                rows = table_data["rows"]
                header = table_data.get("header", [])
                if header:
                    rows = [header] + rows

                if rows and len(rows) >= 2:
                    # 表格容错处理
                    corrected_rows, table_report = process_table_data(rows, confidence)

                    table_obj = TableData(
                        page_num=page_num,
                        rows=corrected_rows,
                        confidence=confidence,
                    )
                    blocks.append(TextBlock(
                        page_num=page_num,
                        content=self._table_to_text(table_obj),
                        content_type="table",
                        confidence=confidence,
                        is_complete=True,
                        risk_level=table_report.get("risk_level", risk_level),
                    ))

        return blocks

    def _extract_text_blocks(self, text: str, page_num: int, confidence: float = 1.0, is_complete: bool = True, risk_level: str = "low") -> List[TextBlock]:
        """提取文本块（区分正文和条款）"""
        blocks = []
        lines = text.split("\n")

        current_clause = []
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测条款编号（如 "第一条"、"3.2"、"（一）"）
            clause_match = self._match_clause_pattern(line)
            if clause_match:
                # 保存之前的文本块
                if current_text:
                    blocks.append(TextBlock(
                        page_num=page_num,
                        content="\n".join(current_text),
                        content_type="text",
                        confidence=confidence,
                        is_complete=is_complete,
                        risk_level=risk_level,
                    ))
                    current_text = []

                current_clause.append(line)
            else:
                # 保存之前的条款块
                if current_clause:
                    blocks.append(TextBlock(
                        page_num=page_num,
                        content="\n".join(current_clause),
                        content_type="clause",
                        clause_level=1,
                        confidence=confidence,
                        is_complete=is_complete,
                        risk_level=risk_level,
                    ))
                    current_clause = []

                current_text.append(line)

        # 保存剩余内容
        if current_clause:
            blocks.append(TextBlock(
                page_num=page_num,
                content="\n".join(current_clause),
                content_type="clause",
                clause_level=1,
                confidence=confidence,
                is_complete=is_complete,
                risk_level=risk_level,
            ))
        if current_text:
            blocks.append(TextBlock(
                page_num=page_num,
                content="\n".join(current_text),
                content_type="text",
                confidence=confidence,
                is_complete=is_complete,
                risk_level=risk_level,
            ))

        return blocks

    def _match_clause_pattern(self, line: str) -> Optional[str]:
        """匹配条款编号模式"""
        patterns = [
            r"^第[一二三四五六七八九十百]+[条章节]",  # 第一条、第一章
            r"^\d+\.\d+",  # 3.2、1.1.1
            r"^[（(][一二三四五六七八九十]+[）)]",  # （一）、（二）
            r"^[A-Z]\.",  # A.、B.
            r"^\d+[、.．]",  # 1、2. 3．
        ]

        for pattern in patterns:
            if re.match(pattern, line):
                return line
        return None

    def _extract_tables_native(self, page) -> List[TextBlock]:
        """原生PDF表格提取（简化版）"""
        # PyMuPDF的表格提取
        tables = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                rows = tab.extract()
                if rows:
                    table_data = TableData(
                        page_num=page.page_num,
                        rows=rows,
                        confidence=1.0,
                    )
                    tables.append(TextBlock(
                        page_num=page.page_num,
                        content=self._table_to_text(table_data),
                        content_type="table",
                        confidence=1.0,
                    ))
        except Exception as e:
            logger.debug(f"原生PDF表格提取失败: {e}")

        return tables

    def _extract_tables_ocr_with_postprocess(self, raw_segments: List[Dict], page_num: int, confidence: float) -> List[TextBlock]:
        """OCR表格识别（带后处理容错）"""
        tables = []

        if not raw_segments:
            return tables

        try:
            # 获取所有文本行的位置信息
            lines_with_pos = []
            for seg in raw_segments:
                box = seg.get('box', [])
                if box and len(box) >= 4:
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x_center = sum(x_coords) / 4
                    y_min = min(y_coords)

                    lines_with_pos.append({
                        'text': seg['text'],
                        'x_center': x_center,
                        'y_min': y_min,
                        'confidence': seg.get('confidence', 1.0),
                        'box': box,
                    })

            if not lines_with_pos:
                return tables

            # 按Y坐标排序
            lines_with_pos.sort(key=lambda x: x['y_min'])

            # 检测表格行
            table_rows = self._detect_table_rows(lines_with_pos)

            # 如果检测到表格且有多行
            if table_rows and len(table_rows) >= 2:
                max_cols = max(len(row) for row in table_rows)
                if max_cols >= 2:
                    # 表格容错处理
                    corrected_rows, table_report = process_table_data(table_rows, confidence)

                    table_data = TableData(
                        page_num=page_num,
                        rows=corrected_rows,
                        confidence=confidence,
                    )

                    # 确定风险等级
                    risk_level = table_report.get('risk_level', 'low')

                    tables.append(TextBlock(
                        page_num=page_num,
                        content=self._table_to_text(table_data),
                        content_type="table",
                        confidence=confidence,
                        is_complete=True,
                        risk_level=risk_level,
                    ))

                    logger.info(
                        f"第{page_num}页检测到表格: {len(corrected_rows)}行, {max_cols}列, "
                        f"容错处理: 过滤空单元格={table_report['filtered_empty_cells']}, "
                        f"纠错={table_report['corrected_values']}, "
                        f"风险等级={risk_level}"
                    )

        except Exception as e:
            logger.debug(f"OCR表格识别失败: {e}")

        return tables

    def _extract_tables_ocr(self, image_path: str, page_num: int, confidence: float) -> List[TextBlock]:
        """OCR表格识别（基于文本行位置检测）- 保留兼容"""
        tables = []

        if not HAS_PADDLEOCR:
            return tables

        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image_path, cls=False, rec=True)

            if not result or not result[0]:
                return tables

            # 获取所有文本行的位置信息
            lines_with_pos = []
            for line in result[0]:
                if line and len(line) >= 2:
                    box = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    conf = text_info[1]

                    # 计算边界框中心点
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x_center = sum(x_coords) / 4
                    y_center = sum(y_coords) / 4
                    y_min = min(y_coords)

                    lines_with_pos.append({
                        'text': text,
                        'x_center': x_center,
                        'y_center': y_center,
                        'y_min': y_min,
                        'confidence': conf,
                    })

            if not lines_with_pos:
                return tables

            # 按Y坐标排序
            lines_with_pos.sort(key=lambda x: x['y_min'])

            # 检测表格行
            table_rows = self._detect_table_rows(lines_with_pos)

            # 如果检测到表格且有多行
            if table_rows and len(table_rows) >= 2:
                # 检查是否是真正的表格（至少2列）
                max_cols = max(len(row) for row in table_rows)
                if max_cols >= 2:
                    table_data = TableData(
                        page_num=page_num,
                        rows=table_rows,
                        confidence=confidence,
                    )
                    tables.append(TextBlock(
                        page_num=page_num,
                        content=self._table_to_text(table_data),
                        content_type="table",
                        confidence=confidence,
                    ))
                    logger.info(f"第{page_num}页检测到表格: {len(table_rows)}行, {max_cols}列")

        except Exception as e:
            logger.debug(f"OCR表格识别失败: {e}")

        return tables
    
    def _detect_table_rows(self, lines_with_pos: List[dict]) -> List[List[str]]:
        """基于位置信息检测表格行（优化版：智能Y阈值、X坐标聚类列对齐）"""
        if not lines_with_pos:
            return []
        
        # 1. 智能计算Y坐标阈值（基于行高统计）
        y_threshold = self._calculate_y_threshold(lines_with_pos)
        
        # 2. 按Y坐标分组（同一行的文本）
        rows = self._group_lines_by_y(lines_with_pos, y_threshold)
        
        # 3. 检测列位置（X坐标聚类）
        column_positions = self._detect_column_positions(rows)
        
        # 4. 列对齐处理
        aligned_rows = self._align_rows_to_columns(rows, column_positions)
        
        # 5. 过滤：至少要有2列才认为是表格
        table_rows = [row for row in aligned_rows if len(row) >= 2]
        
        return table_rows

    def _calculate_y_threshold(self, lines_with_pos: List[dict]) -> float:
        """智能计算Y坐标阈值（基于行高统计）"""
        if len(lines_with_pos) < 2:
            return 15.0
        
        # 计算相邻行的Y坐标差异
        y_diffs = []
        sorted_lines = sorted(lines_with_pos, key=lambda x: x['y_min'])
        for i in range(1, len(sorted_lines)):
            diff = sorted_lines[i]['y_min'] - sorted_lines[i-1]['y_min']
            if diff > 0:
                y_diffs.append(diff)
        
        if not y_diffs:
            return 15.0
        
        # 使用中位数作为阈值（避免异常值影响）
        y_diffs.sort()
        median_diff = y_diffs[len(y_diffs) // 2]
        
        # 阈值设为中位数的50%，确保同一行内文本能被正确分组
        threshold = max(median_diff * 0.5, 5.0)
        
        return threshold

    def _group_lines_by_y(self, lines_with_pos: List[dict], y_threshold: float) -> List[List[dict]]:
        """按Y坐标分组文本行"""
        if not lines_with_pos:
            return []
        
        # 按Y坐标排序
        sorted_lines = sorted(lines_with_pos, key=lambda x: x['y_min'])
        
        rows = []
        current_row = [sorted_lines[0]]
        current_y = sorted_lines[0]['y_min']
        
        for line in sorted_lines[1:]:
            if abs(line['y_min'] - current_y) > y_threshold:
                # 新行
                if current_row:
                    # 按X坐标排序
                    current_row.sort(key=lambda x: x['x_center'])
                    rows.append(current_row)
                current_row = [line]
                current_y = line['y_min']
            else:
                current_row.append(line)
        
        # 添加最后一行
        if current_row:
            current_row.sort(key=lambda x: x['x_center'])
            rows.append(current_row)
        
        return rows

    def _detect_column_positions(self, rows: List[List[dict]]) -> List[float]:
        """检测列位置（基于X坐标聚类）"""
        if not rows:
            return []
        
        # 收集所有X坐标
        all_x_centers = []
        for row in rows:
            for item in row:
                all_x_centers.append(item['x_center'])
        
        if not all_x_centers:
            return []
        
        # X坐标聚类（简单聚类：按间距分组）
        all_x_centers.sort()
        
        # 计算X坐标间距
        x_diffs = []
        for i in range(1, len(all_x_centers)):
            diff = all_x_centers[i] - all_x_centers[i-1]
            if diff > 10:  # 忽略同一单元格内的微小差异
                x_diffs.append((diff, all_x_centers[i-1], all_x_centers[i]))
        
        if not x_diffs:
            return [all_x_centers[0]]
        
        # 找到主要间距（出现频率最高的间距范围）
        # 使用简单的分桶方法
        bucket_size = 20  # 20像素为一个桶
        buckets = {}
        for diff, x1, x2 in x_diffs:
            bucket = int(diff // bucket_size)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append((diff, x1, x2))
        
        # 找到最大的桶（主要间距）
        if not buckets:
            return [all_x_centers[0]]
        
        main_bucket = max(buckets.keys(), key=lambda k: len(buckets[k]))
        main_diffs = buckets[main_bucket]
        
        # 基于主要间距推断列位置
        # 从第一个X坐标开始，按主要间距分组
        column_positions = []
        current_col_start = all_x_centers[0]
        column_positions.append(current_col_start)
        
        avg_main_diff = sum(d for d, _, _ in main_diffs) / len(main_diffs)
        
        for x in all_x_centers[1:]:
            # 如果距离上一列位置超过平均间距的60%，认为是新列
            if x - current_col_start > avg_main_diff * 0.6:
                # 检查是否已有接近的列位置
                is_new_col = True
                for col_pos in column_positions:
                    if abs(x - col_pos) < avg_main_diff * 0.3:
                        is_new_col = False
                        break
                
                if is_new_col:
                    column_positions.append(x)
                    current_col_start = x
        
        return sorted(column_positions)

    def _align_rows_to_columns(self, rows: List[List[dict]], column_positions: List[float]) -> List[List[str]]:
        """将行数据对齐到列"""
        if not rows or not column_positions:
            return [[item['text'] for item in row] for row in rows]
        
        aligned_rows = []
        
        for row in rows:
            # 初始化空列
            num_cols = len(column_positions)
            cells = [''] * num_cols
            
            for item in row:
                x_center = item['x_center']
                
                # 找到最近的列位置
                best_col = 0
                min_dist = float('inf')
                for col_idx, col_pos in enumerate(column_positions):
                    dist = abs(x_center - col_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_col = col_idx
                
                # 将文本放入对应列
                if cells[best_col]:
                    cells[best_col] += ' ' + item['text']
                else:
                    cells[best_col] = item['text']
            
            # 过滤空列（如果整行都是空的）
            if any(cell for cell in cells):
                aligned_rows.append(cells)
        
        return aligned_rows

    def _table_to_text(self, table_data: TableData) -> str:
        """将表格数据转为文本（增强版，包含表头信息）"""
        if not table_data.rows:
            return ""
        
        lines = []
        # 第一行通常是表头
        header = table_data.rows[0] if table_data.rows else []
        
        for row_idx, row in enumerate(table_data.rows):
            if row_idx == 0:
                # 表头行
                lines.append("【表头】" + " | ".join(row))
            else:
                # 数据行，尝试与表头关联
                row_text = " | ".join(row)
                # 如果行数据与表头列数相同，添加行号
                if len(row) == len(header):
                    lines.append(f"【第{row_idx}行】{row_text}")
                else:
                    lines.append(row_text)
        
        return "\n".join(lines)
