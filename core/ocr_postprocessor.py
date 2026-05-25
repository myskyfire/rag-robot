"""OCR后处理模块：文本修复、置信度分级、残缺检测"""
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from utils.logger import logger


@dataclass
class OCRTextSegment:
    """OCR文本片段"""
    text: str
    confidence: float
    is_complete: bool = True  # 是否完整
    risk_level: str = "low"  # 风险等级: low/medium/high
    page_num: int = 0


# 行业纠错字典（通用符号+形近字）
CORRECTION_DICT = {
    # 符号纠错
    '％': '%',
    '∶': ':',
    '·': '·',
    '一': '—',  # 破折号
    '～': '~',
    '×': '×',
    '÷': '÷',
    # 形近字纠错（常见OCR错误）
    'O型': '0型',
    'l': '1',
    'Ｉ': '1',
    'Ｏ': '0',
}

# 单位标准化映射
UNIT_STANDARD_MAP = {
    'mm': 'mm',
    '毫米': 'mm',
    'cm': 'cm',
    '厘米': 'cm',
    'm': 'm',
    '米': 'm',
    'kg': 'kg',
    '千克': 'kg',
    '公斤': 'kg',
    'g': 'g',
    '克': 'g',
}

# 截断检测关键词（句末出现这些词通常表示截断）
TRUNCATION_INDICATORS = [
    '的', '了', '在', '和', '与', '或', '但', '而', '及',
    '进行', '按照', '根据', '对于', '关于', '通过',
    '之比', '大于', '小于', '等于',
]


def postprocess_ocr_text(segments: List[Dict], page_num: int = 0) -> List[OCRTextSegment]:
    """
    OCR文本后处理：断行合并、纠错、标准化、残缺检测、置信度分级
    segments: OCR原始结果列表，每项包含{text, confidence, box}
    page_num: 页码
    返回: 处理后的文本片段列表
    """
    if not segments:
        return []

    # 1. 断行合并
    merged = _merge_broken_lines(segments)

    # 2. 文本纠错+标准化
    corrected = [_correct_and_standardize(seg) for seg in merged]

    # 3. 残缺检测
    for seg in corrected:
        seg.is_complete = _detect_truncation(seg.text)
        if not seg.is_complete:
            logger.debug(f"检测到残缺文本: {seg.text[:30]}...")

    # 4. 置信度分级
    for seg in corrected:
        seg.risk_level = _classify_confidence(seg.confidence)

    # 5. 设置页码
    for seg in corrected:
        seg.page_num = page_num

    return corrected


def _merge_broken_lines(segments: List[Dict]) -> List[Dict]:
    """
    合并非正常断行拆分的语句
    基于Y坐标相近且文本语义连续判断
    """
    if not segments:
        return []

    # 按Y坐标排序
    sorted_segs = sorted(segments, key=lambda x: _get_y_center(x.get('box', [])))

    merged = []
    current = sorted_segs[0].copy()
    current_text = current.get('text', '').strip()
    current_conf = current.get('confidence', 1.0)

    for i in range(1, len(sorted_segs)):
        next_seg = sorted_segs[i]
        next_text = next_seg.get('text', '').strip()
        next_conf = next_seg.get('confidence', 1.0)
        next_box = next_seg.get('box', [])

        # 判断是否应该合并
        should_merge = _should_merge_lines(current_text, next_text, current.get('box', []), next_box)

        if should_merge:
            # 合并文本
            current_text = current_text + next_text
            # 平均置信度
            current_conf = (current_conf + next_conf) / 2
            current['text'] = current_text
            current['confidence'] = current_conf
        else:
            # 保存当前，开始新的
            merged.append({
                'text': current_text,
                'confidence': current_conf,
                'box': current.get('box', []),
            })
            current = next_seg.copy()
            current_text = next_text
            current_conf = next_conf

    # 添加最后一个
    merged.append({
        'text': current_text,
        'confidence': current_conf,
        'box': current.get('box', []),
    })

    return merged


def _get_y_center(box: List) -> float:
    """获取边界框Y中心坐标"""
    if not box or len(box) < 4:
        return 0.0
    y_coords = [point[1] for point in box]
    return sum(y_coords) / len(y_coords)


def _should_merge_lines(text1: str, text2: str, box1: List, box2: List) -> bool:
    """判断两行是否应该合并"""
    if not text1 or not text2:
        return False

    # 规则1: Y坐标相近（同一行被拆分）
    y1 = _get_y_center(box1)
    y2 = _get_y_center(box2)
    y_diff = abs(y2 - y1)

    # 如果Y坐标差异很小，可能是同一行被拆分
    if y_diff < 10:
        return True

    # 规则2: 前一行以逗号、分号、冒号结尾，后一行继续
    if text1.endswith(('，', '；', '：', ',')):
        return True

    # 规则3: 前一行以"的"、"了"等虚词结尾，后一行继续
    if text1.endswith(('的', '了', '在', '和', '与')):
        return True

    # 规则4: 前一行以介词/连词结尾，语义不完整
    if text1.endswith(('除', '按', '据', '以', '从', '对', '为')):
        return True

    return False


def _correct_and_standardize(segment: Dict) -> OCRTextSegment:
    """
    文本纠错+标准化
    包括：形近字纠错、符号纠错、单位标准化、格式统一
    """
    text = segment.get('text', '').strip()
    confidence = segment.get('confidence', 1.0)

    # 1. 形近字+符号纠错
    for wrong, correct in CORRECTION_DICT.items():
        text = text.replace(wrong, correct)

    # 2. 单位标准化
    text = _standardize_units(text)

    # 3. 小数点标准化（全角转半角）
    text = text.replace('．', '.')

    # 4. 比值格式统一（如"1：100" -> "1:100"）
    text = re.sub(r'(\d+)\s*：\s*(\d+)', r'\1:\2', text)

    # 5. 去除多余空格（中文间空格）
    text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', text)

    return OCRTextSegment(
        text=text,
        confidence=confidence,
    )


def _standardize_units(text: str) -> str:
    """统一标准化单位"""
    for chinese_unit, standard_unit in UNIT_STANDARD_MAP.items():
        if chinese_unit != standard_unit:
            text = text.replace(chinese_unit, standard_unit)
    return text


def _detect_truncation(text: str) -> bool:
    """
    检测文本是否被截断
    返回: True表示完整，False表示残缺
    """
    if not text:
        return False

    # 规则1: 以句号、问号、叹号、分号结尾 -> 完整
    if text.endswith(('。', '？', '！', '；', '.', '?', '!', ';')):
        return True

    # 规则2: 以截断指示词结尾 -> 可能残缺（优先检查多字词）
    multi_word_indicators = ['之比', '大于', '小于', '等于']
    if any(text.endswith(indicator) for indicator in multi_word_indicators):
        return False

    # 规则3: 以单字虚词结尾 -> 可能残缺
    single_word_indicators = ['的', '了', '在', '和', '与', '或', '但', '而', '及']
    if any(text.endswith(indicator) for indicator in single_word_indicators):
        return False

    # 规则4: 以介词/动词结尾 -> 可能残缺
    verb_indicators = ['进行', '按照', '根据', '对于', '关于', '通过', '除', '按', '据', '以', '从', '对', '为']
    if any(text.endswith(indicator) for indicator in verb_indicators):
        return False

    # 规则5: 文本过短（<5字符）且无标点结尾 -> 可能残缺
    if len(text) < 5 and not re.search(r'[。！？；.!?;]', text):
        return False

    # 规则6: 包含"..."或"……" -> 明确残缺
    if '...' in text or '……' in text:
        return False

    # 默认认为完整
    return True


def _classify_confidence(confidence: float) -> str:
    """
    置信度分级
    high: >= 0.9
    medium: 0.7 ~ 0.9
    low: < 0.7
    """
    if confidence >= 0.9:
        return "low"  # 低风险（高置信）
    elif confidence >= 0.7:
        return "medium"
    else:
        return "high"  # 高风险（低置信）


def process_table_data(rows: List[List[str]], confidence: float = 1.0) -> Tuple[List[List[str]], Dict]:
    """
    表格数据容错处理
    rows: 原始表格行数据
    confidence: 表格整体置信度
    返回: (处理后的表格数据, 处理报告)
    """
    report = {
        'filtered_empty_cells': 0,
        'corrected_values': 0,
        'aligned_rows': 0,
        'risk_level': 'low',
        'header_detected': False,
    }

    if not rows:
        return rows, report

    # 1. 过滤空白单元格
    cleaned_rows = []
    for row in rows:
        cleaned_row = [cell.strip() for cell in row if cell.strip()]
        if cleaned_row:  # 跳过全空行
            cleaned_rows.append(cleaned_row)
        else:
            report['filtered_empty_cells'] += 1

    if not cleaned_rows:
        return [], report

    # 2. 表头识别与列数推断
    header_row = cleaned_rows[0]
    max_cols = max(len(row) for row in cleaned_rows)
    report['header_detected'] = True

    # 3. 列对齐（确保所有行列数一致）
    aligned_rows = _align_table_rows(cleaned_rows, max_cols)
    report['aligned_rows'] = sum(1 for r in cleaned_rows if len(r) != max_cols)

    # 4. 数值校验与纠错（逐单元格处理）
    corrected_rows = []
    for row_idx, row in enumerate(aligned_rows):
        corrected_row = []
        for col_idx, cell in enumerate(row):
            corrected_cell, is_corrected = _correct_table_cell_with_validation(
                cell, row_idx, col_idx, header_row, aligned_rows
            )
            corrected_row.append(corrected_cell)
            if is_corrected:
                report['corrected_values'] += 1
        corrected_rows.append(corrected_row)

    # 5. 置信度分级
    if confidence < 0.7:
        report['risk_level'] = 'high'
    elif confidence < 0.9:
        report['risk_level'] = 'medium'

    return corrected_rows, report


def _correct_table_cell(cell: str) -> str:
    """表格单元格纠错"""
    # 符号纠错
    for wrong, correct in CORRECTION_DICT.items():
        cell = cell.replace(wrong, correct)

    # 数值格式标准化
    # 去除"一"等OCR错误（在数值位置）
    if re.match(r'^[\d.]+$', cell):
        cell = cell.replace('一', '1')

    # 全角数字转半角
    fullwidth_digits = '０１２３４５６７８９'
    for i, fw in enumerate(fullwidth_digits):
        cell = cell.replace(fw, str(i))

    return cell


def _align_table_rows(rows: List[List[str]], max_cols: int) -> List[List[str]]:
    """规整错位行列数据（确保所有行列数一致）"""
    if not rows:
        return rows

    aligned = []
    for row in rows:
        if len(row) == max_cols:
            aligned.append(row)
        elif len(row) < max_cols:
            # 补齐空单元格
            aligned.append(row + [''] * (max_cols - len(row)))
        else:
            # 列数过多，尝试合并多余列
            aligned.append(row[:max_cols])

    return aligned


def _correct_table_cell_with_validation(
    cell: str, row_idx: int, col_idx: int, header_row: List[str], all_rows: List[List[str]]
) -> Tuple[str, bool]:
    """
    表格单元格纠错与校验
    返回: (修正后的单元格内容, 是否被修正)
    """
    original_cell = cell
    is_corrected = False

    # 1. 基础纠错（符号、全角转半角）
    cell = _correct_table_cell(cell)
    if cell != original_cell:
        is_corrected = True

    # 2. 数值合理性校验
    if _is_numeric(cell):
        corrected_cell, was_corrected = _validate_numeric_value(cell, row_idx, col_idx, header_row, all_rows)
        if was_corrected:
            cell = corrected_cell
            is_corrected = True

    return cell, is_corrected


def _is_numeric(text: str) -> bool:
    """判断文本是否为数值"""
    if not text:
        return False
    # 匹配数字（支持小数、负数、百分比）
    return bool(re.match(r'^-?\d+\.?\d*%?$', text))


def _validate_numeric_value(
    cell: str, row_idx: int, col_idx: int, header_row: List[str], all_rows: List[List[str]]
) -> Tuple[str, bool]:
    """
    数值合理性校验
    基于同列其他数值范围判断当前值是否合理
    """
    try:
        # 提取数值（去除百分号）
        value = float(cell.replace('%', ''))
    except ValueError:
        return cell, False

    # 收集同列其他数值
    column_values = []
    for r_idx, row in enumerate(all_rows):
        if r_idx != row_idx and col_idx < len(row):
            other_cell = row[col_idx]
            if _is_numeric(other_cell):
                try:
                    other_value = float(other_cell.replace('%', ''))
                    column_values.append(other_value)
                except ValueError:
                    pass

    if not column_values:
        return cell, False

    # 计算同列数值的统计范围
    min_val = min(column_values)
    max_val = max(column_values)
    mean_val = sum(column_values) / len(column_values)

    # 异常值检测（超出均值±3倍标准差）
    if len(column_values) >= 3:
        std_dev = (sum((v - mean_val) ** 2 for v in column_values) / len(column_values)) ** 0.5
        if std_dev > 0 and abs(value - mean_val) > 3 * std_dev:
            # 异常值，标记为可能错误（但不自动修正，仅返回原始值）
            logger.debug(f"表格数值异常: 行{row_idx}列{col_idx} 值={value}, 均值={mean_val:.2f}, 标准差={std_dev:.2f}")
            return cell, False

    return cell, False


def add_risk_warning(answer: str, retrieved_docs: List[Dict]) -> str:
    """
    问答联动风险提示
    当答题素材取自残缺文本、低置信识别内容时，在答案中标注风险
    answer: LLM生成的答案
    retrieved_docs: 检索到的文档片段
    返回: 添加风险提示后的答案
    """
    if not retrieved_docs:
        return answer

    risks = []

    for doc in retrieved_docs:
        metadata = doc.get('metadata', {})
        content_type = metadata.get('content_type', '')
        confidence = metadata.get('confidence', 1.0)

        # 检查置信度
        if confidence < 0.7:
            risks.append(f"低置信度内容（置信度: {confidence:.2f}）")

        # 检查是否标注残缺
        if metadata.get('is_complete') is False:
            risks.append("残缺文本片段")

    if risks:
        risk_text = "；".join(set(risks))
        warning = f"\n\n⚠ 风险提示：答案基于{risk_text}，可能存在识别偏差，请谨慎参考。"
        answer += warning

    return answer
