"""输出4个页面OCR识别结果到单独文件夹"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import fitz  # PyMuPDF
from config.settings import settings
from core.content_parser import ContentParser
from core.document_preprocessor import DocumentPreprocessor
from utils.logger import logger

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ocr_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试PDF路径
PDF_PATH = os.path.join(settings.DATA_DIR, "GBT 1568-2008 键 技术条件.pdf")

def export_ocr_results():
    """导出4个页面的OCR识别结果"""
    print("=" * 60)
    print("OCR识别结果导出")
    print("=" * 60)
    
    if not os.path.exists(PDF_PATH):
        print(f"错误: 找不到PDF文件 {PDF_PATH}")
        return
    
    # 1. 预处理PDF
    print("\n正在预处理PDF...")
    preprocessor = DocumentPreprocessor(PDF_PATH)
    pdf_doc = preprocessor.process()
    
    # 2. 解析内容
    print("正在解析内容...")
    parser = ContentParser()
    blocks = parser.parse_document(pdf_doc)
    
    # 3. 选择前4个扫描页
    scanned_pages = [page for page in pdf_doc.pages if page.is_scanned][:4]
    
    if not scanned_pages:
        print("未找到扫描页")
        return
    
    print(f"\n找到 {len(scanned_pages)} 个扫描页，开始导出OCR结果...")
    
    for page in scanned_pages:
        page_num = page.page_num
        print(f"\n处理第 {page_num} 页...")
        
        # 获取该页的文本块
        page_blocks = [b for b in blocks if b.page_num == page_num]
        
        # 构建输出数据
        output_data = {
            "page_num": page_num,
            "is_scanned": True,
            "text_blocks": [],
            "tables": [],
        }
        
        for block in page_blocks:
            if block.content_type == "table":
                output_data["tables"].append({
                    "content": block.content,
                    "confidence": block.confidence,
                    "is_complete": block.is_complete,
                    "risk_level": block.risk_level,
                })
            else:
                output_data["text_blocks"].append({
                    "content": block.content,
                    "content_type": block.content_type,
                    "confidence": block.confidence,
                    "is_complete": block.is_complete,
                    "risk_level": block.risk_level,
                })
        
        # 导出为JSON
        json_path = os.path.join(OUTPUT_DIR, f"page_{page_num}_ocr_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  JSON结果: {json_path}")
        
        # 导出为TXT（可读格式）
        txt_path = os.path.join(OUTPUT_DIR, f"page_{page_num}_ocr_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"第 {page_num} 页 OCR识别结果\n")
            f.write("=" * 60 + "\n\n")
            
            # 文本块
            for i, tb in enumerate(output_data["text_blocks"], 1):
                f.write(f"【文本块 {i}】类型: {tb['content_type']}\n")
                f.write(f"置信度: {tb['confidence']:.3f}\n")
                f.write(f"完整性: {'完整' if tb['is_complete'] else '残缺'}\n")
                f.write(f"风险等级: {tb['risk_level']}\n")
                f.write(f"内容:\n{tb['content']}\n")
                f.write("-" * 40 + "\n\n")
            
            # 表格
            for i, table in enumerate(output_data["tables"], 1):
                f.write(f"【表格 {i}】\n")
                f.write(f"置信度: {table['confidence']:.3f}\n")
                f.write(f"完整性: {'完整' if table['is_complete'] else '残缺'}\n")
                f.write(f"风险等级: {table['risk_level']}\n")
                f.write(f"内容:\n{table['content']}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"  TXT结果: {txt_path}")
    
    print(f"\n{'='*60}")
    print(f"OCR结果导出完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    export_ocr_results()
