"""模块0：多格式统一转PDF前置模块"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config.settings import settings
from utils.logger import logger
from utils.exceptions import PDFReadError


class FileConverter:
    """多格式文件统一转换为PDF"""

    def __init__(self):
        self.supported_formats = {
            # Office文档
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            # 图片
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
            # 文本
            '.txt',
            # 网页
            '.html', '.htm',
            # PDF（直接通过）
            '.pdf'
        }

    def convert_to_pdf(self, file_path: str) -> str:
        """
        统一转换各种格式为PDF
        返回转换后的PDF文件路径
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in self.supported_formats:
            raise PDFReadError(f"不支持的文件格式: {file_ext}")

        if file_ext == '.pdf':
            # PDF直接返回
            logger.info(f"文件已是PDF格式，直接使用: {file_path}")
            return file_path

        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']:
            # 图片转PDF
            return self._image_to_pdf(file_path)

        elif file_ext in ['.txt']:
            # 文本转PDF
            return self._text_to_pdf(file_path)

        elif file_ext in ['.html', '.htm']:
            # HTML转PDF
            return self._html_to_pdf(file_path)

        elif file_ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            # Office文档转PDF
            return self._office_to_pdf(file_path)

        else:
            raise PDFReadError(f"无法处理的文件格式: {file_ext}")

    def _image_to_pdf(self, image_path: str) -> str:
        """图片转PDF"""
        logger.info(f"正在将图片转换为PDF: {image_path}")
        
        # 使用PyMuPDF将图片转为PDF
        doc = fitz.open()
        img = fitz.open(image_path)
        rect = img.rect
        pdf_bytes = img.convert_to_pdf()
        img.close()
        
        img_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(img_pdf)
        img_pdf.close()
        
        output_path = self._get_output_path(image_path, ".pdf")
        doc.save(output_path)
        doc.close()
        
        logger.info(f"图片转PDF完成: {output_path}")
        return output_path

    def _text_to_pdf(self, txt_path: str) -> str:
        """文本转PDF"""
        logger.info(f"正在将文本转换为PDF: {txt_path}")
        
        # 读取文本内容
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用PyMuPDF创建PDF
        doc = fitz.open()
        page = doc.new_page()
        
        # 设置字体和布局
        text_rect = page.rect
        text_rect.x0 += 50  # 左边距
        text_rect.y0 += 50  # 上边距
        text_rect.x1 -= 50  # 右边距
        text_rect.y1 -= 50  # 下边距
        
        # 插入文本
        page.insert_textbox(
            text_rect,
            text,
            fontsize=12,
            fontname="china-s",
            align=fitz.TEXT_ALIGN_LEFT
        )
        
        output_path = self._get_output_path(txt_path, ".pdf")
        doc.save(output_path)
        doc.close()
        
        logger.info(f"文本转PDF完成: {output_path}")
        return output_path

    def _html_to_pdf(self, html_path: str) -> str:
        """HTML转PDF"""
        logger.info(f"正在将HTML转换为PDF: {html_path}")
        
        # 使用PyMuPDF将HTML转为PDF
        doc = fitz.open(html_path)
        output_path = self._get_output_path(html_path, ".pdf")
        doc.save(output_path)
        doc.close()
        
        logger.info(f"HTML转PDF完成: {output_path}")
        return output_path

    def _office_to_pdf(self, office_path: str) -> str:
        """Office文档转PDF（使用LibreOffice）"""
        logger.info(f"正在将Office文档转换为PDF: {office_path}")
        
        # 检查LibreOffice是否安装
        try:
            subprocess.run(["soffice", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise PDFReadError("未找到LibreOffice，无法转换Office文档。请安装LibreOffice并确保soffice命令可用。")

        output_path = self._get_output_path(office_path, ".pdf")
        output_dir = os.path.dirname(output_path)
        
        try:
            # 使用LibreOffice命令行转换
            cmd = [
                "soffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", output_dir,
                office_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise PDFReadError(f"Office转换失败: {result.stderr}")
                
            # LibreOffice默认输出文件名为原文件名.pdf，可能在不同位置
            original_name = Path(office_path).stem
            expected_pdf = os.path.join(output_dir, f"{original_name}.pdf")
            
            if os.path.exists(expected_pdf):
                # 如果在输出目录找到了，直接使用
                return expected_pdf
            else:
                # 如果没在预期位置，查找当前目录
                current_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
                if current_files:
                    # 返回最新生成的PDF
                    latest_pdf = max([os.path.join(output_dir, f) for f in current_files], key=os.path.getctime)
                    return latest_pdf
                else:
                    raise PDFReadError(f"未找到转换后的PDF文件: {expected_pdf}")
                    
        except subprocess.TimeoutExpired:
            raise PDFReadError("Office文档转换超时")
        except Exception as e:
            raise PDFReadError(f"Office文档转换失败: {e}")

    def _get_output_path(self, input_path: str, output_ext: str) -> str:
        """生成输出文件路径"""
        input_path = Path(input_path)
        output_filename = input_path.stem + output_ext
        return str(settings.CACHE_DIR / output_filename)
