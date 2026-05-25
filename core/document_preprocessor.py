"""模块1：文档判别与预处理"""
import os
from typing import List, Tuple
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from config.settings import settings
from utils.logger import logger
from utils.exceptions import PDFReadError, PDFParseError


@dataclass
class PageContent:
    """单页内容"""
    page_num: int  # 页码（从1开始）
    page_index: int  # 页面索引（从0开始）
    text: str  # 提取的文本
    is_scanned: bool  # 是否为扫描页
    image_path: str = ""  # 如果是扫描页，图片路径


@dataclass
class PDFDocument:
    """PDF文档信息"""
    file_path: str
    file_name: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    is_native: bool = True  # 整体是否为原生PDF
    error_pages: List[int] = field(default_factory=list)  # 异常页码


class DocumentPreprocessor:
    """PDF文档判别与预处理"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.file_name = os.path.basename(pdf_path)

        if not os.path.exists(pdf_path):
            raise PDFReadError(f"PDF文件不存在: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise PDFReadError(f"非PDF文件: {pdf_path}")

    def process(self) -> PDFDocument:
        """完整处理流程：读取、判别、提取"""
        logger.info(f"开始处理PDF: {self.file_name}")

        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            raise PDFReadError(f"无法打开PDF: {e}")

        total_pages = len(doc)
        logger.info(f"PDF总页数: {total_pages}")

        pdf_doc = PDFDocument(
            file_path=self.pdf_path,
            file_name=self.file_name,
            total_pages=total_pages,
        )

        scanned_count = 0
        for page_idx in range(total_pages):
            try:
                page_content = self._process_page(doc, page_idx)
                pdf_doc.pages.append(page_content)
                if page_content.is_scanned:
                    scanned_count += 1
            except Exception as e:
                logger.error(f"第{page_idx + 1}页处理异常: {e}")
                pdf_doc.error_pages.append(page_idx + 1)

        doc.close()

        # 判断整体类型：超过50%页面为扫描则判定为扫描PDF
        if total_pages > 0:
            pdf_doc.is_native = (scanned_count / total_pages) < 0.5
            logger.info(
                f"PDF类型判定: {'原生PDF' if pdf_doc.is_native else '扫描PDF'} "
                f"(扫描页: {scanned_count}/{total_pages})"
            )

        if pdf_doc.error_pages:
            logger.warning(f"以下页面处理失败已跳过: {pdf_doc.error_pages}")

        logger.info(f"PDF处理完成: {self.file_name}")
        return pdf_doc

    def _process_page(self, doc: fitz.Document, page_idx: int) -> PageContent:
        """处理单页：提取文本、判断类型、必要时转图"""
        page = doc[page_idx]

        # 1. 尝试提取原生文本
        text = page.get_text("text")
        text_len = len(text.strip())

        # 2. 判断是否为扫描页
        is_scanned = text_len < settings.PDF_TEXT_THRESHOLD

        page_content = PageContent(
            page_num=page_idx + 1,
            page_index=page_idx,
            text=text,
            is_scanned=is_scanned,
        )

        if is_scanned:
            logger.debug(f"第{page_idx + 1}页判定为扫描页(文本长度: {text_len})，转为图片")
            # 扫描页转为图片供OCR使用
            image_path = self._page_to_image(doc, page_idx)
            page_content.image_path = image_path

        return page_content

    def _page_to_image(self, doc: fitz.Document, page_idx: int) -> str:
        """将PDF页面转为图片"""
        page = doc[page_idx]
        mat = fitz.Matrix(2, 2)  # 2倍分辨率
        pix = page.get_pixmap(matrix=mat)

        image_dir = settings.CACHE_DIR / "page_images"
        image_dir.mkdir(exist_ok=True)

        image_path = str(image_dir / f"page_{page_idx + 1}.png")
        pix.save(image_path)

        return image_path
