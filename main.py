"""主流程入口"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 禁用 oneDNN 以避免兼容性问题
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_cudnn"] = "0"

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from config.business import get_business_config
from core.file_converter import FileConverter
from core.document_preprocessor import DocumentPreprocessor
from core.content_parser import ContentParser
from core.chunker import TextChunker
from core.vectorstore import VectorStore
from core.retriever import Retriever
from core.llm_service import LLMService
from core.answer_generator import AnswerGenerator
from core.answer_checker import AnswerChecker
from core.embedding_service import OllamaEmbeddingService
from utils.logger import logger
from utils.exceptions import (
    PDFReadError, PDFParseError, OCRParseError,
    VectorStoreError, RetrievalError, LLMError
)


class RAGRobot:
    """通用多格式文档智能问答Agent"""

    def __init__(self, collection_name: str = "default", business_type: str = "general"):
        self.collection_name = collection_name
        self.business_type = business_type

        # 初始化各模块
        self.file_converter = FileConverter()
        self.content_parser = ContentParser()
        self.llm_service = LLMService()
        self.answer_generator = AnswerGenerator(self.llm_service)
        self.answer_checker = AnswerChecker()
        self.embedding_service = OllamaEmbeddingService()

        # 向量库和检索器
        self.vectorstore = VectorStore(index_name=self.collection_name)
        self.retriever = None

        # 已处理文件记录
        self.processed_files: set = set()

    def build(self, file_path: str):
        """
        构建知识库完整流程
        任意格式文件 → 统一转为标准 PDF → 判断原生/扫描 PDF → 页面预处理 →
        文本/条款/表格结构化提取 → 语义分块 → 向量知识库构建
        """
        file_path = str(Path(file_path).resolve())
        file_name = Path(file_path).name

        if file_name in self.processed_files:
            logger.info(f"文件已处理过，跳过: {file_name}")
            return

        logger.info("=" * 60)
        logger.info(f"开始构建知识库: {file_name}")
        logger.info("=" * 60)

        try:
            # 模块0：多格式转PDF
            logger.info("[模块0] 多格式统一转PDF")
            pdf_path = self.file_converter.convert_to_pdf(file_path)
            logger.info(f"转换完成: {pdf_path}")

            # 模块1：PDF类型判别与页面预处理
            logger.info("[模块1] PDF类型判别与页面预处理")
            preprocessor = DocumentPreprocessor(pdf_path)
            pdf_doc = preprocessor.process()
            logger.info(f"PDF类型: {'原生' if pdf_doc.is_native else '扫描'}, 页数: {pdf_doc.total_pages}")

            # 模块2：文本/条款/表格结构化解析
            logger.info("[模块2] 多内容结构化解析")
            text_blocks = self.content_parser.parse_document(pdf_doc)
            logger.info(f"解析完成: {len(text_blocks)}个文本块")

            # 模块3：智能分块+向量知识库构建
            logger.info("[模块3] 智能分块与向量知识库构建")
            chunker = TextChunker()
            chunks = chunker.chunk_blocks(text_blocks)
            logger.info(f"分块完成: {len(chunks)}个chunks")

            # 生成向量嵌入
            embeddings = self._generate_embeddings(chunks)
            logger.info(f"向量生成完成: {len(embeddings)}个")

            # 构建向量库
            self.vectorstore.build(chunks, embeddings, source_file=file_name)
            logger.info("向量库构建完成")

            # 初始化检索器
            self.retriever = Retriever(self.vectorstore)

            # 记录已处理
            self.processed_files.add(file_name)

            logger.info("=" * 60)
            logger.info(f"知识库构建完成: {file_name}")
            logger.info("=" * 60)

        except PDFReadError as e:
            logger.error(f"PDF读取失败: {e}")
            raise
        except PDFParseError as e:
            logger.error(f"PDF解析失败: {e}")
            raise
        except OCRParseError as e:
            logger.error(f"OCR识别失败: {e}")
            raise
        except VectorStoreError as e:
            logger.error(f"向量库构建失败: {e}")
            raise
        except Exception as e:
            logger.error(f"知识库构建过程中发生未知错误: {e}")
            raise

    def scan_directory(self, scan_dir: str = None):
        """
        扫描目录，自动处理所有支持的文件
        """
        scan_dir = scan_dir or settings.SCAN_DIR
        scan_dir = Path(scan_dir)

        if not scan_dir.exists():
            logger.warning(f"扫描目录不存在: {scan_dir}")
            return

        logger.info(f"开始扫描目录: {scan_dir}")

        # 支持的文件扩展名
        supported_exts = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
            '.txt', '.html', '.htm'
        }

        # 收集所有支持的文件
        files_to_process = []
        for file_path in scan_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                # 跳过缓存目录中的文件
                if "cache" in str(file_path).lower():
                    continue
                files_to_process.append(str(file_path))

        if not files_to_process:
            logger.info("扫描目录中未找到支持的文件")
            return

        logger.info(f"扫描完成，找到 {len(files_to_process)} 个待处理文件")

        success_count = 0
        fail_count = 0

        for file_path in files_to_process:
            try:
                self.build(file_path)
                success_count += 1
            except Exception as e:
                logger.error(f"文件处理失败: {file_path}, 错误: {e}")
                fail_count += 1

        logger.info(f"目录扫描处理完成: 成功={success_count}, 失败={fail_count}")

    def ask(self, query: str) -> str:
        """
        问答流程
        用户问题解析 → 多路检索召回 → LLM生成答案 → 答案可靠性自检 → 带溯源结果输出
        """
        if self.retriever is None:
            raise RetrievalError("知识库未构建，请先调用build()或scan_directory()")

        logger.info(f"收到问题: {query}")

        try:
            # 模块4：问题理解+多路检索召回
            query_embedding = self.embedding_service.embed(query)
            retrieved_docs = self.retriever.retrieve(query, query_embedding)

            # 模块5：LLM答案生成+溯源拼接
            answer_result = self.answer_generator.generate(query, retrieved_docs)

            # 模块6：答案自检+边界风控
            check_results = self.answer_checker.check_answer(
                answer_result.answer,
                answer_result.sources,
                query
            )

            # 格式化输出
            final_output = self.answer_checker.format_result(
                answer_result.answer,
                answer_result.sources,
                check_results
            )

            logger.info("问答完成")
            return final_output

        except RetrievalError as e:
            logger.error(f"检索失败: {e}")
            return f"检索过程中出现错误: {e}"
        except LLMError as e:
            logger.error(f"LLM调用失败: {e}")
            return f"答案生成失败: {e}"
        except Exception as e:
            logger.error(f"问答过程中发生未知错误: {e}")
            return f"处理问题时出现错误: {e}"

    def _generate_embeddings(self, chunks) -> List[List[float]]:
        """批量生成向量嵌入"""
        texts = [chunk.content for chunk in chunks]
        return self.embedding_service.embed_batch(texts)

    def save(self):
        """保存向量库"""
        self.vectorstore.save()

    def load(self):
        """加载向量库"""
        loaded = self.vectorstore.load()
        if loaded:
            self.retriever = Retriever(self.vectorstore)
            logger.info(f"向量库已加载: {self.collection_name}")
        return loaded

    def reset(self):
        """重置知识库"""
        self.vectorstore.reset()
        self.retriever = None
        self.processed_files.clear()
        logger.info("知识库已重置")


def create_rag_robot(collection_name: str = "default", business_type: str = "general") -> RAGRobot:
    """创建RAG机器人实例"""
    return RAGRobot(collection_name=collection_name, business_type=business_type)


def main():
    """程序入口：启动时自动扫描目录"""
    logger.info("=" * 60)
    logger.info("RAG Robot 启动中...")
    logger.info("=" * 60)

    # 创建机器人
    robot = create_rag_robot(
        collection_name="rag_robot_main",
        business_type=settings.BUSINESS_TYPE,
    )

    # 尝试加载已有索引
    if robot.load():
        logger.info("成功加载已有向量库索引")
    else:
        logger.info("未找到已有索引，将从头构建")

    # 自动扫描目录
    if settings.AUTO_SCAN_ON_STARTUP:
        logger.info(f"自动扫描目录: {settings.SCAN_DIR}")
        robot.scan_directory()

    # 保存索引
    robot.save()

    logger.info("=" * 60)
    logger.info("RAG Robot 启动完成，进入交互模式")
    logger.info("输入问题开始问答，输入 'quit' 或 'exit' 退出")
    logger.info("=" * 60)

    # 交互模式
    while True:
        try:
            query = input("\n[问题] > ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                logger.info("退出程序")
                break

            result = robot.ask(query)
            print(f"\n[回答]\n{result}")

        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            break
        except Exception as e:
            logger.error(f"交互异常: {e}")
            print(f"错误: {e}")

    # 最终保存
    robot.save()
    logger.info("向量库已保存")


if __name__ == "__main__":
    main()
