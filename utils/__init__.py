"""工具模块"""
from .logger import logger
from .exceptions import (
    PDFReadError, PDFParseError, OCRParseError,
    VectorStoreError, RetrievalError, LLMError, SelfCheckError
)
