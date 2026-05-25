"""异常处理"""


class PDFReadError(Exception):
    """PDF读取异常"""
    pass


class PDFParseError(Exception):
    """PDF解析异常"""
    pass


class OCRParseError(Exception):
    """OCR识别异常"""
    pass


class VectorStoreError(Exception):
    """向量库异常"""
    pass


class RetrievalError(Exception):
    """检索异常"""
    pass


class LLMError(Exception):
    """LLM调用异常"""
    pass


class SelfCheckError(Exception):
    """自检异常"""
    pass
