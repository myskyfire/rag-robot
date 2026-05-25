"""全局配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # 项目路径
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    LOG_DIR = BASE_DIR / "logs"

    # LLM配置
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")

    # Embedding配置
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    # PDF处理配置
    PDF_TEXT_THRESHOLD: int = int(os.getenv("PDF_TEXT_THRESHOLD", "50"))  # 判断原生/扫描PDF的阈值
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))

    # 分块配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # 检索配置
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    KEYWORD_TOP_K: int = int(os.getenv("KEYWORD_TOP_K", "5"))

    # 自检配置
    SELF_CHECK_THRESHOLD: float = float(os.getenv("SELF_CHECK_THRESHOLD", "0.7"))

    # 多模态模型配置（OCR兜底识别，可选）
    MULTIMODAL_API_KEY: str = os.getenv("MULTIMODAL_API_KEY", "")
    MULTIMODAL_BASE_URL: str = os.getenv("MULTIMODAL_BASE_URL", "")
    MULTIMODAL_MODEL: str = os.getenv("MULTIMODAL_MODEL", "qwen-vl-plus")
    OCR_FALLBACK_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_FALLBACK_CONFIDENCE_THRESHOLD", "0.6"))  # 低于此值触发多模态兜底

    # 业务适配配置（可扩展）
    BUSINESS_TYPE: str = os.getenv("BUSINESS_TYPE", "general")  # general/contract/finance/manual/standard

    # 扫描目录配置
    SCAN_DIR: str = os.getenv("SCAN_DIR", str(Path(__file__).resolve().parent.parent / "data"))
    AUTO_SCAN_ON_STARTUP: bool = os.getenv("AUTO_SCAN_ON_STARTUP", "true").lower() == "true"
    SCAN_FILE_EXTENSIONS: list = None  # None表示支持所有支持的格式

    def __init__(self):
        # 确保目录存在
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        Path(self.SCAN_DIR).mkdir(parents=True, exist_ok=True)


settings = Settings()
