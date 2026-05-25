"""日志系统"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "rag_robot", log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """配置全局日志"""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 格式化
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    log_file = os.path.join(log_dir, f"rag_robot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logger()
