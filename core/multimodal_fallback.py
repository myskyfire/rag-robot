"""多模态模型OCR兜底识别服务（可选）

⚠️ 风险提示：
1. 多模态模型识别成本较高（按token计费），大量使用可能产生显著费用
2. 多模态模型识别速度较慢（通常3-10秒/页），影响整体处理性能
3. 多模态模型对表格、公式等专业内容的识别准确率不一定优于PaddleOCR
4. 多模态模型输出格式不固定，需要额外解析和校验
5. 依赖外部API，网络不稳定时可能失败
6. 多模态模型可能产生幻觉，识别出原文不存在的内容

建议仅在以下场景使用：
- PaddleOCR识别置信度极低（<0.6）的页面
- 关键文档需要二次校验
- OCR后处理仍无法修复的严重识别错误
"""
import base64
import json
from typing import Optional, Dict, List

import requests

from config.settings import settings
from utils.logger import logger


class MultimodalFallbackService:
    """多模态模型OCR兜底识别服务"""

    def __init__(self):
        self.api_key = settings.MULTIMODAL_API_KEY
        self.base_url = settings.MULTIMODAL_BASE_URL
        self.model = settings.MULTIMODAL_MODEL
        self.is_available = bool(self.api_key and self.base_url)

        if not self.is_available:
            logger.info("多模态模型未配置，OCR兜底识别功能不可用")
        else:
            logger.info(f"多模态模型OCR兜底服务已初始化: model={self.model}")

    def recognize_page(self, image_path: str) -> Optional[Dict]:
        """
        使用多模态模型识别页面内容
        image_path: 页面图片路径
        返回: {
            "text": "识别的文本内容",
            "tables": [{"rows": [[...]]}],  # 识别的表格
            "confidence": 0.85,  # 自评估置信度
        }
        """
        if not self.is_available:
            logger.warning("多模态模型未配置，跳过兜底识别")
            return None

        try:
            # 读取图片并转为base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 构建prompt
            prompt = self._build_prompt()

            # 调用多模态模型API
            result = self._call_multimodal_api(image_base64, prompt)

            if result:
                logger.info(f"多模态模型兜底识别完成: {image_path}")
                return result
            else:
                logger.warning(f"多模态模型兜底识别失败: {image_path}")
                return None

        except Exception as e:
            logger.error(f"多模态模型兜底识别异常: {e}")
            return None

    def _build_prompt(self) -> str:
        """构建多模态识别prompt"""
        return """请识别图片中的所有文本内容，并按以下JSON格式返回：

{
    "text": "完整的文本内容，保留段落结构",
    "tables": [
        {
            "header": ["列1", "列2", "列3"],
            "rows": [["值1", "值2", "值3"], ...]
        }
    ],
    "confidence": 0.85
}

要求：
1. 准确识别所有文字，包括标题、正文、条款编号
2. 识别表格内容，保持行列结构
3. 保留原始标点符号和格式
4. confidence为自评估置信度（0-1之间）
5. 仅返回JSON，不要其他内容"""

    def _call_multimodal_api(self, image_base64: str, prompt: str) -> Optional[Dict]:
        """调用多模态模型API"""
        try:
            # 构建请求
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1,  # 低温度确保输出稳定
            }

            # 发送请求
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            # 解析响应
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # 解析JSON输出
            return self._parse_multimodal_output(content)

        except requests.exceptions.RequestException as e:
            logger.error(f"多模态API调用失败: {e}")
            return None
        except Exception as e:
            logger.error(f"多模态API响应解析失败: {e}")
            return None

    def _parse_multimodal_output(self, content: str) -> Optional[Dict]:
        """解析多模态模型输出"""
        try:
            # 尝试直接解析JSON
            result = json.loads(content)

            # 验证必要字段
            if "text" not in result:
                logger.warning("多模态输出缺少text字段")
                return None

            # 设置默认值
            result.setdefault("tables", [])
            result.setdefault("confidence", 0.7)

            return result

        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            try:
                # 查找JSON块
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    result.setdefault("tables", [])
                    result.setdefault("confidence", 0.7)
                    return result
            except Exception as e:
                logger.error(f"从多模态输出中提取JSON失败: {e}")

            return None

    def should_fallback(self, ocr_confidence: float) -> bool:
        """判断是否需要触发多模态兜底"""
        if not self.is_available:
            return False

        return ocr_confidence < settings.OCR_FALLBACK_CONFIDENCE_THRESHOLD
