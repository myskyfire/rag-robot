# 通用多格式文档智能问答 Agent

## 项目定位
通用 RAG 智能问答体，支持 Word/Excel/PPT/图片/TXT/网页/PDF 全格式输入，先统一转标准 PDF，再自动区分原生文本 PDF、扫描图片 PDF；完成解析、结构化提取、向量建库、检索问答、来源溯源、答案幻觉自检全闭环。

## 技术栈

### 核心框架
- **纯 Python 实现**：不依赖 LangChain/LlamaIndex，全流程手写，透明可控

### 文档处理
- **全格式转 PDF**：LibreOffice（Office文档）、PyMuPDF（图片/文本）
- **PDF 读取与判别**：PyMuPDF（逐页文本提取、扫描页判断、页面转图）
- **OCR 识别**：PaddleOCR（中文文字+表格识别，支持 Python 3.12）

### 向量检索
- **Embedding 模型**：支持本地 Ollama（bge-m3）或在线 API（DashScope/OpenAI）
- **向量数据库**：FAISS（Facebook 开源，本地部署、检索高效）
- **后续可替换**：ChromaDB、Milvus、Qdrant 等（只需替换 vectorstore.py 实现）

### 大模型
- **LLM**：OpenAI 兼容 API（默认通义千问 qwen3.6-flash）
- **支持任意兼容 OpenAI 格式的模型**：修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL` 即可切换

### 可选组件
- **OpenCV**：图像预处理（倾斜矫正、灰度化、降噪），未安装时自动跳过
- **多模态模型**：OCR 识别差时的兜底方案（Qwen-VL/GPT-4V 等），配置可选

## 项目结构
```
rag-robot/
├── config/
│   ├── settings.py          # 全局配置（读取.env）
│   └── business.py          # 业务适配配置（关键词库、分块参数）
├── core/
│   ├── file_converter.py    # 0号模块：全格式转PDF
│   ├── document_preprocessor.py  # 1号模块：PDF判别预处理
│   ├── content_parser.py    # 2号模块：结构化解析（含多模态兜底）
│   ├── ocr_postprocessor.py # OCR后处理（断行合并/纠错/置信度分级/表格容错）
│   ├── chunker.py           # 3号模块：分块策略
│   ├── vectorstore.py       # 3号模块：FAISS向量库
│   ├── retriever.py         # 4号模块：多路检索（向量+关键词）
│   ├── llm_service.py       # 5号模块：LLM服务
│   ├── answer_generator.py  # 5号模块：答案生成溯源
│   ├── answer_checker.py    # 6号模块：自检风控
│   ├── embedding_service.py # Embedding服务（Ollama）
│   └── multimodal_fallback.py # 多模态兜底服务
├── utils/                   # 日志、异常捕获、通用工具
├── test/                    # 自动化测试脚本
├── data/                    # 存放原始文件
├── cache/                   # 缓存目录
├── logs/                    # 日志目录
├── docs/                    # 设计文档、测试报告、扩展方案
├── ocr_results/             # OCR导出结果
├── .env                     # 环境配置（实际使用）
├── .env.example             # 环境配置模板
├── requirements.txt         # Python依赖
├── main.py                  # 程序入口
├── test_questions_direct.py # 批量测试脚本
└── export_ocr_results.py    # OCR结果导出脚本
```

## 快速开始

### 1. 环境要求
- **Python 3.12**（PaddleOCR 仅支持 Python 3.12 及以下版本）
- **Ollama**：本地 Embedding 服务（需安装并启动）
- **LibreOffice**（可选）：Office 文档转 PDF

### 2. 安装依赖
```bash
# 创建 Python 3.12 虚拟环境
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 Embedding 服务（二选一）

**方案 A：本地 Ollama（推荐，免费）**
```bash
# 安装 Ollama（https://ollama.com）
# 拉取 embedding 模型
ollama pull bge-m3

# 启动服务（默认端口 11434）
ollama serve
```

**方案 B：在线 API（DashScope/OpenAI 等）**
```bash
# 无需安装 Ollama，直接配置 .env 中的在线服务地址
# 示例：DashScope text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIM=1024
```

### 4. 配置环境变量
复制 `.env.example` 为 `.env`，修改以下配置：
```bash
cp .env.example .env
```

**必须修改的配置**：
```env
# LLM API Key（替换为你的实际密钥）
LLM_API_KEY=your-api-key

# LLM 模型（可选，默认通义千问）
LLM_MODEL=qwen3.6-flash-2026-04-16
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Embedding 服务配置（二选一）
# 方案 A：本地 Ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

# 方案 B：在线 API（注释掉方案 A，启用方案 B）
# EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# EMBEDDING_MODEL=text-embedding-v3
# EMBEDDING_DIM=1024
```

**可选配置**：
```env
# 多模态模型（OCR兜底识别，不配置则不启用）
MULTIMODAL_API_KEY=your-multimodal-api-key
MULTIMODAL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MULTIMODAL_MODEL=qwen-vl-plus
OCR_FALLBACK_CONFIDENCE_THRESHOLD=0.6

# 业务类型（general/contract/finance/manual/standard）
BUSINESS_TYPE=general

# 扫描目录
SCAN_DIR=data
AUTO_SCAN_ON_STARTUP=true
```

### 5. 放置文档
将待处理的文档放入 `data/` 目录，支持格式：
- Office：doc/docx/xls/xlsx/ppt/pptx
- 图片：jpg/png/bmp/gif/tiff
- 文本：txt、html
- PDF：原生文本 PDF、扫描版 PDF

### 6. 启动程序
```bash
# 使用 Python 3.12 环境运行
.\.venv312\Scripts\python.exe main.py
```

程序启动后会自动扫描 `data/` 目录，构建向量知识库，进入交互问答模式。

### 7. 运行测试
```bash
# 批量自动化测试（36个测试用例）
.\.venv312\Scripts\python.exe test/test_rag.py

# 测试结果保存到 test/test_results.txt
```

### 8. 导出 OCR 结果
```bash
# 导出4个页面的OCR识别结果到 ocr_results/ 目录
.\.venv312\Scripts\python.exe export_ocr_results.py
```

## 模块说明

### 业务适配配置（config/business.py）
不同业务场景（通用/合同/金融/产品手册/标准文档）的差异化配置，包括：
- **关键词库**：各领域专业术语，用于检索时增强关键词匹配
- **分块参数**：不同场景的 chunk_size / overlap 可独立调整
- **一键切换**：修改 `.env` 中 `BUSINESS_TYPE` 即可

| 业务类型 | 说明 | 分块大小 |
|---------|------|---------|
| general | 通用文档 | 512/64 |
| contract | 合同文档（权责、日期、金额） | 384/48 |
| finance | 金融文档（利率、费率、账目） | 384/48 |
| manual | 产品手册（参数、规格、功能） | 512/64 |
| standard | 标准文档（标准号、技术要求） | 384/48 |

### 模块0：多格式统一转PDF
- Office 文档 → LibreOffice 转换
- 图片 → PyMuPDF 拼接生成 PDF
- 文本/HTML → PyMuPDF 导出 PDF

### 模块1：PDF类型判别与预处理
- 逐页读取 PDF，提取原生文本
- 文本量 < 阈值 → 扫描型 PDF，转图片等待 OCR
- 文本完整可读 → 原生文本 PDF，直接抽取

### 模块2：多内容结构化解析
- 普通正文：保留段落顺序，清洗噪点
- 编号条款：识别条文序号、层级关系
- 表格内容：OCR 表格解析（智能 Y 阈值、X 坐标聚类列对齐、数值校验）
- 多模态兜底：OCR 置信度低时，自动调用多模态模型二次识别

### OCR后处理模块
- **断行合并**：基于 Y 坐标 + 语义规则，自动合并非正常断行
- **文本纠错**：内置形近字/符号纠错字典
- **单位标准化**：统一标准化单位、编号、小数、比值格式
- **残缺检测**：判断语句末尾截断，标记完整/残缺状态
- **置信度分级**：高/中/低三档，低置信内容打风险标签
- **表格容错**：过滤空白单元格，校验数值合理性，规整错位行列

### 模块3：智能分块与向量知识库构建
- 通用滑动窗口分块，保证语义完整
- 表格数据独立切块存储
- 向量存入 FAISS 索引，携带置信度、完整性、风险等级等元数据

### 模块4：问题理解与多路检索召回
- **向量相似度检索**：召回全局高相关语义片段
- **关键词精准匹配**：条款编号、表格字段、专业名词
- **双路合并去重**：综合排序，返回 Top-K 结果

### 模块5：LLM答案生成与溯源拼接
- 强制约束仅依据检索到的文档内容作答
- 输出格式：回答 + 引用来源（页码+内容类型+摘要）+ 原文片段
- 自动添加 OCR 风险提示

### 模块6：答案自检与边界风控
- **无依据校验**：无匹配文档内容，直接拒答
- **幻觉校验**：比对答案与原文语义，超出范围则修正
- **OCR 风险校验**：低置信识别片段标注可能存在误差
- **边界场景**：模糊提问、跨页内容、无关问题统一合规应答

## 技术选型说明

| 组件 | 当前方案 | 可替换方案 | 选型理由 |
|------|---------|-----------|---------|
| 向量数据库 | FAISS | ChromaDB、Milvus、Qdrant | 本地部署、检索高效、无额外服务依赖 |
| Embedding | Ollama (bge-m3) | OpenAI text-embedding、DashScope | 本地服务、免费、中文效果好 |
| LLM | 通义千问 (OpenAI兼容API) | GPT-4、Claude、本地部署模型 | API 调用简单、中文理解好 |
| OCR | PaddleOCR | Tesseract、MinerU | 中文识别率高、支持表格识别 |
| 框架 | 纯 Python 手写 | LangChain、LlamaIndex | 流程透明可控、便于理解底层原理 |

## 已知局限
- LibreOffice 转换需要系统安装 LibreOffice
- PaddleOCR 首次加载较慢（需下载模型）
- 复杂表格 OCR 识别准确率有待提升
- Python 3.13+ 不支持 PaddleOCR，需使用 Python 3.12

## 文档索引
- [设计文档](docs/设计文档.md)：系统架构、模块设计、关键决策、完成度说明
- [AI工具使用说明](docs/AI工具使用说明.md)：AI工具使用记录、校验方法、效率提升说明
- [测试与评估报告](docs/测试与评估报告.md)：测试用例、测试结果、评估指标、回归测试计划
- [业务场景扩展方案](docs/业务场景扩展方案.md)：金融、合同、合规、产品手册场景适配方案
- [多模态兜底说明](docs/多模态兜底说明.md)：多模态模型配置、触发条件、风险提示
- [问题汇总](问题.md)：开发过程中遇到的问题及解决方案
