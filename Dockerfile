# 多阶段构建：先安装系统依赖，再安装 Python 依赖
FROM python:3.12-slim AS base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libreoffice \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p data cache logs ocr_results

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口（如果有 Web 服务）
EXPOSE 8000

# 默认命令
CMD ["python", "main.py"]
