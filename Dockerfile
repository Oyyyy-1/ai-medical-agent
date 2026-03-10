# ============================================================
# Dockerfile — AI 医学影像智能诊断系统
# ============================================================
#
# 构建命令：docker compose up --build
# 访问地址：http://localhost:8501
# ============================================================

# ── 第1步：选择基础镜像 ──────────────────────────────────────
# python:3.10-slim 是官方 Python 精简版（基于 Debian）
# 约 130MB，包含 Python 3.10 + pip
FROM python:3.10-slim

# ── 第2步：设置工作目录 ──────────────────────────────────────
WORKDIR /app

# ── 第3步：换国内软件源 + 安装系统依赖 ───────────────────────
# 为什么换源：
#   默认 deb.debian.org 是官方源，国内访问经常超时或 502
#   换成清华大学镜像源（mirrors.tuna.tsinghua.edu.cn），速度快且稳定
#
# sed 命令原理：
#   把 sources.list 里所有 deb.debian.org 替换成清华镜像地址
#   把 security.debian.org 也替换成清华安全更新镜像
#
# 系统包说明：
#   libsqlite3-dev : chromadb 底层用 sqlite3 存储向量索引
#   libgomp1       : sentence-transformers 的 OpenMP 并行计算支持
#   git            : 某些 pip 包从 git 安装时需要
#   curl           : 健康检查用
#   build-essential: 编译 C 扩展的工具链
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y \
    libsqlite3-dev \
    libgomp1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── 第4步：换国内 pip 源 + 安装 Python 依赖 ──────────────────
# 为什么换 pip 源：
#   默认 pypi.org 国内访问慢，换成清华 pip 镜像
#   -i 指定镜像地址，--trusted-host 信任该域名（跳过 SSL 校验问题）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# ── 第5步：预下载 sentence-transformers 嵌入模型 ─────────────
# 在构建阶段下载，打进镜像，容器启动后直接本地加载
# 设置 HuggingFace 镜像为国内镜像站，解决直连超时问题
RUN HF_ENDPOINT=https://hf-mirror.com python -c "\
from sentence_transformers import SentenceTransformer; \
print('正在下载嵌入模型...'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('模型下载完成')"

# ── 第6步：复制项目文件 ──────────────────────────────────────
COPY . .

# ── 第7步：声明端口 ──────────────────────────────────────────
EXPOSE 8501

# ── 第8步：健康检查 ──────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── 第9步：启动命令 ──────────────────────────────────────────
# --server.address=0.0.0.0  容器内必须监听所有接口，否则宿主机访问不到
# --server.headless=true    无头模式，不自动打开浏览器
# --server.fileWatcherType=none  关闭文件监控，节省资源
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
