# 🏥 AI 医学影像智能诊断系统

<p align="center">
  <img src="assets/demo_screenshot.png" alt="系统界面演示" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/LangGraph-工作流状态机-blueviolet" />
  <img src="https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink" />
  <img src="https://img.shields.io/badge/Function%20Calling-Tool%20Use-orange" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-yellow" />
  <img src="https://img.shields.io/badge/Docker-容器化部署-2496ED?logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> 基于多模态大模型的医学影像 AI 辅助诊断系统。以 **LangGraph 有向图状态机**编排完整诊断工作流，支持本地 Ollama 离线推理与云端多平台 API，集成 RAG 医学知识检索和 Function Calling Tool Use Agent，输出结构化中文诊断报告并支持 PDF 导出。支持 **Docker 一键部署**与**流式实时输出**。
>
> **⚠️ 免责声明**：本项目仅供学习和研究用途，不构成任何医疗建议，所有分析结果须由专业医疗人员审核。

---

## ✨ 功能特性

| 功能模块 | 技术实现 | 说明 |
|---------|---------|------|
| 🔀 **LangGraph 工作流** | StateGraph + 条件边 | 有向图状态机，图像质量预检→分析→条件分支→检索→报告 |
| ⚡ **流式输出** | SSE + Python Generator | 模型逐 token 实时渲染，感知延迟从 10 秒降至 <1 秒 |
| 🖥️ **本地模型推理** | Ollama + LLaVA / moondream | 完全离线，无 API 费用，动态检测已安装模型，🔄 一键重新检测 |
| ☁️ **云端多平台 API** | OpenAI 兼容格式 | 支持阿里百炼 / 硅基流动 / 火山方舟 / Gemini，侧边栏一键切换 |
| 🗄️ **RAG 知识检索** | LangChain + ChromaDB + sentence-transformers | 本地向量库，检索 PubMed 真实文献辅助诊断，top-k 可调 |
| 🔧 **Tool Use Agent** | OpenAI Function Calling 协议 | 模型自主决定调用 RAG 工具或网络搜索，多轮对话循环 |
| 📋 **结构化报告** | Pydantic 数据模型 | 规范 AI 输出字段，支持 JSON / PDF 两种格式导出 |
| 🐳 **Docker 部署** | Dockerfile + docker-compose | 一键启动，无需配置环境，数据卷持久化，含国内镜像源加速 |
| 🔐 **密钥管理** | python-dotenv `.env` | 环境变量自动注入，密钥不进代码仓库 |

---

## 🏗️ 系统架构

### LangGraph 工作流（核心）

```
用户上传医学影像
        │
        ▼
┌───────────────────────┐
│  node_check_quality   │  ← 纯本地逻辑，不调用 AI
│  图像质量预检          │    分辨率不足或纯色图 → 直接拦截
└──────────┬────────────┘
           │ quality_ok = True
           ▼
┌───────────────────────┐
│  node_analyze_image   │  ← 调用模型（Ollama 或云端 API）
│  影像分析              │    图像 base64 编码 → 结构化 JSON 输出
└──────────┬────────────┘
           │
    [条件边 route_by_severity]
           │
     ┌─────┴──────────────┐
     │                     │
  Normal / Unknown    Mild / Moderate / Severe
     │                     │
     ▼                     ▼
┌──────────┐      ┌─────────────────────┐
│ 快速路径  │      │  node_deep_retrieval │
│ 基础 RAG  │      │  深度检索            │
│ 不调工具  │      │  RAG + Tool Use      │
└────┬─────┘      └──────────┬──────────┘
     │                       │
     └───────────┬───────────┘
                 ▼
     ┌───────────────────────┐
     │  node_generate_report │  ← 汇总所有结果，输出最终 State
     └───────────┬───────────┘
                 ▼
                END
```

### Tool Use 循环（异常路径内）

```
诊断结果 + 两个工具定义
        │
        ▼
   第1轮请求 → 模型返回 tool_calls
        │
        ▼
   本地执行工具
   ├─ search_rag_knowledge      → ChromaDB 余弦相似度检索
   └─ search_medical_guidelines → DuckDuckGo 即时搜索
        │
        ▼
   工具结果以 role="tool" 追加到 messages
        │
        ▼
   第2轮请求 → finish_reason="stop" → 最终综合建议
   （最多循环 3 轮，防止无限调用）
```

### 项目模块结构

```
app.py                    ← Streamlit UI 入口（纯界面，无业务逻辑）
core/
  ├─ workflow.py           LangGraph 图定义与编排
  ├─ tool_use.py           Function Calling 多轮循环
  ├─ rag.py                向量库构建与检索
  ├─ cloud_analyzer.py     云端 API 调用（含流式输出）
  ├─ ollama_analyzer.py    本地 Ollama 调用（环境变量控制地址）
  ├─ pdf_report.py         PDF 报告生成
  └─ models.py             Pydantic 数据模型
```

---

## 🛠️ 技术栈

- **工作流引擎**：LangGraph — StateGraph 有向图，4节点 + 条件边，异常影像自动走深度检索分支
- **流式输出**：SSE（Server-Sent Events）+ Python Generator，`requests stream=True` 逐行读取，`st.empty()` 逐字渲染
- **Web 框架**：Streamlit — Session State、`@st.cache_resource` 缓存、`st.rerun()` 强制刷新
- **本地模型**：Ollama — HTTP REST API 直调，地址由 `OLLAMA_HOST` 环境变量控制，Docker 内自动切换为 `host.docker.internal`
- **云端模型**：阿里百炼 Qwen-VL-Max / 硅基流动 / 火山方舟豆包 / Google Gemini（OpenAI 兼容格式）
- **RAG 框架**：LangChain + ChromaDB — 向量化存储，余弦相似度检索，持久化避免重复构建
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2（本地运行，384 维，约 80MB，构建时预下载进镜像）
- **Tool Use**：OpenAI Function Calling 协议 — JSON Schema 工具定义，多轮对话历史维护
- **文献数据**：PubMed E-utilities API（Bio.Entrez）— 抓取真实放射学文献摘要
- **结构化输出**：Pydantic BaseModel + 四层 repair_json 容错解析
- **PDF 生成**：reportlab — 内存缓冲区生成，不落磁盘
- **容器化**：Docker + docker-compose — 国内镜像源加速构建，数据卷持久化，env_file 密钥注入

---

## 🚀 快速开始

### 方式一：Docker 一键部署（推荐，无需配置 Python 环境）

**前置条件**：安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. 克隆项目
git clone https://github.com/Oyyyy-1/ai-medical-agent.git
cd ai-medical-agent

# 2. 配置 API Key
echo "DASHSCOPE_API_KEY=sk-your-key-here" > .env

# 3. 一键启动（首次约 10-20 分钟构建，之后几秒）
docker compose up -d

# 访问 http://localhost:8501
```

**常用 Docker 命令：**

```bash
docker compose up -d          # 后台启动
docker compose down           # 停止
docker compose logs -f        # 查看实时日志
docker compose up --build     # 修改代码后重新构建
docker compose ps             # 查看容器状态
docker system prune -f        # 定期清理无用镜像和缓存（释放磁盘空间）
```

> **注意**：Docker 模式下使用本地 Ollama，需在宿主机运行 `ollama serve`，容器会自动通过 `host.docker.internal` 连接。

---

### 方式二：本地直接运行

```bash
# 1. 克隆并安装依赖
git clone https://github.com/Oyyyy-1/ai-medical-agent.git
cd ai-medical-agent
pip install -r requirements.txt

# 2. 配置 API Key（.env 文件）
DASHSCOPE_API_KEY=sk-your-key-here

# 3. （可选）本地模型
ollama pull moondream     # 轻量版，约 1.8GB
ollama pull llava:7b      # 标准版，约 4.7GB，需 8GB+ 内存

# 4. （可选）构建 PubMed 知识库
pip install biopython
python build_knowledge_base.py

# 5. 启动
streamlit run app.py
# 访问 http://localhost:8501
```

---

## 📖 使用流程

1. **选择模型**：侧边栏切换本地 Ollama 或云端 API；若 Ollama 未启动可点击 🔄 按钮重新检测
2. **上传影像**：支持 JPG、PNG（X光、CT、MRI、超声等）
3. **开始分析**：点击"🔍 开始分析"
   - 云端 + 流式开启：模型逐字实时输出，<1 秒看到第一个字
   - 云端 + 流式关闭：LangGraph 工作流，等待约 10-20 秒
   - 本地 Ollama：CPU 推理，等待约 1-3 分钟
4. **查看报告**：
   - 顶部执行路径条（🟢 正常路径 / 🔴 异常路径触发深度检索）
   - 影像概览 / 主要发现 / 主要诊断 / 鉴别诊断 / 临床建议
   - 患者友好说明 / Tool Use Agent 综合建议 / RAG 知识库参考
5. **导出**：下载 JSON 或 PDF 格式

---

## 🔑 支持的云端平台

| 平台 | 推荐模型 | Base URL | 免费额度 |
|------|---------|----------|---------|
| 阿里百炼 | `qwen-vl-max` | `dashscope.aliyuncs.com/compatible-mode/v1` | 新用户100万tokens |
| 硅基流动 | `Pro/Qwen/Qwen2.5-VL-7B-Instruct` | `api.siliconflow.cn/v1` | 注册赠送（视觉收费） |
| 火山方舟 | `doubao-vision-pro-32k` | `ark.cn-beijing.volces.com/api/v3` | 新用户有免费额度 |
| Google Gemini | `gemini-2.0-flash` | `generativelanguage.googleapis.com/v1beta/openai` | 1500次/天（需代理） |
| 自定义 | 任意 OpenAI 兼容模型 | 填入对应地址 | 取决于平台 |

---

## 📁 项目结构

```
ai-medical-agent/
├── app.py                        ← Streamlit 主入口
├── core/
│   ├── __init__.py
│   ├── workflow.py               ← LangGraph 工作流
│   ├── tool_use.py               ← Function Calling Agent
│   ├── rag.py                    ← RAG 知识库
│   ├── cloud_analyzer.py         ← 云端 API 调用（含流式）
│   ├── ollama_analyzer.py        ← 本地 Ollama 调用
│   ├── pdf_report.py             ← PDF 生成
│   └── models.py                 ← Pydantic 数据模型
├── build_knowledge_base.py       ← PubMed 文献抓取脚本
├── medical_knowledge_base.json   ← RAG 知识库原始数据
├── Dockerfile                    ← Docker 镜像构建（含国内源加速）
├── docker-compose.yml            ← 一键启动配置
├── .dockerignore                 ← Docker 构建排除文件
├── requirements.txt
├── .env                          ← API Keys（!!!已加入 .gitignore，防止API key上传）
├── .gitignore
└── README.md
```

> `medical_kb_store/`（ChromaDB 向量数据库）首次启动自动生成，不纳入版本控制。Docker 模式下通过 volumes 挂载到宿主机，容器重建后数据保留。

---

## 💡 技术要点

| 技术点 | 在本项目中的体现 |
|-------|---------------|
| LangGraph | StateGraph 有向图，4节点+条件边，异常影像走深度检索分支，正常影像走快速路径 |
| 流式输出 | SSE + Python Generator，`requests stream=True`，`st.empty()` 逐字渲染，感知延迟<1秒 |
| RAG | LangChain + ChromaDB，PubMed 文献嵌入，余弦相似度检索，持久化向量库 |
| Function Calling | OpenAI 协议 tool_calls 多轮循环，RAG检索 + DuckDuckGo 网络搜索两个工具 |
| Prompt Engineering | 结构化 JSON 输出，四层 repair_json 容错，temperature=0.1 保证格式稳定 |
| 多模态模型 | 图像 base64 编码，image_url 格式传入，LLaVA / Qwen-VL 视觉模型 |
| Docker | 分层构建缓存、国内源加速、模型预下载进镜像、volumes 持久化、env_file 密钥注入、OLLAMA_HOST 环境变量跨容器访问宿主机 |
| 工程实践 | 单一职责模块化，.env 密钥管理，@st.cache_resource，st.rerun() 强制刷新，多平台兼容 |

---

## 🤝 致谢

基于 [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) 二次开发，在原有基础上扩展了 LangGraph 工作流、流式输出、RAG、多平台 API、Tool Use、Docker 部署、Ollama 动态检测、模块化重构、中文化界面等功能。
