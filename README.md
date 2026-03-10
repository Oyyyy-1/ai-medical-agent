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
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> 基于多模态大模型的医学影像 AI 辅助诊断系统。以 **LangGraph 有向图状态机**编排完整诊断工作流，支持本地 Ollama 离线推理与云端多平台 API，集成 RAG 医学知识检索和 Function Calling Tool Use Agent，输出结构化中文诊断报告并支持 PDF 导出。
>
> **⚠️ 免责声明**：本项目仅供学习和研究用途，不构成任何医疗建议，所有分析结果须由专业医疗人员审核。

---

## ✨ 功能特性

| 功能模块 | 技术实现 | 说明 |
|---------|---------|------|
| 🔀 **LangGraph 工作流** | StateGraph + 条件边 | 有向图状态机，图像质量预检→分析→条件分支→检索→报告 |
| 🖥️ **本地模型推理** | Ollama + LLaVA / moondream | 完全离线，无 API 费用，支持动态检测已安装模型 |
| ☁️ **云端多平台 API** | OpenAI 兼容格式 | 支持阿里百炼 / 硅基流动 / Gemini，侧边栏一键切换平台 |
| 🗄️ **RAG 知识检索** | LangChain + ChromaDB + sentence-transformers | 本地向量库，检索 PubMed 真实文献辅助诊断，top-k 可调 |
| 🔧 **Tool Use Agent** | OpenAI Function Calling 协议 | 模型自主决定调用 RAG 工具或网络搜索，多轮对话循环 |
| 📋 **结构化报告** | Pydantic 数据模型 | 规范 AI 输出字段，支持 JSON / PDF 两种格式导出 |
| 🔐 **密钥管理** | python-dotenv `.env` | 环境变量自动注入，无需每次手动填写 API Key |

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
app.py                    ← Streamlit UI 入口
core/
  ├─ workflow.py           LangGraph 图定义与编排
  ├─ tool_use.py           Function Calling 多轮循环
  ├─ rag.py                向量库构建与检索
  ├─ cloud_analyzer.py     云端 API 调用（OpenAI 兼容）
  ├─ ollama_analyzer.py    本地 Ollama 调用（HTTP 直调）
  ├─ pdf_report.py         PDF 报告生成
  └─ models.py             Pydantic 数据模型
```

---

## 🛠️ 技术栈

- **工作流引擎**：LangGraph — StateGraph 有向图，4个节点 + 条件边，异常影像自动走深度检索分支
- **Web 框架**：Streamlit — 页面配置、Session State、`@st.cache_resource` 缓存
- **本地模型**：Ollama — HTTP REST API 直调（绕过 SDK 兼容问题，等价于 curl）
- **云端模型**：阿里百炼 Qwen-VL-Max / 硅基流动 / Google Gemini（OpenAI 兼容格式）
- **RAG 框架**：LangChain + ChromaDB — 向量化存储，余弦相似度检索，持久化避免重复构建
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2（本地运行，384 维向量，约 80MB）
- **Tool Use**：OpenAI Function Calling 协议 — JSON Schema 工具定义，多轮对话历史维护
- **文献数据**：PubMed E-utilities API（Bio.Entrez）— 抓取真实放射学文献摘要
- **结构化输出**：Pydantic BaseModel + 四层 repair_json 容错解析
- **PDF 生成**：reportlab — 内存缓冲区生成，不落磁盘
- **密钥管理**：python-dotenv — .env 文件自动注入环境变量

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Oyyyy-1/ai-medical-agent.git
cd ai-medical-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key（使用云端模型时需要）

在项目目录下创建 `.env` 文件：

```env
# 阿里百炼（推荐，国内直连，新用户有免费额度）
# 获取地址：bailian.console.aliyun.com
DASHSCOPE_API_KEY=sk-your-key-here
```

### 4. （可选）本地模型 — 完全离线免费

```bash
# 安装 Ollama：https://ollama.com
ollama pull moondream     # 轻量版，约 1.8GB，适合低配机器
ollama pull llava:7b      # 标准版，约 4.7GB，需 8GB+ 内存
```

### 5. （可选）构建 PubMed 真实文献知识库

```bash
pip install biopython
python build_knowledge_base.py   # 约 3-5 分钟，生成 medical_knowledge_base.json
```

> 不运行此步骤也可以正常使用，系统会自动使用内置的 15 条知识片段作为 RAG 知识库。

### 6. 启动

```bash
streamlit run app.py
```

访问 `http://localhost:8501`

---

## 📖 使用流程

1. **选择模型**：侧边栏切换本地 Ollama 或云端 API，云端模式选择平台预设
2. **上传影像**：支持 JPG、PNG（X光、CT、MRI、超声等）
3. **开始分析**：点击"🔍 开始分析"，云端模式约 10-20 秒完成全流程
4. **查看报告**：
   - 顶部工作流路径条（🟢 正常路径 / 🔴 异常路径触发深度检索）
   - 影像概览（类型 / 部位 / 严重程度 / 可信度）
   - 主要发现 & 图像质量 & 主要诊断 & 鉴别诊断 & 临床建议
   - 患者友好说明（通俗中文解读）
   - **🔧 Tool Use Agent 综合建议**（异常影像时，模型自主调用工具后的汇总）
   - RAG 知识库参考（可调节 top-k）
5. **导出报告**：下载 JSON 或 PDF 格式

---

## 🔑 支持的云端平台

| 平台 | 推荐模型 | Base URL | 免费额度 |
|------|---------|----------|---------|
| 阿里百炼 | `qwen-vl-max` | `dashscope.aliyuncs.com/compatible-mode/v1` | 新用户100万tokens |
| 硅基流动 | `Pro/Qwen/Qwen2.5-VL-7B-Instruct` | `api.siliconflow.cn/v1` | 注册赠送（视觉收费） |
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
│   ├── cloud_analyzer.py         ← 云端 API 调用
│   ├── ollama_analyzer.py        ← 本地 Ollama 调用
│   ├── pdf_report.py             ← PDF 生成
│   └── models.py                 ← Pydantic 数据模型
├── build_knowledge_base.py       ← PubMed 文献抓取脚本
├── medical_knowledge_base.json   ← RAG 知识库原始数据
├── requirements.txt
├── .env                          ← API Keys（！！！加入 .gitignore防止上传API到GitHub）
├── .gitignore
└── README.md
```

> `medical_kb_store/`（ChromaDB 向量数据库）由程序首次启动时自动生成，不纳入版本控制。第二次启动后直接从磁盘加载，速度从 30 秒降至 1-2 秒。

---

## 💡 技术要点

| 技术点 | 在本项目中的体现 |
|-------|---------------|
| LangGraph | StateGraph 有向图，4节点+条件边，异常影像走深度检索分支，正常影像走快速路径 |
| RAG | LangChain + ChromaDB，PubMed 文献嵌入，余弦相似度检索，持久化向量库 |
| Function Calling | OpenAI 协议 tool_calls 多轮循环，两个工具：RAG检索 + DuckDuckGo 网络搜索 |
| Prompt Engineering | 结构化 JSON 输出，四层 repair_json 容错，temperature=0.1 保证格式稳定 |
| 多模态模型 | 图像 base64 编码，image_url 格式传入，LLaVA / Qwen-VL 视觉模型 |
| 工程实践 | 单一职责模块化拆分，.env 密钥管理，@st.cache_resource 缓存，多平台兼容设计 |

---

## 🤝 致谢

基于 [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) 二次开发，在原有基础上扩展了 LangGraph 工作流、RAG、多平台 API、Tool Use、模块化重构、中文化界面等功能。
