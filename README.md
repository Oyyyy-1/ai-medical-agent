# 🏥 AI 医学影像智能诊断系统

<p align="center">
  <img src="assets/demo_screenshot.png" alt="系统界面演示" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-orange" />
  <img src="https://img.shields.io/badge/Function%20Calling-Tool%20Use-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> 基于多模态大模型的医学影像 AI 辅助诊断系统。支持本地 Ollama 离线推理与云端多平台 API，集成 RAG 医学知识检索和 Function Calling Tool Use，输出结构化中文诊断报告并支持 PDF 导出。
>
> **⚠️ 免责声明**：本项目仅供学习和研究用途，不构成任何医疗建议，所有分析结果须由专业医疗人员审核。

---

## ✨ 功能特性

| 功能模块 | 技术实现 | 说明 |
|---------|---------|------|
| 🖥️ **本地模型推理** | Ollama + LLaVA / moondream | 完全离线，无 API 费用，支持动态检测已安装模型 |
| ☁️ **云端多平台 API** | OpenAI 兼容格式 | 支持阿里百炼 / 硅基流动 / Gemini，侧边栏一键切换平台 |
| 🗄️ **RAG 知识检索** | LangChain + ChromaDB + sentence-transformers | 本地向量库，检索 PubMed 真实文献辅助诊断，top-k 可调 |
| 🔧 **Tool Use Agent** | OpenAI Function Calling 协议 | 模型自主决定调用 RAG 工具或网络搜索，多轮对话循环 |
| 📋 **结构化报告** | Pydantic 数据模型 | 规范 AI 输出字段，支持 JSON / PDF 两种格式导出 |
| 🔐 **密钥管理** | python-dotenv `.env` | 环境变量自动注入，无需每次手动填写 API Key |

---

## 🏗️ 系统架构

```
用户上传医学影像
        │
        ▼
  ┌─────────────┐       ┌──────────────────┐
  │ 本地 Ollama │  或   │   云端 API       │
  │ LLaVA/moon  │       │ 百炼/硅基/Gemini  │
  └──────┬──────┘       └────────┬─────────┘
         │                       │
         └──────────┬────────────┘
                    │ Step 1: 影像分析（多模态推理）
                    ▼
         ┌──────────────────────┐
         │  结构化 JSON 解析     │
         │  + repair_json 修复  │
         └──────────┬───────────┘
                    │ Step 2: RAG 检索
                    ▼
         ┌──────────────────────┐
         │  ChromaDB 向量检索   │
         │  PubMed 文献知识库   │
         └──────────┬───────────┘
                    │ Step 3: Tool Use（云端模式）
                    ▼
         ┌──────────────────────────────────────┐
         │        Function Calling Loop         │
         │  模型 → tool_calls → 执行工具         │
         │  ├─ search_rag_knowledge (本地RAG)   │
         │  └─ search_medical_guidelines (网络) │
         │  → 工具结果 → 模型综合 → 最终建议    │
         └──────────┬───────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Streamlit UI 渲染   │
         │  + JSON / PDF 导出   │
         └──────────────────────┘
```

---

## 🛠️ 技术栈

- **Web 框架**：Streamlit — 页面配置、Session State、cache_resource 缓存
- **本地模型**：Ollama — HTTP REST API 调用（直接 requests，不用 SDK 规避兼容问题）
- **云端模型**：阿里百炼 Qwen-VL-Max / 硅基流动 / Google Gemini（OpenAI 兼容格式）
- **RAG 框架**：LangChain + ChromaDB — 向量化存储与余弦相似度检索
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2（本地运行，384维向量）
- **Tool Use**：OpenAI Function Calling 协议 — JSON Schema 工具定义，多轮对话循环
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
# 安装 biopython
pip install biopython

# 运行抓取脚本（约 3-5 分钟，生成 medical_knowledge_base.json）
python build_knowledge_base.py
```

> 不运行此步骤也可以正常使用，系统会自动使用内置的 15 条知识片段作为 RAG 知识库。

### 6. 启动

```bash
streamlit run ai_medical_imaging.py
```

访问 `http://localhost:8501`

---

## 📖 使用流程

1. **选择模型**：侧边栏切换本地 Ollama 或云端 API，云端模式选择平台预设
2. **上传影像**：支持 JPG、PNG（X光、CT、MRI、超声等）
3. **开始分析**：点击"🔍 开始分析"，云端模式约 10-20 秒完成全流程
4. **查看报告**：
   - 影像概览（类型 / 部位 / 严重程度 / 可信度）
   - 主要发现 & 图像质量 & 主要诊断 & 鉴别诊断 & 临床建议
   - 患者友好说明（通俗中文解读）
   - **🔧 Tool Use Agent 综合建议**（云端模式，模型自主调用工具后的汇总）
   - RAG 知识库参考（可调节 top-k）
5. **导出报告**：下载 JSON 或 PDF 格式

---

## 🔑 支持的云端平台

| 平台 | 推荐模型 | Base URL | 免费额度 |
|------|---------|----------|---------|
| 阿里百炼 | `qwen-vl-max` | `dashscope.aliyuncs.com/compatible-mode/v1` | 新用户100万tokens |
| 硅基流动 | `Pro/Qwen/Qwen2.5-VL-7B-Instruct` | `api.siliconflow.cn/v1` | 注册赠送（视觉收费） |
| Google Gemini | `gemini-2.5-flash` | `generativelanguage.googleapis.com/v1beta/openai` | 1500次/天（需代理） |
| 自定义 | 任意 OpenAI 兼容模型 | 填入对应地址 | 取决于平台 |

---

## 📁 项目结构

```
ai-medical-agent/
├── ai_medical_imaging_enhance.py        # 主程序（Streamlit 应用）
├── build_knowledge_base.py      # PubMed 文献抓取脚本（一次性运行）
├── medical_knowledge_base.json  # RAG 知识库原始数据（PubMed 文献摘要）
├── requirements.txt             # Python 依赖
├── assets/
│   └── demo_screenshot.png      # 演示截图
└── README.md
```

> `medical_kb_store/`（ChromaDB向量数据库）由程序启动时自动生成，不纳入版本控制。

---

## 💡 面试技术要点

| 技术点 | 在本项目中的体现 |
|-------|---------------|
| RAG | LangChain + ChromaDB，PubMed文献嵌入，余弦相似度检索 |
| Function Calling | OpenAI 协议 tool_calls 循环，两个工具：RAG检索+网络搜索 |
| Prompt Engineering | 结构化JSON输出，四层repair_json容错，temperature=0.1 |
| 多模态模型 | 图像base64编码，image_url格式传入，LLaVA/Qwen-VL |
| Agent 设计 | 感知→推理→工具调用→综合输出的完整Agent循环 |
| 工程实践 | .env密钥管理，@st.cache_resource缓存，多平台兼容设计 |

---

## 🤝 致谢

基于 [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) 二次开发，在原有基础上扩展了 RAG、多平台 API、Tool Use、本地模型支持、中文化界面等功能。