"""
app.py  —  AI 医学影像智能诊断系统
====================================
入口文件，只负责 Streamlit UI 组装。
所有业务逻辑在 core/ 包中：
  core/models.py          Pydantic 数据模型
  core/rag.py             RAG 知识库
  core/ollama_analyzer.py 本地模型调用
  core/cloud_analyzer.py  云端 API 调用（含流式）
  core/tool_use.py        Function Calling Agent
  core/workflow.py        LangGraph 工作流
  core/pdf_report.py      PDF 生成

运行：
  streamlit run app.py

流式输出说明：
  云端模式下，影像分析阶段使用流式输出（analyze_with_cloud_stream）。
  用户几乎立刻看到模型开始输出，逐字渲染，感知延迟从10秒降至0.5秒。
  流结束后自动解析 JSON，进入 RAG + Tool Use 后续流程。
  本地 Ollama 模式暂不支持流式（moondream 等小模型不稳定）。
"""

import os
import io
import json
import requests
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image as PILImage

from core.rag            import build_rag_knowledge_base, search_medical_knowledge, RAG_AVAILABLE
from core.pdf_report     import generate_pdf_report, PDF_AVAILABLE
from core.workflow       import run_workflow, LANGGRAPH_AVAILABLE
from core.cloud_analyzer import analyze_with_cloud_stream
from core.ollama_analyzer import analyze_with_ollama, get_ollama_base_url
from core.tool_use       import analyze_with_tool_use


# ============================================================
# UI 渲染函数
# ============================================================

def render_report_ui(report: dict, rag_context: list, tool_use_result: dict = None):
    """渲染结构化诊断报告界面。"""
    if "error" in report and report.get("image_type") == "Error":
        st.error(f"❌ 分析失败：{report['error']}")
        st.info(report.get("patient_explanation", ""))
        return

    severity_map = {
        "Normal":   ("🟢", "正常",    "normal"),
        "Mild":     ("🟡", "轻度异常", "off"),
        "Moderate": ("🟠", "中度异常", "off"),
        "Severe":   ("🔴", "重度异常", "inverse"),
        "Unknown":  ("⚪", "未知",    "off"),
    }
    severity = report.get("severity", "Unknown")
    sev_icon, sev_cn, _ = severity_map.get(severity, ("⚪", severity, "off"))
    conf_cn = {"Low": "低", "Medium": "中", "High": "高"}.get(
        report.get("confidence_level", ""), report.get("confidence_level", "N/A"))

    # ── 顶部四项核心指标 ──
    st.markdown("### 📊 影像概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("影像类型",   report.get("image_type", "N/A"))
    c2.metric("检查部位",   report.get("anatomical_region", "N/A"))
    c3.metric("严重程度",   f"{sev_icon} {sev_cn}")
    c4.metric("诊断可信度", conf_cn)
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 🔬 主要影像发现")
        for f in report.get("key_findings", []):
            st.markdown(f"&nbsp;&nbsp;• {f}")
        st.markdown("#### 📐 图像质量评估")
        st.info(f"📋 {report.get('image_quality', 'N/A')}")
        abnorms = [a for a in report.get("abnormalities", [])
                   if a.lower() not in ("none detected", "none", "")]
        if abnorms:
            st.markdown("#### ⚠️ 异常发现")
            for a in abnorms:
                st.warning(f"⚠️ {a}")

    with right:
        st.markdown("#### 🩺 主要诊断")
        st.success(f"**{report.get('primary_diagnosis', 'N/A')}**")
        diffs = [d for d in report.get("differential_diagnoses", [])
                 if d.lower() not in ("none applicable", "none", "n/a", "")]
        st.markdown("#### 🗂️ 鉴别诊断")
        if diffs:
            for i, d in enumerate(diffs, 1):
                st.markdown(f"&nbsp;&nbsp;**{i}.** {d}")
        else:
            st.caption("无其他需鉴别的诊断")
        recs = report.get("recommendations", [])
        if recs:
            st.markdown("#### 💡 临床建议")
            for r in recs:
                st.markdown(f"&nbsp;&nbsp;→ {r}")

    st.divider()

    st.markdown("#### 👤 患者友好说明")
    with st.expander("展开查看通俗解读", expanded=True):
        st.markdown(
            f"<div style='background:#1e3a5f;border-left:4px solid #4a9eff;"
            f"padding:12px 16px;border-radius:4px;line-height:1.8;'>"
            f"{report.get('patient_explanation', 'N/A')}</div>",
            unsafe_allow_html=True
        )

    if tool_use_result and tool_use_result.get("summary"):
        st.markdown("#### 🔧 Tool Use Agent 综合建议")
        tool_labels = {
            "search_rag_knowledge": "🗄️ RAG知识库",
            "search_medical_guidelines": "🌐 网络搜索"
        }
        tools_called = tool_use_result.get("tools_called", [])
        if tools_called:
            badges = " &nbsp; ".join(
                f"<span style='background:#1a3a5c;border:1px solid #4a9eff;"
                f"padding:2px 10px;border-radius:12px;font-size:0.82em;'>"
                f"{tool_labels.get(t, t)}</span>" for t in tools_called
            )
            st.markdown(
                f"<div style='margin-bottom:8px;'>已调用工具：{badges}"
                f"&nbsp;&nbsp;<span style='color:#888;font-size:0.8em;'>"
                f"（{tool_use_result.get('rounds', 0)}轮对话）</span></div>",
                unsafe_allow_html=True
            )
        st.markdown(
            f"<div style='background:#1a2f1a;border-left:4px solid #66bb6a;"
            f"padding:12px 16px;border-radius:4px;line-height:1.8;'>"
            f"{tool_use_result['summary']}</div>",
            unsafe_allow_html=True
        )

    if rag_context:
        st.markdown("#### 🗄️ RAG 知识库参考")
        with st.expander(f"展开查看 {len(rag_context)} 条匹配医学知识"):
            for i, ctx in enumerate(rag_context, 1):
                st.markdown(
                    f"<div style='background:#1a2f1a;border-left:3px solid #4caf50;"
                    f"padding:10px 14px;margin-bottom:8px;border-radius:4px;font-size:0.88em;'>"
                    f"<b>参考 [{i}]</b>&nbsp;&nbsp;{ctx}</div>",
                    unsafe_allow_html=True
                )

    st.divider()
    col_m, col_t = st.columns(2)
    col_m.caption(f"🤖 分析模型：{report.get('model_used', 'N/A')}")
    col_t.caption(f"⏱ 分析完成时间：{report.get('analysis_timestamp', 'N/A')}")


# ============================================================
# 流式分析核心逻辑（云端模式专用）
# ============================================================

def run_stream_analysis(img_bytes: bytes, vectorstore, rag_k: int) -> tuple:
    """
    云端流式分析完整流程：
      1. 流式调用模型 → 实时渲染原始文本
      2. 流结束后拿到解析好的报告字典
      3. 按严重程度决定是否触发 RAG + Tool Use

    返回：(report, rag_context, tool_use_result, workflow_path)

    为什么不走 LangGraph：
      LangGraph 的 invoke() 是同步阻塞的，流式 Generator 无法嵌入节点内部
      直接在 app.py 实现流式分析，LangGraph 保留用于非流式的 Ollama 路径
    """
    api_key  = st.session_state.google_api_key or ""
    base_url = st.session_state.cloud_base_url or ""
    model    = st.session_state.cloud_model or "qwen-vl-max"

    report = None
    workflow_path = "流式分析"

    # ── Step 1：流式影像分析 ──
    st.markdown("#### 🔄 模型实时输出")
    stream_box = st.empty()   # 占位容器，用于逐字更新文本

    # 用于在 stream_box 里模拟打字效果
    # Streamlit 没有原生"逐字追加"组件，用 st.empty() + 累积文本替代
    accumulated = ""
    with st.spinner(""):     # 空 spinner，视觉上只看到文字流动
        for chunk in analyze_with_cloud_stream(img_bytes, api_key, model, base_url):
            if isinstance(chunk, str):
                # 收到文本片段：累积并刷新显示
                accumulated += chunk
                stream_box.markdown(
                    f"<div style='background:#0e1117;border:1px solid #333;"
                    f"border-radius:6px;padding:12px;font-family:monospace;"
                    f"font-size:0.85em;line-height:1.6;max-height:300px;"
                    f"overflow-y:auto;white-space:pre-wrap;'>{accumulated}▌</div>",
                    unsafe_allow_html=True
                )
            elif isinstance(chunk, dict):
                # 收到字典：流结束，拿到解析好的报告
                report = chunk

    # 流结束后用最终文本替换（去掉光标符 ▌）
    stream_box.markdown(
        f"<div style='background:#0e1117;border:1px solid #2a2a2a;"
        f"border-radius:6px;padding:12px;font-family:monospace;"
        f"font-size:0.85em;line-height:1.6;max-height:300px;"
        f"overflow-y:auto;white-space:pre-wrap;color:#aaa;'>"
        f"{accumulated}</div>",
        unsafe_allow_html=True
    )

    if report is None or report.get("image_type") == "Error":
        return report or {}, [], None, "流式分析失败"

    # ── Step 2：按严重程度决定后续路径 ──
    severity = report.get("severity", "Normal")
    is_abnormal = severity in ("Mild", "Moderate", "Severe")

    rag_context    = []
    tool_use_result = None

    if is_abnormal:
        workflow_path = "流式分析 → 异常路径 → 深度检索"
        st.info(f"⚠️ 严重程度：{severity}，触发深度检索路径...")

        # RAG 检索
        if vectorstore and RAG_AVAILABLE:
            q = f"{report.get('primary_diagnosis','')} {report.get('image_type','')}".strip()
            if q:
                rag_context = search_medical_knowledge(vectorstore, q, k=rag_k)

        # Tool Use Agent
        with st.spinner("🔧 Tool Use Agent 调用工具中..."):
            tool_use_result = analyze_with_tool_use(
                diagnosis   = report.get("primary_diagnosis", "未知诊断"),
                image_type  = report.get("image_type", "未知影像"),
                api_key     = api_key,
                base_url    = base_url,
                model       = model,
                vectorstore = vectorstore,
            )
        if tool_use_result:
            report["tool_use_summary"] = tool_use_result.get("summary", "")
            report["tools_called"]     = tool_use_result.get("tools_called", [])
    else:
        workflow_path = "流式分析 → 正常路径 → 基础RAG"
        # 正常路径：只做基础 RAG，不调 Tool Use
        if vectorstore and RAG_AVAILABLE:
            q = f"{report.get('primary_diagnosis','')} {report.get('image_type','')}".strip()
            if q:
                rag_context = search_medical_knowledge(vectorstore, q, k=rag_k)

    return report, rag_context, tool_use_result, workflow_path


# ============================================================
# RAG 初始化缓存
# ============================================================

@st.cache_resource
def _init_rag():
    return build_rag_knowledge_base()


# ============================================================
# 主程序
# ============================================================

def main():
    st.set_page_config(
        page_title="AI 医学影像诊断",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Session State 初始化 ──
    defaults = {
        "google_api_key": (
            os.getenv("DASHSCOPE_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or
            os.getenv("GEMINI_API_KEY")    or os.getenv("CLOUD_API_KEY")
        ),
        "cloud_model":         os.getenv("CLOUD_MODEL", "qwen-vl-max"),
        "cloud_base_url":      os.getenv("CLOUD_BASE_URL",
                                         "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "use_streaming":       True,    # 流式开关，默认开启
        "analysis_history":    [],
        "current_report":      None,
        "current_rag_context": [],
        "current_tool_use":    None,
        "workflow_path":       "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── RAG 初始化 ──
    vectorstore, rag_source_desc = None, "未初始化"
    if RAG_AVAILABLE:
        with st.spinner("🗄️ 初始化 RAG 知识库（首次启动约需30秒）..."):
            vectorstore, rag_source_desc = _init_rag()

    # ================================================================
    # 侧边栏
    # ================================================================
    with st.sidebar:
        st.title("⚙️ 配置")
        st.divider()

        # ── 模型选择 ──
        st.markdown("### 🤖 模型选择")
        model_choice = st.radio(
            "选择分析模式",
            options=["🖥️ 本地模型 (Ollama)", "☁️ 云端模型 (API)"],
            index=0
        )
        use_ollama = "本地" in model_choice

        if use_ollama:
            # ── Ollama 服务检测 ──────────────────────────────────────
            # 每次侧边栏渲染都重新发请求，不缓存检测结果
            # 地址从 ollama_analyzer 的环境变量读取：
            #   直接运行：127.0.0.1:11434
            #   Docker：  host.docker.internal:11434
            ollama_base = get_ollama_base_url()

            # 重新检测按钮：点击后 st.rerun() 强制重跑整个脚本
            # 解决"先开 web 后开 ollama，刷新检测不到"的问题
            col_detect, col_btn = st.columns([3, 1])
            with col_detect:
                st.caption(f"🔗 连接地址：`{ollama_base}`")
            with col_btn:
                if st.button("🔄", help="重新检测 Ollama 服务"):
                    st.rerun()

            ollama_model = None
            try:
                resp = requests.get(f"{ollama_base}/api/tags", timeout=3)
                resp.raise_for_status()
                local_models = [m["name"] for m in resp.json().get("models", [])]
                vision_kw     = ["llava", "moondream", "vision", "phi3"]
                vision_models = [m for m in local_models
                                 if any(k in m.lower() for k in vision_kw)]
                sorted_models = vision_models + [m for m in local_models
                                                 if m not in vision_models]

                if sorted_models:
                    st.success(f"✅ Ollama 运行中，找到 {len(local_models)} 个模型")
                    ollama_model = st.selectbox("选择视觉模型", sorted_models)
                    if ollama_model not in vision_models:
                        st.warning("⚠️ 该模型可能不支持图像输入")
                else:
                    st.warning("⚠️ Ollama 运行中但未安装任何模型")
                    st.code("ollama pull moondream")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ 连接不到 Ollama（{ollama_base}）")
                if "host.docker.internal" in ollama_base:
                    st.info("Docker 模式：请确保宿主机已运行 `ollama serve`")
                else:
                    st.info("请先运行 `ollama serve`，然后点击右上角 🔄 重新检测")
            except Exception as e:
                st.warning(f"⚠️ 检测失败：{e}")
                st.button("点击重新检测", on_click=st.rerun)
        else:
            ollama_model = None
            st.markdown("**☁️ 云端 API 配置**")

            PRESETS = {
                "阿里百炼 DashScope": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-vl-max", "key_hint": "sk-...",
                    "note": "国内直连，新用户有免费额度 | bailian.console.aliyun.com"
                },
                "硅基流动 SiliconFlow": {
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct", "key_hint": "sk-...",
                    "note": "国内直连，视觉模型需付费额度 | cloud.siliconflow.cn"
                },
                "火山方舟 豆包": {
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "model": "doubao-vision-pro-32k", "key_hint": "填入火山方舟 Key",
                    "note": "国内直连，新用户有免费额度 | console.volcengine.com"
                },
                "Gemini (需代理)": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "model": "gemini-2.0-flash", "key_hint": "AIzaSy...",
                    "note": "需要 VPN | aistudio.google.com，每天免费1500次"
                },
                "自定义": {
                    "base_url": "", "model": "", "key_hint": "API Key",
                    "note": "任意 OpenAI 兼容平台"
                },
            }
            preset = st.selectbox("平台预设", list(PRESETS.keys()), index=0)
            p = PRESETS[preset]
            st.caption(p["note"])

            cloud_base_url = st.text_input(
                "Base URL",
                value=st.session_state.cloud_base_url if preset == "自定义" else p["base_url"]
            )
            cloud_model = st.text_input(
                "模型名称",
                value=st.session_state.cloud_model if preset == "自定义" else p["model"]
            )
            api_key_input = st.text_input(
                "API Key", type="password",
                value=st.session_state.google_api_key or "",
                placeholder=p["key_hint"]
            )

            st.session_state.google_api_key = api_key_input or st.session_state.google_api_key
            st.session_state.cloud_base_url = cloud_base_url
            st.session_state.cloud_model    = cloud_model

            if st.session_state.google_api_key:
                st.success("✅ API Key 已配置")
            else:
                st.warning("⚠️ 请填入 API Key")

            # ── 流式输出开关 ──
            st.divider()
            st.markdown("### ⚡ 流式输出")
            st.session_state.use_streaming = st.toggle(
                "启用流式输出（Streaming）",
                value=st.session_state.use_streaming,
                help=(
                    "开启后模型逐字输出，几乎立刻看到第一个字（感知延迟 < 1秒）。\n"
                    "关闭后等待完整响应再显示（等待约10-20秒）。\n"
                    "仅云端模式有效，本地 Ollama 暂不支持流式。"
                )
            )
            if st.session_state.use_streaming:
                st.caption("✅ 流式已开启 — 模型输出实时渲染")
            else:
                st.caption("⏸ 流式已关闭 — 等待完整响应")

        st.divider()

        # ── LangGraph 状态 ──
        st.markdown("### 🔀 工作流引擎")
        if LANGGRAPH_AVAILABLE:
            st.success("✅ LangGraph 已就绪\n有向图状态机（4节点+条件边）")
        else:
            st.warning("⚠️ LangGraph 未安装\n`pip install langgraph==0.2.73`")
        if not use_ollama and st.session_state.use_streaming:
            st.caption("ℹ️ 流式模式下直接执行，不经过 LangGraph 图")

        st.divider()

        # ── RAG 状态 ──
        st.markdown("### 🗄️ RAG 知识库")
        if vectorstore:
            st.success(f"✅ {rag_source_desc}")
            rag_k = st.slider("检索条数 (top-k)", 1, 8, 3)
        else:
            rag_k = 3
            st.warning("⚠️ RAG 未就绪")

        st.divider()

        # ── 历史记录 ──
        if st.session_state.analysis_history:
            st.markdown("### 📂 历史记录")
            st.caption(f"共 {len(st.session_state.analysis_history)} 条")
            for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(
                    f"#{len(st.session_state.analysis_history)-i} {hist['timestamp']}"
                ):
                    st.write(f"**诊断**: {hist['diagnosis']}")
                    st.write(f"**严重度**: {hist['severity']}")
            if st.button("🗑️ 清除历史"):
                st.session_state.analysis_history = []
                st.rerun()

        st.divider()
        st.warning("⚠️ **免责声明**\n本工具仅供教育和研究目的，所有结果须由专业医疗人员审核。")

    # ================================================================
    # 主内容区
    # ================================================================
    st.title("🏥 AI 医学影像智能诊断系统")

    # 副标题根据当前模式动态变化
    if not use_ollama and st.session_state.use_streaming:
        st.markdown("*⚡ 流式输出 · LangGraph 工作流 · RAG医学知识库 · Function Calling · 结构化报告*")
    else:
        st.markdown("*LangGraph 工作流 · RAG医学知识库 · Function Calling · 结构化报告*")
    st.divider()

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "📁 上传医学影像图像",
            type=["jpg", "jpeg", "png"],
            help="支持 JPG/PNG，DICOM 请先转换为 PNG"
        )
    with col_info:
        st.markdown("**支持的影像类型**")
        st.markdown("• X-Ray（胸片、骨骼）\n• MRI\n• CT Scan\n• Ultrasound\n• 其他放射影像")

    if uploaded_file is not None:
        image = PILImage.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            analyze_btn = st.button(
                "🔍 开始分析", type="primary", use_container_width=True,
                disabled=(
                    (use_ollama and not ollama_model) or
                    (not use_ollama and not st.session_state.google_api_key)
                )
            )

        if use_ollama and not ollama_model:
            st.warning("请确保 Ollama 服务运行并已安装视觉模型")
        elif not use_ollama and not st.session_state.google_api_key:
            st.warning("请在侧边栏填入 API Key")

        # ── 点击分析 ──
        if analyze_btn:
            img_io = io.BytesIO()
            image.convert("RGB").save(img_io, format="PNG")
            img_bytes = img_io.getvalue()

            # ── 路径选择 ──────────────────────────────────────────
            # 云端 + 流式开启 → run_stream_analysis（实时渲染）
            # 云端 + 流式关闭 → run_workflow（LangGraph，一次性返回）
            # 本地 Ollama     → run_workflow（LangGraph，CPU推理）
            # ─────────────────────────────────────────────────────
            use_stream = not use_ollama and st.session_state.use_streaming

            if use_stream:
                # 流式路径：直接在主界面实时渲染，不用 spinner 遮挡
                st.markdown("---")
                st.markdown("## 📋 影像分析报告")
                st.caption("⚡ 流式模式 — 模型正在实时生成分析结果...")

                report, rag_context, tool_use_result, workflow_path = run_stream_analysis(
                    img_bytes, vectorstore, rag_k
                )
            else:
                # 非流式路径：LangGraph 工作流，spinner 等待
                spinner_msg = (
                    "🔄 本地模型分析中，CPU推理约 1-3 分钟，请耐心等待..."
                    if use_ollama else
                    "🔄 LangGraph 工作流运行中，约 10-20 秒..."
                )
                with st.spinner(spinner_msg):
                    report, rag_context, tool_use_result, workflow_path = run_workflow(
                        image_bytes  = img_bytes,
                        use_ollama   = use_ollama,
                        ollama_model = ollama_model or "",
                        api_key      = st.session_state.google_api_key or "",
                        base_url     = st.session_state.cloud_base_url or "",
                        model        = st.session_state.cloud_model or "qwen-vl-max",
                        vectorstore  = vectorstore,
                        rag_k        = rag_k,
                    )

            # tool_use 结果写入 report
            if tool_use_result:
                report["tool_use_summary"] = tool_use_result.get("summary", "")
                report["tools_called"]      = tool_use_result.get("tools_called", [])

            # 保存到 Session State
            st.session_state.current_report       = report
            st.session_state.current_rag_context  = rag_context
            st.session_state.current_tool_use     = tool_use_result
            st.session_state.workflow_path        = workflow_path
            st.session_state.analysis_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename":  uploaded_file.name,
                "diagnosis": report.get("primary_diagnosis", "N/A"),
                "severity":  report.get("severity", "N/A"),
                "model":     report.get("model_used", "Unknown")
            })

        # ── 显示结构化报告 ──
        if st.session_state.current_report:
            # 流式模式下标题和路径条已在 run_stream_analysis 前输出，这里补充非流式的头部
            if not (analyze_btn and not use_ollama and st.session_state.use_streaming):
                st.divider()
                st.markdown("## 📋 影像分析报告")

            # 工作流路径条
            wf_path = st.session_state.get("workflow_path", "")
            if wf_path:
                is_deep = "深度检索" in wf_path or "异常" in wf_path
                bg_color = "#1a3a5c" if is_deep else "#1a2f1a"
                icon     = "🔴" if is_deep else "🟢"
                label    = "（异常路径：触发深度检索）" if is_deep else "（正常路径：快速生成）"
                st.markdown(
                    f"<div style='background:{bg_color};border-radius:6px;"
                    f"padding:6px 14px;margin-bottom:8px;font-size:0.85em;'>"
                    f"{icon} <b>执行路径：</b>{wf_path} {label}</div>",
                    unsafe_allow_html=True
                )

            render_report_ui(
                st.session_state.current_report,
                st.session_state.current_rag_context,
                st.session_state.current_tool_use
            )

            # ── 下载按钮 ──
            st.divider()
            col_dl1, col_dl2, _ = st.columns(3)
            with col_dl1:
                st.download_button(
                    "📥 下载 JSON 报告",
                    data=json.dumps(
                        st.session_state.current_report, indent=2, ensure_ascii=False
                    ),
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            with col_dl2:
                if PDF_AVAILABLE:
                    pdf_bytes = generate_pdf_report(
                        st.session_state.current_report,
                        st.session_state.current_rag_context
                    )
                    if pdf_bytes:
                        st.download_button(
                            "📄 下载 PDF 报告",
                            data=pdf_bytes,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
    else:
        st.info("👆 请上传医学影像图像开始分析")
        st.markdown("---")
        st.markdown("### 🛠️ 技术架构")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**⚡ 流式输出**\nSSE 逐token渲染\n感知延迟<1秒")
        c2.markdown("**🔀 LangGraph**\n有向图工作流\n条件分支路由")
        c3.markdown("**🗄️ RAG**\nLangChain + ChromaDB\nPubMed文献检索")
        c4.markdown("**🔧 Tool Use**\nFunction Calling\n模型自主调用工具")


if __name__ == "__main__":
    main()