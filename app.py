"""
app.py  —  AI 医学影像智能诊断系统
====================================
入口文件，只负责 Streamlit UI 组装。
所有业务逻辑在 core/ 包中：
  core/models.py          Pydantic 数据模型
  core/rag.py             RAG 知识库
  core/ollama_analyzer.py 本地模型调用
  core/cloud_analyzer.py  云端 API 调用
  core/tool_use.py        Function Calling Agent
  core/workflow.py        LangGraph 工作流
  core/pdf_report.py      PDF 生成

运行：
  streamlit run app.py
"""

import os
import io
import json
import requests
from datetime import datetime

# ---- .env 密钥加载（必须在 streamlit 之前）----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image as PILImage

# ---- 从 core 包导入所有功能 ----
from core.rag          import build_rag_knowledge_base, search_medical_knowledge, RAG_AVAILABLE
from core.pdf_report   import generate_pdf_report, PDF_AVAILABLE
from core.workflow     import run_workflow, LANGGRAPH_AVAILABLE


# ============================================================
# UI 工具函数
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

    # ── 主体两栏 ──
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

    # ── 患者说明 ──
    st.markdown("#### 👤 患者友好说明")
    with st.expander("展开查看通俗解读", expanded=True):
        st.markdown(
            f"<div style='background:#1e3a5f;border-left:4px solid #4a9eff;"
            f"padding:12px 16px;border-radius:4px;line-height:1.8;'>"
            f"{report.get('patient_explanation', 'N/A')}</div>",
            unsafe_allow_html=True
        )

    # ── Tool Use Agent 结果 ──
    if tool_use_result and tool_use_result.get("summary"):
        st.markdown("#### 🔧 Tool Use Agent 综合建议")
        tool_labels = {"search_rag_knowledge": "🗄️ RAG知识库", "search_medical_guidelines": "🌐 网络搜索"}
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

    # ── RAG 知识库参考 ──
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
# 主程序
# ============================================================

@st.cache_resource
def _init_rag():
    """RAG 知识库初始化（缓存，整个会话只执行一次）。"""
    return build_rag_knowledge_base()


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
            os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY")
        ),
        "cloud_model":    os.getenv("CLOUD_MODEL", "qwen-vl-max"),
        "cloud_base_url": os.getenv("CLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "analysis_history":  [],
        "current_report":    None,
        "current_rag_context": [],
        "current_tool_use":  None,
        "workflow_path":     "",
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
            st.success("✅ 本地 Ollama，无需 API Key")
            try:
                resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
                local_models = [m["name"] for m in resp.json().get("models", [])]
                vision_kw = ["llava", "moondream", "vision", "phi3"]
                vision_models = [m for m in local_models if any(k in m.lower() for k in vision_kw)]
                sorted_models = vision_models + [m for m in local_models if m not in vision_models]
                st.success(f"✅ 找到 {len(local_models)} 个模型")
                ollama_model = st.selectbox("选择视觉模型", sorted_models) if sorted_models else None
                if ollama_model and ollama_model not in vision_models:
                    st.warning("⚠️ 该模型可能不支持图像输入")
            except requests.exceptions.ConnectionError:
                st.error("❌ Ollama 服务未运行\n请执行 `ollama serve`")
                ollama_model = None
            except Exception as e:
                st.warning(f"⚠️ 获取模型列表失败：{e}")
                ollama_model = None
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
                "自定义": {"base_url": "", "model": "", "key_hint": "API Key", "note": "任意 OpenAI 兼容平台"},
            }
            preset = st.selectbox("平台预设", list(PRESETS.keys()), index=0)
            p = PRESETS[preset]
            st.caption(p["note"])

            cloud_base_url = st.text_input("Base URL", value=st.session_state.cloud_base_url
                                           if preset == "自定义" else p["base_url"])
            cloud_model    = st.text_input("模型名称", value=st.session_state.cloud_model
                                           if preset == "自定义" else p["model"])
            api_key_input  = st.text_input("API Key", type="password",
                                           value=st.session_state.google_api_key or "",
                                           placeholder=p["key_hint"])

            st.session_state.google_api_key = api_key_input or st.session_state.google_api_key
            st.session_state.cloud_base_url = cloud_base_url
            st.session_state.cloud_model    = cloud_model

            if st.session_state.google_api_key:
                st.success("✅ API Key 已配置")
            else:
                st.warning("⚠️ 请填入 API Key")

        st.divider()

        # ── LangGraph 状态 ──
        st.markdown("### 🔀 工作流引擎")
        if LANGGRAPH_AVAILABLE:
            st.success("✅ LangGraph 已就绪\n有向图状态机（4节点+条件边）")
        else:
            st.warning("⚠️ LangGraph 未安装\n`pip install langgraph==0.2.73`")

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
                with st.expander(f"#{len(st.session_state.analysis_history)-i} {hist['timestamp']}"):
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
            with st.spinner("🔄 LangGraph 工作流运行中..." if not use_ollama
                            else "🔄 本地模型分析中，CPU推理约 1-3 分钟..."):
                img_io = io.BytesIO()
                image.convert("RGB").save(img_io, format="PNG")
                img_bytes = img_io.getvalue()

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

            if tool_use_result:
                report["tool_use_summary"] = tool_use_result.get("summary", "")
                report["tools_called"]      = tool_use_result.get("tools_called", [])

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

        # ── 显示报告 ──
        if st.session_state.current_report:
            st.divider()
            st.markdown("## 📋 影像分析报告")

            wf_path = st.session_state.get("workflow_path", "")
            if wf_path:
                is_deep = "深度检索" in wf_path
                st.markdown(
                    f"<div style='background:{'#1a3a5c' if is_deep else '#1a2f1a'};"
                    f"border-radius:6px;padding:6px 14px;margin-bottom:8px;font-size:0.85em;'>"
                    f"{'🔴' if is_deep else '🟢'} <b>工作流路径：</b>{wf_path}"
                    f"{'（异常路径：触发深度检索）' if is_deep else '（正常路径：快速生成）'}"
                    f"</div>",
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
                    data=json.dumps(st.session_state.current_report, indent=2, ensure_ascii=False),
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
        c1.markdown("**🔀 LangGraph**\n有向图工作流\n条件分支路由")
        c2.markdown("**🗄️ RAG**\nLangChain + ChromaDB\nPubMed文献检索")
        c3.markdown("**🔧 Tool Use**\nFunction Calling\n模型自主调用工具")
        c4.markdown("**📄 结构化输出**\nPydantic + PDF\n标准报告导出")


if __name__ == "__main__":
    main()