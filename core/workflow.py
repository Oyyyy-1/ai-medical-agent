"""
LangGraph 工作流状态机模块。

图结构：
  START
    ↓
  check_quality（图像质量预检，纯本地逻辑）
    ↓
  analyze_image（调用模型分析）
    ↓
  [条件边 route_by_severity]
    ├─ normal   → generate_report（快速路径，基础RAG）
    └─ abnormal → deep_retrieval（深度检索：RAG + Tool Use）
                        ↓
                  generate_report
    ↓
  END

条件分支的价值：
  正常影像不触发 Tool Use，节省 API 调用和时间。
  异常影像才走深度检索，提供循证医学支持。
"""

import base64
from datetime import datetime
from typing import TypedDict

from core.rag import search_medical_knowledge, RAG_AVAILABLE
from core.ollama_analyzer import analyze_with_ollama
from core.cloud_analyzer import analyze_with_cloud
from core.tool_use import analyze_with_tool_use

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ── State 定义 ───────────────────────────────────────────────
class GraphState(TypedDict):
    """
    工作流共享状态（TypedDict）。

    贯穿整个图，所有节点读写同一个字典。
    节点函数只需返回修改的字段，LangGraph 自动合并到 State。
    """
    image_bytes:     bytes
    image_b64:       str
    api_key:         str
    base_url:        str
    model:           str
    use_ollama:      bool
    ollama_model:    str
    vectorstore:     object
    rag_k:           int
    quality_ok:      bool
    quality_note:    str
    report:          dict
    rag_context:     list
    tool_use_result: dict
    workflow_path:   str
    error:           str


# ── 节点函数 ─────────────────────────────────────────────────

def node_check_quality(state: GraphState) -> dict:
    """
    节点1：图像质量预检。
    用启发式规则快速判断，不调用 AI，几乎不耗时。
      - 分辨率 < 100×100 → 不合格
      - 图像近似纯色（std < 5）→ 不合格
    """
    try:
        import io
        import numpy as np
        from PIL import Image as PIL

        img = PIL.open(io.BytesIO(state["image_bytes"]))
        w, h = img.size

        if w < 100 or h < 100:
            return {
                "quality_ok": False,
                "quality_note": f"分辨率过低（{w}×{h}），请上传至少 200×200 像素的图像",
                "workflow_path": "质量检查 → 不合格"
            }

        arr = np.array(img.convert("L").resize((10, 10)))
        if arr.std() < 5:
            return {
                "quality_ok": False,
                "quality_note": "图像为纯色，请检查上传文件是否正确",
                "workflow_path": "质量检查 → 不合格"
            }

        return {
            "quality_ok": True,
            "quality_note": f"图像质量合格（{w}×{h}像素）",
            "workflow_path": "质量检查 → 合格"
        }
    except Exception as e:
        return {"quality_ok": True, "quality_note": f"质量检查跳过：{e}",
                "workflow_path": "质量检查 → 跳过"}


def node_analyze_image(state: GraphState) -> dict:
    """节点2：调用模型分析影像。质量不合格时直接返回错误报告。"""
    if not state.get("quality_ok", True):
        return {"report": {
            "error": state.get("quality_note", "图像质量不合格"),
            "image_type": "Error", "primary_diagnosis": "图像质量不合格，无法分析",
            "severity": "Unknown", "confidence_level": "Low",
            "key_findings": [state.get("quality_note", "")], "abnormalities": [],
            "differential_diagnoses": [], "recommendations": ["请重新上传清晰的医学影像"],
            "patient_explanation": state.get("quality_note", ""),
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "anatomical_region": "Unknown", "image_quality": "不合格"
        }}
    try:
        if state.get("use_ollama"):
            report = analyze_with_ollama(state["image_bytes"], model=state.get("ollama_model", "moondream:latest"))
        else:
            report = analyze_with_cloud(
                state["image_bytes"],
                api_key=state["api_key"],
                model=state["model"],
                base_url=state["base_url"]
            )
        return {"report": report}
    except Exception as e:
        return {"report": {
            "error": str(e), "image_type": "Error", "primary_diagnosis": "分析失败",
            "severity": "Unknown", "confidence_level": "Low",
            "key_findings": [f"错误：{e}"], "abnormalities": [],
            "differential_diagnoses": [], "recommendations": ["检查模型配置后重试"],
            "patient_explanation": f"分析出错：{e}",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "anatomical_region": "Unknown", "image_quality": "分析失败"
        }}


def route_by_severity(state: GraphState) -> str:
    """
    条件边路由函数：根据严重程度决定走哪条分支。
    返回字符串必须和 add_conditional_edges mapping 的 key 一致。
      Normal/Unknown/Error → "normal"（快速路径）
      Mild/Moderate/Severe → "abnormal"（深度检索）
    """
    report = state.get("report", {})
    if report.get("image_type") == "Error":
        return "normal"
    if report.get("severity") in ("Mild", "Moderate", "Severe"):
        return "abnormal"
    return "normal"


def node_deep_retrieval(state: GraphState) -> dict:
    """
    节点3a：异常影像深度检索（RAG + Tool Use）。
    仅在 route_by_severity 返回 'abnormal' 时触发。
    """
    report = state.get("report", {})
    vectorstore = state.get("vectorstore")
    rag_k = state.get("rag_k", 3)

    rag_context = []
    if vectorstore and RAG_AVAILABLE:
        q = f"{report.get('primary_diagnosis','')} {report.get('image_type','')}".strip()
        if q:
            rag_context = search_medical_knowledge(vectorstore, q, k=rag_k)

    tool_use_result = None
    if not state.get("use_ollama") and state.get("api_key"):
        tool_use_result = analyze_with_tool_use(
            diagnosis=report.get("primary_diagnosis", "未知诊断"),
            image_type=report.get("image_type", "未知影像"),
            api_key=state["api_key"],
            base_url=state["base_url"],
            model=state["model"],
            vectorstore=vectorstore,
        )
        if tool_use_result:
            report["tool_use_summary"] = tool_use_result.get("summary", "")
            report["tools_called"] = tool_use_result.get("tools_called", [])

    return {
        "rag_context": rag_context,
        "tool_use_result": tool_use_result,
        "report": report,
        "workflow_path": state.get("workflow_path", "") + " → 深度检索 → 报告生成"
    }


def node_generate_report(state: GraphState) -> dict:
    """
    节点3b：正常影像快速路径，仅做基础 RAG 检索。
    比异常路径少一次 Tool Use API 调用，响应更快。
    """
    report = state.get("report", {})
    vectorstore = state.get("vectorstore")
    rag_k = state.get("rag_k", 3)

    rag_context = []
    if vectorstore and RAG_AVAILABLE and report.get("image_type") != "Error":
        q = f"{report.get('primary_diagnosis','')} {report.get('image_type','')}".strip()
        if q:
            rag_context = search_medical_knowledge(vectorstore, q, k=rag_k)

    return {
        "rag_context": rag_context,
        "tool_use_result": None,
        "workflow_path": state.get("workflow_path", "") + " → 报告生成"
    }


# ── 图构建 ───────────────────────────────────────────────────

def build_workflow():
    """
    构建并编译 LangGraph 工作流图。

    compile() 做图合法性检查（孤立节点、入口/出口等），
    返回可执行的 CompiledGraph 对象。
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(GraphState)

    graph.add_node("check_quality",   node_check_quality)
    graph.add_node("analyze_image",   node_analyze_image)
    graph.add_node("deep_retrieval",  node_deep_retrieval)
    graph.add_node("generate_report", node_generate_report)

    graph.add_edge(START,           "check_quality")
    graph.add_edge("check_quality", "analyze_image")

    # 条件边：analyze_image → 根据 severity 分叉
    graph.add_conditional_edges(
        "analyze_image",
        route_by_severity,
        {"normal": "generate_report", "abnormal": "deep_retrieval"}
    )

    graph.add_edge("deep_retrieval",  "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# 模块级单例，避免重复编译
_WORKFLOW = None

def get_workflow():
    """懒加载工作流实例（单例模式）。"""
    global _WORKFLOW
    if _WORKFLOW is None and LANGGRAPH_AVAILABLE:
        _WORKFLOW = build_workflow()
    return _WORKFLOW


def run_workflow(
    image_bytes: bytes,
    use_ollama: bool,
    ollama_model: str,
    api_key: str,
    base_url: str,
    model: str,
    vectorstore,
    rag_k: int = 3
) -> tuple:
    """
    运行完整工作流，返回 (report, rag_context, tool_use_result, workflow_path)。

    LangGraph 不可用时自动回退到线性流程，保证功能不中断。
    """
    workflow = get_workflow()

    if workflow is None:
        # 回退：线性执行
        report = (analyze_with_ollama(image_bytes, model=ollama_model)
                  if use_ollama
                  else analyze_with_cloud(image_bytes, api_key=api_key, model=model, base_url=base_url))
        rag_context = []
        if vectorstore and RAG_AVAILABLE:
            q = f"{report.get('primary_diagnosis','')} {report.get('image_type','')}".strip()
            if q:
                rag_context = search_medical_knowledge(vectorstore, q, k=rag_k)
        return report, rag_context, None, "线性流程（LangGraph未安装）"

    initial_state: GraphState = {
        "image_bytes":     image_bytes,
        "image_b64":       base64.b64encode(image_bytes).decode("utf-8"),
        "api_key":         api_key or "",
        "base_url":        base_url or "",
        "model":           model or "",
        "use_ollama":      use_ollama,
        "ollama_model":    ollama_model or "",
        "vectorstore":     vectorstore,
        "rag_k":           rag_k,
        "quality_ok":      True,
        "quality_note":    "",
        "report":          {},
        "rag_context":     [],
        "tool_use_result": None,
        "workflow_path":   "",
        "error":           "",
    }

    final_state = workflow.invoke(initial_state)
    report = final_state.get("report", {})
    tool_use_result = final_state.get("tool_use_result")

    if tool_use_result:
        report["tool_use_summary"] = tool_use_result.get("summary", "")
        report["tools_called"] = tool_use_result.get("tools_called", [])

    return (
        report,
        final_state.get("rag_context", []),
        tool_use_result,
        final_state.get("workflow_path", "")
    )