"""
Function Calling / Tool Use Agent 模块。

核心流程（OpenAI Function Calling 协议）：
  1. 定义工具 JSON Schema，告知模型可用工具及参数
  2. 第一次请求：messages + tools → 模型返回 tool_calls
  3. 本地执行工具，获取结果
  4. 把 role="tool" 的结果追加到对话历史
  5. 第二次请求：含工具结果的完整历史 → 模型综合输出
  6. 重复直到 finish_reason == "stop"

为什么维护完整对话历史：
  大模型无状态，每次请求必须携带所有上下文，
  包括 tool_calls 和 tool 结果，模型才知道"上一步做了什么"。
"""

import json
import requests
from core.rag import search_medical_knowledge, RAG_AVAILABLE


# ── 工具定义（JSON Schema 格式）─────────────────────────────

TOOL_RAG_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_rag_knowledge",
        "description": (
            "从本地医学影像 RAG 知识库中检索与诊断相关的参考知识。"
            "当需要查找某种疾病的影像学特征、诊断标准或鉴别要点时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，如 '肺炎 胸片' 或 'pneumonia chest X-ray'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回前k条结果，默认3，最大5",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_medical_guidelines",
        "description": (
            "在网络上搜索最新医学指南、临床建议或相关研究。"
            "当需要了解疾病的最新治疗建议或循证医学证据时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索词，如 '肺结节随访指南 Fleischner'"
                }
            },
            "required": ["query"]
        }
    }
}


def _execute_tool(tool_name: str, tool_args: dict, vectorstore) -> str:
    """
    本地执行工具调用，返回结果字符串。

    这是 Tool Use 循环的执行层：
    模型说"我要调用 search_rag_knowledge"，
    这里真正执行并把结果返回给模型。
    """
    if tool_name == "search_rag_knowledge":
        query = tool_args.get("query", "")
        top_k = min(tool_args.get("top_k", 3), 5)
        if vectorstore is None or not RAG_AVAILABLE:
            return "RAG知识库暂不可用"
        results = search_medical_knowledge(vectorstore, query, k=top_k)
        if not results:
            return f"未找到与'{query}'相关的知识"
        formatted = "\n\n".join([f"[参考{i+1}] {r}" for i, r in enumerate(results)])
        return f"RAG知识库检索结果（{len(results)}条）：\n\n{formatted}"

    elif tool_name == "search_medical_guidelines":
        query = tool_args.get("query", "")
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                timeout=10
            )
            data = resp.json()
            results = []
            if data.get("AbstractText"):
                results.append(f"摘要：{data['AbstractText']}")
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"])
            return ("网络搜索结果：\n\n" + "\n\n".join(results)
                    if results else f"未找到'{query}'的相关网络结果")
        except Exception as e:
            return f"网络搜索失败：{str(e)}"

    return f"未知工具：{tool_name}"


def analyze_with_tool_use(
    diagnosis: str,
    image_type: str,
    api_key: str,
    base_url: str,
    model: str,
    vectorstore,
    max_rounds: int = 3
) -> dict:
    """
    Tool Use Agent 主循环。

    让模型自主决定调用 RAG 检索或网络搜索来补充诊断依据，
    最终综合所有工具结果给出临床建议。

    参数：
      diagnosis  : 主要诊断（从影像分析结果提取）
      image_type : 影像类型
      api_key    : 云端 API Key
      base_url   : API Base URL
      model      : 模型名（需支持 Function Calling）
      vectorstore: ChromaDB 实例
      max_rounds : 最大循环轮数（防无限循环）

    返回：
      {
        "summary"     : 模型综合建议（字符串）,
        "tools_called": 调用过的工具名列表（去重）,
        "rag_results" : RAG 工具返回内容列表,
        "web_results" : 网络搜索返回内容列表,
        "rounds"      : 实际循环轮数
      }
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位资深放射科医生的AI助手。"
                "根据影像诊断结论，主动使用可用工具检索相关临床知识和最新医学指南，"
                "为诊断提供循证医学支持。请用中文回答。"
            )
        },
        {
            "role": "user",
            "content": (
                f"影像类型：{image_type}\n主要诊断：{diagnosis}\n\n"
                "请使用可用工具检索相关医学知识和临床指南，"
                "然后给出综合性临床建议总结（2-4句话）。"
            )
        }
    ]

    tools_called, rag_results, web_results = [], [], []
    final_summary = ""
    round_num = 0

    for round_num in range(max_rounds):
        try:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model, "messages": messages,
                    "tools": [TOOL_RAG_SEARCH, TOOL_WEB_SEARCH],
                    "tool_choice": "auto",
                    "max_tokens": 1024, "temperature": 0.1
                },
                timeout=60
            )

            if resp.status_code != 200:
                final_summary = f"API 请求失败（HTTP {resp.status_code}）"
                break

            choice = resp.json()["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            message = choice["message"]
            messages.append(message)  # 维护完整对话历史

            if finish_reason == "tool_calls" or message.get("tool_calls"):
                for tc in message.get("tool_calls", []):
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    tool_result = _execute_tool(name, args, vectorstore)
                    tools_called.append(name)
                    if name == "search_rag_knowledge":
                        rag_results.append(tool_result)
                    elif name == "search_medical_guidelines":
                        web_results.append(tool_result)
                    # role="tool" 是 OpenAI 协议要求的格式
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result
                    })
            else:
                final_summary = message.get("content", "")
                break

        except Exception as e:
            final_summary = f"Tool Use 执行出错：{str(e)}"
            break

    # 达到 max_rounds 仍未 stop，做最后一次无工具请求强制收尾
    if not final_summary:
        try:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": 512, "temperature": 0.1},
                timeout=30
            )
            if resp.status_code == 200:
                final_summary = resp.json()["choices"][0]["message"].get("content", "")
        except Exception:
            final_summary = "Tool Use 达到最大轮数，未能生成总结"

    return {
        "summary":      final_summary,
        "tools_called": list(set(tools_called)),
        "rag_results":  rag_results,
        "web_results":  web_results,
        "rounds":       round_num + 1
    }