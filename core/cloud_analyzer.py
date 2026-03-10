"""
云端多平台 API 调用模块（OpenAI 兼容格式）。

支持所有实现了 /v1/chat/completions 接口的平台：
  阿里百炼、硅基流动、火山方舟、Gemini、OpenAI 官方。

提供两种调用模式：
  analyze_with_cloud        普通模式，等待完整响应后返回解析好的字典
  analyze_with_cloud_stream 流式模式，逐 token yield 文本片段，
                            最后 yield 一个 dict（解析好的报告）

流式原理（SSE - Server-Sent Events）：
  请求时加 "stream": true，服务端每生成一个 token 立刻推送一行：
    data: {"choices":[{"delta":{"content":"字"},...}]}
  最后一行：data: [DONE]
  客户端用 response.iter_lines() 逐行读取，解析 delta.content 拼接。

  优点：用户几乎立刻看到第一个字，感知延迟从 10 秒降至 0.5 秒。
  难点：JSON 需完整才能解析 → 先流式展示原始文本，
        流结束后对完整文本做 repair_json，最后 yield 解析好的字典。
"""

import re
import json
import base64
import requests
from datetime import datetime
from typing import Generator


# ── JSON 修复工具 ────────────────────────────────────────────

def _repair_json(text: str) -> dict:
    """
    修复模型返回的不规范 JSON。
    云端大模型比本地稳定，但仍可能输出 ```json...``` 包裹或末尾多余逗号。
    """
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except Exception:
            pass
        chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
        open_b, close_b = chunk.count("{"), chunk.count("}")
        if open_b > close_b:
            chunk += "}" * (open_b - close_b)
        try:
            return json.loads(chunk)
        except Exception:
            pass

    return {
        "image_type": "Unknown", "anatomical_region": "Unknown",
        "image_quality": "无法解析响应",
        "key_findings": ["见下方完整分析"],
        "abnormalities": [], "primary_diagnosis": "见下方说明",
        "differential_diagnoses": [], "severity": "Unknown",
        "confidence_level": "Low", "patient_explanation": text,
        "recommendations": ["请咨询专业医疗人员"]
    }


# ── 共用提示词和请求体构建 ────────────────────────────────────

_ANALYSIS_PROMPT = """你是一位资深放射科医生，请用中文分析这张医学影像图像。
重要要求：
1. 只输出纯JSON，不要有任何多余文字、markdown标记或代码块
2. 除 severity 和 confidence_level 两个字段保持英文外，所有字段值必须用中文
3. 字符串值不能包含双引号

{
  "image_type": "影像类型（如：胸部X光/头部MRI/腹部CT）",
  "anatomical_region": "检查部位和体位",
  "image_quality": "图像质量评估",
  "key_findings": ["发现1（中文）", "发现2（中文）"],
  "abnormalities": ["异常描述，如无异常填：未见明显异常"],
  "primary_diagnosis": "主要诊断结论（中文）",
  "differential_diagnoses": ["鉴别诊断1（中文）"],
  "severity": "Normal or Mild or Moderate or Severe",
  "confidence_level": "Low or Medium or High",
  "patient_explanation": "用患者能理解的通俗中文解释",
  "recommendations": ["建议1（中文）", "建议2（中文）"]
}"""


def _build_payload(image_b64: str, model: str, stream: bool = False) -> dict:
    """构建请求体，stream 参数控制是否启用流式输出。"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": _ANALYSIS_PROMPT}
        ]}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": stream   # False = 普通模式，True = 流式模式
    }


def _error_report(msg: str) -> dict:
    """统一的错误报告格式。"""
    return {
        "error": msg, "image_type": "Error", "anatomical_region": "Error",
        "image_quality": "分析失败", "key_findings": [f"错误: {msg}"],
        "abnormalities": [], "primary_diagnosis": "分析失败",
        "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
        "patient_explanation": f"云端分析失败：{msg}",
        "recommendations": ["检查 API Key 和 Base URL", "确认账户有余额"],
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ── 普通模式 ─────────────────────────────────────────────────

def analyze_with_cloud(
    image_bytes: bytes,
    api_key: str,
    model: str = "qwen-vl-max",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
) -> dict:
    """
    普通（非流式）调用云端视觉模型。
    等待服务端生成完整响应后一次性返回解析好的字典。
    由 LangGraph workflow 节点调用。
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"

    try:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=_build_payload(image_b64, model, stream=False),
            timeout=60
        )
        if resp.status_code == 401:
            raise Exception("API Key 无效或已过期")
        elif resp.status_code in (402, 403):
            raise Exception(f"余额不足或无权限: {resp.text[:200]}")
        elif resp.status_code == 404:
            raise Exception(f"模型不存在或 Base URL 错误: {endpoint}")
        elif resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")

        raw_text = resp.json()["choices"][0]["message"]["content"]
        result = _repair_json(raw_text)
        result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["model_used"] = f"Cloud: {model}"
        return result

    except requests.exceptions.ConnectionError:
        raise Exception(f"无法连接到 {endpoint}，请检查 Base URL 或网络")
    except requests.exceptions.Timeout:
        raise Exception("请求超时（60秒），请检查网络")
    except Exception as e:
        return _error_report(str(e))


# ── 流式模式 ─────────────────────────────────────────────────

def analyze_with_cloud_stream(
    image_bytes: bytes,
    api_key: str,
    model: str = "qwen-vl-max",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
) -> Generator:
    """
    流式调用云端视觉模型，用 Python 生成器（Generator）逐步产出内容。

    yield 规则：
      - 正常情况：先 yield 多个 str（每个是一小段文本 token），
                  最后 yield 一个 dict（完整解析好的报告）
      - 出错情况：yield 一个 dict（错误报告）

    调用方（app.py）处理方式：
      full_text = ""
      for chunk in analyze_with_cloud_stream(...):
          if isinstance(chunk, str):
              full_text += chunk
              # 实时渲染文本
          elif isinstance(chunk, dict):
              report = chunk
              # 流结束，拿到解析好的报告

    为什么用生成器（Generator）而不是回调：
      生成器用 yield 暂停/恢复，调用方用 for 循环消费，
      天然适合"产生一个消费一个"的流式场景，
      比回调函数更简洁，和 Streamlit 的 st.write_stream 完美配合。

    SSE 数据格式（每行）：
      data: {"id":"...","choices":[{"delta":{"content":"字"},"finish_reason":null}]}
      data: [DONE]   ← 流结束标志
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"

    try:
        # stream=True 让 requests 不缓冲响应，保持连接打开逐行读取
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=_build_payload(image_b64, model, stream=True),
            timeout=60,
            stream=True    # requests 层面也要设 stream=True，才能逐行迭代
        )

        if resp.status_code != 200:
            yield _error_report(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return

        full_text = ""   # 累积完整响应文本，流结束后用来解析 JSON

        # iter_lines() 逐行读取 SSE 数据，自动处理换行符
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue   # 跳过空行（SSE 用空行分隔事件）

            # raw_line 是 bytes，decode 成字符串
            line = raw_line.decode("utf-8", errors="replace")

            # SSE 每行格式：data: {...} 或 data: [DONE]
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()

            # [DONE] 是流结束标志，跳出循环
            if data_str == "[DONE]":
                break

            try:
                # 解析单个 SSE chunk 的 JSON
                chunk_data = json.loads(data_str)
                # delta.content 是本次新增的文本片段（可能为 None）
                delta_content = (
                    chunk_data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta_content:
                    full_text += delta_content
                    yield delta_content   # 把文本片段 yield 给调用方实时渲染

            except json.JSONDecodeError:
                # 某个 chunk 解析失败时跳过，不影响整体流
                continue

        # 流结束，对完整文本做 JSON 解析
        if full_text:
            result = _repair_json(full_text)
            result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["model_used"] = f"Cloud(stream): {model}"
            yield result   # 最后 yield 解析好的字典
        else:
            yield _error_report("流式响应为空，请重试")

    except requests.exceptions.ConnectionError:
        yield _error_report(f"无法连接到 {endpoint}，请检查网络")
    except requests.exceptions.Timeout:
        yield _error_report("请求超时（60秒）")
    except Exception as e:
        yield _error_report(str(e))