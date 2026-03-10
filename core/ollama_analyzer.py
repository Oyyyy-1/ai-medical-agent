"""
本地 Ollama 模型调用模块。

设计决策：直接用 requests 调 Ollama REST API，完全绕过 ollama Python SDK。
原因：SDK 版本升级后字段名变化导致兼容问题，直接 HTTP 调用等价于 curl，
最稳定，和 SDK 版本解耦。
"""

import re
import json
import base64
import requests
from datetime import datetime


def _repair_json(text: str) -> dict:
    """
    四层策略修复模型输出的不规范 JSON。

    小模型（moondream等）常见问题：
      - 末尾缺少 }
      - 列表末尾多余逗号
      - 字符串内含未转义换行符

    策略顺序：直接解析 → 提取{} → 修复语法 → 原文兜底
    """
    # 层1：直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 层2：提取第一个{到最后一个}
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except Exception:
            pass

        # 层3：修复常见语法错误
        chunk = re.sub(r',\s*([}\]])', r'\1', chunk)          # 末尾多余逗号
        chunk = re.sub(r'(?<=")(\n)(?=[^"]*")', ' ', chunk)   # 字符串内换行
        open_b, close_b = chunk.count("{"), chunk.count("}")
        if open_b > close_b:
            chunk += "}" * (open_b - close_b)                  # 补缺失的 }
        try:
            return json.loads(chunk)
        except Exception:
            pass

    # 层4：兜底，把原文塞入 patient_explanation
    return {
        "image_type": "Unknown", "anatomical_region": "Unknown",
        "image_quality": "Unable to parse response",
        "key_findings": ["模型返回了非JSON格式，见下方原始输出"],
        "abnormalities": [], "primary_diagnosis": "解析失败",
        "differential_diagnoses": [], "severity": "Unknown",
        "confidence_level": "Low",
        "patient_explanation": text,
        "recommendations": ["请咨询专业医疗人员"]
    }


def analyze_with_ollama(image_bytes: bytes, model: str = "moondream:latest") -> dict:
    """
    使用本地 Ollama 模型分析医学图像。

    参数：
      image_bytes : 图像二进制数据
      model       : Ollama 模型名，如 moondream:latest / llava:7b

    返回：
      结构化报告字典
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    analysis_prompt = """Analyze this medical image. Reply with ONLY a JSON object, no other text.
All text values must be in Chinese (中文).

{
  "image_type": "影像类型",
  "anatomical_region": "检查部位",
  "image_quality": "图像质量：好/中/差",
  "key_findings": ["发现1", "发现2"],
  "abnormalities": ["异常发现，或：未见明显异常"],
  "primary_diagnosis": "主要诊断",
  "differential_diagnoses": ["鉴别诊断1"],
  "severity": "Normal or Mild or Moderate or Severe",
  "confidence_level": "Low or Medium or High",
  "patient_explanation": "通俗解释",
  "recommendations": ["建议1"]
}"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": analysis_prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024}
    }

    try:
        resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=300)
        if resp.status_code != 200:
            raise Exception(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
        raw_text = resp.json()["message"]["content"]
        result = _repair_json(raw_text)
        result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["model_used"] = f"Ollama {model} (Local)"
        return result

    except requests.exceptions.Timeout:
        return {
            "error": "timeout", "image_type": "Timeout", "anatomical_region": "N/A",
            "image_quality": "超时", "key_findings": ["300秒内未完成分析"],
            "abnormalities": [], "primary_diagnosis": "分析超时",
            "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
            "patient_explanation": f"模型 [{model}] 超时。建议：关闭其他程序、压缩图像后重试。",
            "recommendations": ["压缩图像到500KB以下", "关闭其他占内存的程序", "再试一次"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "error": str(e), "image_type": "Error", "anatomical_region": "Error",
            "image_quality": "分析失败", "key_findings": [f"错误: {str(e)}"],
            "abnormalities": [], "primary_diagnosis": "分析失败",
            "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
            "patient_explanation": f"本地分析失败：{str(e)}\n请确认 Ollama 服务已启动，模型已安装。",
            "recommendations": [f"运行: ollama pull {model}", "确认 Ollama 服务已启动: ollama serve"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }