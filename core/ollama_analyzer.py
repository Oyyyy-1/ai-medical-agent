"""
core/ollama_analyzer.py
=======================
本地 Ollama 模型调用模块。

Ollama 地址解决方案：
  直接运行（streamlit run app.py）：
    OLLAMA_HOST 未设置 → 默认 127.0.0.1
  Docker 容器内运行：
    容器的 127.0.0.1 是容器自己，访问不到宿主机
    docker-compose.yml 注入 OLLAMA_HOST=host.docker.internal
    host.docker.internal 是 Docker Desktop 提供的特殊域名，指向宿主机
  两种方式共用同一套代码，互不干扰。
"""

import os
import re
import json
import base64
import requests
from datetime import datetime

# 从环境变量读取 Ollama 主机地址，默认 127.0.0.1
# Docker 运行时由 docker-compose.yml 注入 host.docker.internal
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
_OLLAMA_BASE = f"http://{_OLLAMA_HOST}:11434"


def get_ollama_base_url() -> str:
    """返回当前 Ollama 地址，供 app.py 侧边栏检测用。"""
    return _OLLAMA_BASE


def _repair_json(text: str) -> dict:
    """四层策略修复模型输出的不规范 JSON。"""
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
        chunk = re.sub(r'(?<=")(\n)(?=[^"]*")', ' ', chunk)
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
        "key_findings": ["模型返回了非JSON格式，见下方原始输出"],
        "abnormalities": [], "primary_diagnosis": "解析失败",
        "differential_diagnoses": [], "severity": "Unknown",
        "confidence_level": "Low", "patient_explanation": text,
        "recommendations": ["请咨询专业医疗人员"]
    }


def analyze_with_ollama(image_bytes: bytes, model: str = "moondream:latest") -> dict:
    """调用本地 Ollama 模型分析医学图像。"""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    analysis_prompt = """Analyze this medical image. Reply ONLY with a JSON object.
All text values must be in Chinese.
{
  "image_type": "影像类型",
  "anatomical_region": "检查部位",
  "image_quality": "图像质量",
  "key_findings": ["发现1"],
  "abnormalities": ["异常发现或：未见明显异常"],
  "primary_diagnosis": "主要诊断",
  "differential_diagnoses": ["鉴别诊断"],
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
        resp = requests.post(f"{_OLLAMA_BASE}/api/chat", json=payload, timeout=300)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
        raw_text = resp.json()["message"]["content"]
        result = _repair_json(raw_text)
        result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["model_used"] = f"Ollama {model} (Local)"
        return result
    except requests.exceptions.Timeout:
        return {
            "error": "timeout", "image_type": "Timeout", "anatomical_region": "N/A",
            "image_quality": "超时", "key_findings": ["300秒内未完成"],
            "abnormalities": [], "primary_diagnosis": "分析超时",
            "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
            "patient_explanation": f"模型 [{model}] 超时，建议压缩图像后重试。",
            "recommendations": ["压缩图像到500KB以下", "关闭其他占内存程序"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "error": str(e), "image_type": "Error", "anatomical_region": "Error",
            "image_quality": "分析失败", "key_findings": [f"错误: {str(e)}"],
            "abnormalities": [], "primary_diagnosis": "分析失败",
            "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
            "patient_explanation": f"本地分析失败：{str(e)}",
            "recommendations": [f"运行: ollama pull {model}", "确认 Ollama 已启动"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }