"""
云端多平台 API 调用模块（OpenAI 兼容格式）。

支持所有实现了 /v1/chat/completions 接口的平台：
  阿里百炼、硅基流动、火山方舟、Gemini、OpenAI 官方、任意中转镜像。

设计原则：不写死任何平台，base_url 和 model 全由调用方传入。
"""

import re
import json
import base64
import requests
from datetime import datetime


def _repair_json(text: str) -> dict:
    """
    修复云端模型返回的不规范 JSON（去除 markdown 代码块、修复语法）。
    云端大模型比本地小模型稳定，但仍可能输出 ```json ... ``` 包裹。
    """
    # 去除 markdown 代码块标记
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


def analyze_with_cloud(
    image_bytes: bytes,
    api_key: str,
    model: str = "qwen-vl-max",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
) -> dict:
    """
    调用云端视觉模型分析医学图像。

    参数：
      image_bytes : 图像二进制数据
      api_key     : 平台 API Key
      model       : 模型名称
      base_url    : 平台 Base URL（不含 /chat/completions）

    返回：
      结构化报告字典
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    analysis_prompt = """你是一位资深放射科医生，请用中文分析这张医学影像图像。
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

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": analysis_prompt}
        ]}],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    try:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
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
        return {
            "error": str(e), "image_type": "Error", "anatomical_region": "Error",
            "image_quality": "分析失败", "key_findings": [f"错误: {str(e)}"],
            "abnormalities": [], "primary_diagnosis": "分析失败",
            "differential_diagnoses": [], "severity": "Unknown", "confidence_level": "Low",
            "patient_explanation": f"云端分析失败：{str(e)}",
            "recommendations": ["检查 API Key 和 Base URL", "确认账户有余额"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }