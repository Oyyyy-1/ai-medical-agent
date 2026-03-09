"""
AI Medical Imaging Diagnosis Agent - 增强版
==========================================
技术栈：
  - Streamlit：Web界面框架
  - Ollama + LLaVA 7B：本地视觉模型（无需API Key，完全免费）
  - Google Gemini 2.0 Flash：云端模型（可选，需API Key）
  - LangChain + ChromaDB：RAG知识库（向量检索）
  - sentence-transformers：免费本地嵌入模型
  - Agno：Multi-Agent框架（启用Gemini时使用）
  - Pydantic：结构化输出数据模型
  - reportlab：PDF报告导出

运行方式：
  streamlit run ai_medical_imaging.py

依赖安装（复制到终端执行）：
  pip install streamlit pillow agno ollama langchain langchain-community
  pip install chromadb sentence-transformers pydantic reportlab
"""

# ============================================================
# 1. 导入所有依赖库
# ============================================================
import os
import json
import base64
import tempfile
import io
import requests
from datetime import datetime
from typing import Optional

# ---- 加载 .env 文件（必须在所有其他导入之前执行）----
# python-dotenv 会读取同目录下的 .env 文件，
# 把里面的 KEY=VALUE 注入到 os.environ，
# 之后用 os.getenv("KEY") 就能取到值。
# 如果 .env 不存在也不报错，只是 os.getenv 返回 None。
try:
    from dotenv import load_dotenv
    load_dotenv()   # 默认读取当前目录下的 .env 文件
except ImportError:
    pass  # python-dotenv 未安装时跳过，不影响手动输入 key 的流程

import streamlit as st
from PIL import Image as PILImage
from pydantic import BaseModel  # 用于定义结构化输出的数据模型

# ---- Ollama（本地模型） ----
# ollama 是调用本地 Ollama 服务的 Python SDK
# 确保你已经安装 Ollama 并执行了 ollama pull llava:7b
import ollama as ollama_client

# ---- Agno + Gemini（云端模型，可选） ----
# 只在用户提供了 Gemini API Key 时才会用到
try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.media import Image as AgnoImage
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False

# ---- LangChain + ChromaDB（RAG知识库） ----
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ---- PDF 导出 ----
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================
# 2. Pydantic 数据模型 —— 定义结构化诊断报告的字段
# ============================================================
class MedicalReport(BaseModel):
    """
    结构化医学诊断报告。
    使用 Pydantic BaseModel 可以：
      1. 强制 AI 返回指定字段
      2. 自动做类型校验
      3. 方便序列化为 JSON / 导出 PDF
    """
    image_type: str                       # 影像类型，如 X-ray、MRI、CT
    anatomical_region: str                # 解剖区域，如 胸部、腹部
    image_quality: str                    # 图像质量评估
    key_findings: list[str]               # 主要发现（列表）
    abnormalities: list[str]              # 异常发现（列表）
    primary_diagnosis: str                # 主要诊断
    differential_diagnoses: list[str]     # 鉴别诊断（列表）
    severity: str                         # 严重程度：Normal / Mild / Moderate / Severe
    confidence_level: str                 # AI 置信度：Low / Medium / High
    patient_explanation: str              # 患者友好的白话解释
    recommendations: list[str]            # 建议（列表）
    analysis_timestamp: str               # 分析时间戳


# ============================================================
# 3. RAG 知识库工具函数
# ============================================================

# 内置的医学放射学知识片段（作为演示用途）
# 实际项目中可替换为从 PubMed 下载的真实文献摘要
# 获取方式：pip install biopython，然后用 Entrez API 下载
MEDICAL_KNOWLEDGE_BASE = [
    "Chest X-ray is the most common imaging study used in emergency medicine. Normal chest X-ray shows clear lung fields, normal cardiac silhouette (less than 50% of thoracic diameter), and sharp costophrenic angles.",
    "Pneumonia on chest X-ray appears as consolidation or infiltrates in the lung parenchyma. Common findings include air space opacity, air bronchograms, and lobar or segmental involvement.",
    "Pleural effusion on chest X-ray presents as blunting of the costophrenic angle, homogeneous opacity in the lower lung fields, and mediastinal shift away from the effusion in large cases.",
    "Pneumothorax is identified on chest X-ray by absence of lung markings peripheral to the visceral pleural line. Tension pneumothorax shows mediastinal shift toward the opposite side.",
    "CT scan of the head is the first-line imaging for suspected intracranial hemorrhage. Acute blood appears hyperdense on non-contrast CT, typically measuring 50-80 Hounsfield units.",
    "MRI is superior to CT for evaluation of soft tissue, posterior fossa structures, spinal cord, and subacute or chronic hemorrhage. T1-weighted images show fat as bright and fluid as dark.",
    "Bone fractures on X-ray appear as lucent lines through cortical bone. Stress fractures may initially be invisible on plain films and require MRI or bone scan for detection.",
    "Pulmonary edema on chest X-ray shows bilateral perihilar infiltrates in a 'bat wing' pattern, Kerley B lines, and upper lobe vascular redistribution.",
    "Abdominal X-ray can identify bowel obstruction (dilated loops, air-fluid levels, absence of gas in colon), free air under the diaphragm indicating perforation, and calcifications.",
    "Ultrasound of the abdomen is preferred for evaluation of gallbladder (gallstones appear as echogenic foci with posterior acoustic shadowing), liver, kidneys, and pelvic organs.",
    "DEXA scan measures bone mineral density. T-score of -1.0 or above is normal; -1.0 to -2.5 indicates osteopenia; below -2.5 indicates osteoporosis.",
    "Brain MRI for stroke: DWI (diffusion-weighted imaging) shows acute ischemic changes within minutes to hours as bright areas; FLAIR shows chronic or subacute changes.",
    "Lung nodule evaluation on CT: nodules under 6mm low risk, 6-8mm intermediate risk, over 8mm high risk. Follow-up per Fleischner Society guidelines based on size and patient risk factors.",
    "Cardiac MRI can assess myocardial viability, function, and structure. Late gadolinium enhancement indicates myocardial fibrosis or scar.",
    "PET scan uses radioactive glucose (FDG) to detect metabolically active tissue. Cancer cells show increased FDG uptake. Used for staging and treatment response evaluation.",
]

@st.cache_resource
# @st.cache_resource 的作用：Streamlit 会缓存这个函数的返回值
# 这意味着向量数据库只会构建一次，重复刷新页面不会重复构建，节省时间
def build_rag_knowledge_base():
    """
    构建本地 RAG 向量知识库。

    流程：
      1. 用 sentence-transformers 把文本转换为向量（嵌入）
      2. 存入 ChromaDB 本地向量数据库
      3. 查询时把问题也转为向量，找最相似的文本片段

    嵌入模型：all-MiniLM-L6-v2
      - 完全免费，下载后本地运行
      - 约 80MB，首次运行会自动下载
    """
    if not RAG_AVAILABLE:
        return None

    try:
        # HuggingFaceEmbeddings 会自动从 HuggingFace 下载嵌入模型到本地缓存
        # 之后离线也能用，不需要 API Key
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}   # 用 CPU 运行，无需 GPU
        )

        # 创建 ChromaDB 内存向量库（不持久化，每次启动重建，适合演示）
        # 如果想持久化，改为：Chroma(persist_directory="./medical_kb", ...)
        vectorstore = Chroma.from_texts(
            texts=MEDICAL_KNOWLEDGE_BASE,
            embedding=embeddings,
            collection_name="medical_knowledge"
        )
        return vectorstore
    except Exception as e:
        st.warning(f"RAG 知识库初始化失败（不影响主功能）: {e}")
        return None


def search_medical_knowledge(vectorstore, query: str, k: int = 3) -> list[str]:
    """
    在 RAG 知识库中检索与查询最相关的 k 条知识。

    参数：
      vectorstore: ChromaDB 向量库实例
      query: 检索关键词（通常是 AI 给出的主要诊断）
      k: 返回最相似的前 k 条结果

    返回：
      相关医学知识文本列表
    """
    if vectorstore is None:
        return []
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception:
        return []


# ============================================================
# 4. 核心功能：用 Ollama 本地模型分析医学图像
# ============================================================

def analyze_with_ollama(image_bytes: bytes, model: str = "moondream:latest") -> dict:
    """
    使用本地 Ollama 模型分析医学图像。

    重要：直接用 requests 发 HTTP 请求到 Ollama REST API，
    完全绕过 ollama Python SDK，避免 SDK 版本兼容问题。
    这和在终端执行 curl 命令完全等价，是最稳定的调用方式。

    参数：
      image_bytes: 图像的二进制数据（bytes）
      model: Ollama 模型名称，如 "moondream:latest" 或 "llava:7b"

    返回：
      包含结构化报告字段的字典
    """

    # 图像转 base64 字符串（和 curl 测试用的格式完全一致）
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # 分析提示词：尽量简短，减少小模型输出错误JSON的概率
    # moondream 能力有限，提示词越简单越好
    analysis_prompt = """Analyze this medical image. Reply with ONLY a JSON object, no other text before or after.
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

    # 构造请求体（和 curl -d 里的 JSON 完全一致）
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": analysis_prompt,
            "images": [image_b64]
        }],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024
        }
    }

    def repair_json(text: str) -> dict:
        """
        尝试多种策略解析/修复模型返回的 JSON 字符串。

        小模型（moondream等）经常输出有缺陷的JSON，常见问题：
          - 末尾缺少 }
          - 列表最后一项后面多了逗号
          - 字符串值里含有未转义的换行符
          - 字段值里含有单引号而不是双引号

        策略（按顺序尝试）：
          1. 直接解析
          2. 提取 { } 之间的内容再解析
          3. 用 re 修复常见语法错误后解析
          4. 全部失败则把原文放入 patient_explanation 返回
        """
        import re

        # 策略1：直接解析
        try:
            return json.loads(text)
        except Exception:
            pass

        # 策略2：提取第一个 { 到最后一个 } 之间的内容
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except Exception:
                pass

            # 策略3：修复常见 JSON 语法错误
            # 3a. 去掉列表/对象末尾多余的逗号（如 ["a", "b",] ）
            chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
            # 3b. 把字符串内部的换行符替换为空格
            chunk = re.sub(r'(?<=")(\n)(?=[^"]*")', ' ', chunk)
            # 3c. 补上末尾缺失的 }（统计括号差值）
            open_b = chunk.count("{")
            close_b = chunk.count("}")
            if open_b > close_b:
                chunk += "}" * (open_b - close_b)
            try:
                return json.loads(chunk)
            except Exception:
                pass

        # 策略4：全部失败，把原始文本当作分析结果返回
        return {
            "image_type": "Unknown",
            "anatomical_region": "Unknown",
            "image_quality": "Unable to assess",
            "key_findings": ["Model returned non-JSON response - see explanation"],
            "abnormalities": ["See patient explanation below"],
            "primary_diagnosis": "See full analysis",
            "differential_diagnoses": [],
            "severity": "Unknown",
            "confidence_level": "Low",
            "patient_explanation": text,   # 把模型原始输出展示给用户
            "recommendations": ["Please consult a healthcare professional"]
        }

    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json=payload,
            timeout=300   # CPU推理较慢，给5分钟
        )

        if resp.status_code != 200:
            raise Exception(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

        resp_data = resp.json()
        raw_text = resp_data["message"]["content"]

        # 用多策略 JSON 修复函数解析输出
        result = repair_json(raw_text)
        result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result

    except requests.exceptions.Timeout:
        return {
            "error": "timeout",
            "image_type": "Timeout",
            "anatomical_region": "N/A",
            "image_quality": "Analysis timed out",
            "key_findings": ["Request timed out after 300 seconds"],
            "abnormalities": [],
            "primary_diagnosis": "Analysis timed out",
            "differential_diagnoses": [],
            "severity": "Unknown",
            "confidence_level": "Low",
            "patient_explanation": (
                f"模型 [{model}] 在300秒内未能完成分析。\n\n"
                "这通常是因为：\n"
                "1. 图像文件过大，请压缩后重试\n"
                "2. CPU负载过高，请关闭其他程序后重试\n"
                "3. 模型第一次加载需要更长时间，请再试一次"
            ),
            "recommendations": [
                "关闭其他占用内存的程序（浏览器多余标签页、IDE等）",
                "压缩图像到500KB以下再上传",
                "再点一次分析按钮重试（第二次通常更快）"
            ],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "error": str(e),
            "image_type": "Error",
            "anatomical_region": "Error",
            "image_quality": "Analysis failed",
            "key_findings": [f"Error: {str(e)}"],
            "abnormalities": [],
            "primary_diagnosis": "Analysis failed",
            "differential_diagnoses": [],
            "severity": "Unknown",
            "confidence_level": "Low",
            "patient_explanation": (
                f"Analysis failed using model [{model}].\n\n"
                f"Error details: {str(e)}\n\n"
                f"Please ensure:\n"
                f"1. Ollama service is running (ollama serve)\n"
                f"2. Model is installed: ollama pull {model}\n"
                f"3. Sufficient memory available (moondream needs ~2GB, llava:7b needs ~5GB)"
            ),
            "recommendations": [
                "Check Ollama service status",
                f"Verify model is installed: ollama pull {model}",
                "Try a lighter model like moondream if memory is limited"
            ],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ============================================================
# 5. 核心功能：云端模型分析（通用 OpenAI 兼容格式，支持任意平台）
# ============================================================

def analyze_with_cloud(image_bytes: bytes, api_key: str,
                       model: str = "gemini-2.0-flash",
                       base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai") -> dict:
    """
    使用任意 OpenAI 兼容的云端视觉 API 分析医学图像。
    不写死任何平台地址，base_url 由调用方（侧边栏用户输入）传入。

    常用平台 base_url 参考：
      Gemini (需代理):  https://generativelanguage.googleapis.com/v1beta/openai
      硅基流动:         https://api.siliconflow.cn/v1
      火山方舟(豆包):   https://ark.cn-beijing.volces.com/api/v3
      阿里百炼:         https://dashscope.aliyuncs.com/compatible-mode/v1
      OpenAI 官方:      https://api.openai.com/v1
      任意中转/镜像:    填对应地址即可
    """
    import re
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    analysis_prompt = """你是一位资深放射科医生，请分析这张医学影像图像。
请严格按照以下JSON格式回复，不要包含任何markdown标记、代码块或其他文字，只输出纯JSON。
所有字段的值必须使用中文填写（severity和confidence_level除外，保持英文）。

{
  "image_type": "影像类型（如X光/MRI/CT/超声）",
  "anatomical_region": "检查部位和体位描述",
  "image_quality": "图像质量评估描述",
  "key_findings": ["主要发现1", "主要发现2", "主要发现3"],
  "abnormalities": ["异常发现描述，如无异常填写：未见明显异常"],
  "primary_diagnosis": "主要诊断结论",
  "differential_diagnoses": ["鉴别诊断1", "鉴别诊断2"],
  "severity": "Normal or Mild or Moderate or Severe",
  "confidence_level": "Low or Medium or High",
  "patient_explanation": "用患者能理解的通俗中文解释检查结果，说明是否正常及需要注意的事项",
  "recommendations": ["临床建议1", "临床建议2"]
}"""

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": analysis_prompt}
            ]
        }],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    def repair_json(text: str) -> dict:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except Exception:
                pass
            chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
            open_b = chunk.count("{")
            close_b = chunk.count("}")
            if open_b > close_b:
                chunk += "}" * (open_b - close_b)
            try:
                return json.loads(chunk)
            except Exception:
                pass
        return {
            "image_type": "Unknown", "anatomical_region": "Unknown",
            "image_quality": "Unable to parse response",
            "key_findings": ["See full analysis in explanation below"],
            "abnormalities": [], "primary_diagnosis": "See explanation",
            "differential_diagnoses": [], "severity": "Unknown",
            "confidence_level": "Low", "patient_explanation": text,
            "recommendations": ["Please consult a healthcare professional"]
        }

    endpoint = base_url.rstrip("/") + "/chat/completions"

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
            raise Exception(f"API 返回错误 HTTP {resp.status_code}: {resp.text[:300]}")

        raw_text = resp.json()["choices"][0]["message"]["content"]
        result = repair_json(raw_text)
        result["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["model_used"] = f"Cloud API: {model}"
        return result

    except requests.exceptions.ConnectionError:
        raise Exception(f"无法连接到 {endpoint}，请检查 Base URL 或网络/代理设置")
    except requests.exceptions.Timeout:
        raise Exception("请求超时（60秒），请检查网络")
    except Exception as e:
        return {
            "error": str(e),
            "image_type": "Error", "anatomical_region": "Error",
            "image_quality": "Analysis failed",
            "key_findings": [f"Cloud API error: {str(e)}"],
            "abnormalities": [], "primary_diagnosis": "Analysis failed",
            "differential_diagnoses": [], "severity": "Unknown",
            "confidence_level": "Low",
            "patient_explanation": f"云端分析失败：{str(e)}",
            "recommendations": ["检查 API Key 和 Base URL", "确认账户有余额"],
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ============================================================
# 6. PDF 报告生成
# ============================================================

def generate_pdf_report(report: dict, rag_context: list[str]) -> bytes:
    """
    将分析结果生成 PDF 格式的诊断报告。

    使用 reportlab 库构建 PDF，包含：
      - 报告头部（标题、时间戳）
      - 影像信息（类型、区域、质量）
      - 主要发现
      - 诊断评估
      - RAG 参考知识
      - 免责声明

    返回：
      PDF 文件的二进制字节数据（用于 Streamlit 下载按钮）
    """
    if not PDF_AVAILABLE:
        return None

    # 使用内存缓冲区，不写磁盘
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=20*mm, bottomMargin=20*mm,
                             leftMargin=20*mm, rightMargin=20*mm)

    styles = getSampleStyleSheet()
    story = []  # story 是 reportlab 的内容列表，按顺序渲染

    # 自定义样式
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=18, textColor=colors.HexColor('#1a5276'),
                                  spaceAfter=6)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontSize=12, textColor=colors.HexColor('#2980b9'),
                                    spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, spaceAfter=3)
    warning_style = ParagraphStyle('Warning', parent=styles['Normal'],
                                    fontSize=9, textColor=colors.HexColor('#7f8c8d'),
                                    backColor=colors.HexColor('#f8f9fa'))

    # 标题
    story.append(Paragraph("🏥 Medical Imaging Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {report.get('analysis_timestamp', 'N/A')} | Model: {report.get('model_used', 'LLaVA 7B (Local)')}", body_style))
    story.append(Spacer(1, 5*mm))

    # 影像信息表格
    story.append(Paragraph("Image Information", heading_style))
    img_data = [
        ["Field", "Value"],
        ["Image Type", report.get("image_type", "N/A")],
        ["Anatomical Region", report.get("anatomical_region", "N/A")],
        ["Image Quality", report.get("image_quality", "N/A")],
        ["Severity", report.get("severity", "N/A")],
        ["Confidence Level", report.get("confidence_level", "N/A")],
    ]
    table = Table(img_data, colWidths=[60*mm, 110*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(table)
    story.append(Spacer(1, 4*mm))

    # 主要发现
    story.append(Paragraph("Key Findings", heading_style))
    for finding in report.get("key_findings", []):
        story.append(Paragraph(f"• {finding}", body_style))

    # 异常发现
    if report.get("abnormalities"):
        story.append(Paragraph("Abnormalities", heading_style))
        for abnorm in report.get("abnormalities", []):
            story.append(Paragraph(f"⚠ {abnorm}", body_style))

    # 诊断评估
    story.append(Paragraph("Diagnostic Assessment", heading_style))
    story.append(Paragraph(f"<b>Primary Diagnosis:</b> {report.get('primary_diagnosis', 'N/A')}", body_style))
    diffs = report.get("differential_diagnoses", [])
    if diffs:
        story.append(Paragraph("<b>Differential Diagnoses:</b>", body_style))
        for i, d in enumerate(diffs, 1):
            story.append(Paragraph(f"  {i}. {d}", body_style))

    # 患者解释
    story.append(Paragraph("Patient-Friendly Explanation", heading_style))
    story.append(Paragraph(report.get("patient_explanation", "N/A"), body_style))

    # 建议
    story.append(Paragraph("Recommendations", heading_style))
    for rec in report.get("recommendations", []):
        story.append(Paragraph(f"→ {rec}", body_style))

    # RAG 参考知识
    if rag_context:
        story.append(Paragraph("Reference Knowledge (RAG Retrieved)", heading_style))
        for i, ctx in enumerate(rag_context, 1):
            story.append(Paragraph(f"[{i}] {ctx[:300]}...",
                                    ParagraphStyle('Small', parent=styles['Normal'],
                                                   fontSize=8, textColor=colors.HexColor('#555'))))
            story.append(Spacer(1, 2*mm))

    # 免责声明
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This AI analysis is for educational and informational purposes only. "
        "All findings must be reviewed by qualified healthcare professionals. "
        "Do not make clinical decisions based solely on this report.",
        warning_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 7. Streamlit UI 主界面
# ============================================================

def render_report_ui(report: dict, rag_context: list[str]):
    """
    渲染结构化诊断报告界面。
    - 全面中文化
    - 模型输出内容（英文）保持原样，仅 UI 标签翻译为中文
    - 增加分割线、图标、配色，让界面更像专业医疗报告而非大模型对话框
    """
    if "error" in report and report.get("image_type") == "Error":
        st.error(f"❌ 分析失败：{report['error']}")
        st.info(report.get("patient_explanation", ""))
        return

    # ── 严重程度配色和中文映射 ──
    severity = report.get("severity", "Unknown")
    severity_map = {
        "Normal":   ("🟢", "正常",   "normal"),
        "Mild":     ("🟡", "轻度异常", "off"),
        "Moderate": ("🟠", "中度异常", "off"),
        "Severe":   ("🔴", "重度异常", "inverse"),
        "Unknown":  ("⚪", "未知",   "off"),
    }
    sev_icon, sev_cn, sev_delta = severity_map.get(severity, ("⚪", severity, "off"))

    confidence_map = {"Low": "低", "Medium": "中", "High": "高"}
    conf_cn = confidence_map.get(report.get("confidence_level", ""), report.get("confidence_level", "N/A"))

    # ── 顶部四项核心指标 ──
    st.markdown("### 📊 影像概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("影像类型", report.get("image_type", "N/A"))
    c2.metric("检查部位", report.get("anatomical_region", "N/A"))
    c3.metric("严重程度", f"{sev_icon} {sev_cn}")
    c4.metric("诊断可信度", conf_cn)

    st.divider()

    # ── 主体两栏：左侧发现 / 右侧诊断 ──
    left, right = st.columns([3, 2])

    with left:
        # 主要发现
        st.markdown("#### 🔬 主要影像发现")
        findings = report.get("key_findings", [])
        if findings:
            for f in findings:
                st.markdown(f"&nbsp;&nbsp;• {f}")
        else:
            st.caption("暂无记录")

        st.markdown("")  # 间距

        # 图像质量评估
        st.markdown("#### 📐 图像质量评估")
        quality = report.get("image_quality", "N/A")
        st.info(f"📋 {quality}")

        # 异常发现（仅有异常时显示）
        abnorms = [a for a in report.get("abnormalities", [])
                   if a.lower() not in ("none detected", "none", "")]
        if abnorms:
            st.markdown("#### ⚠️ 异常发现")
            for a in abnorms:
                st.warning(f"⚠️ {a}")

    with right:
        # 主要诊断（高亮绿框）
        st.markdown("#### 🩺 主要诊断")
        primary = report.get("primary_diagnosis", "N/A")
        st.success(f"**{primary}**")

        # 鉴别诊断
        diffs = [d for d in report.get("differential_diagnoses", [])
                 if d.lower() not in ("none applicable", "none", "n/a", "")]
        if diffs:
            st.markdown("#### 🗂️ 鉴别诊断")
            for i, d in enumerate(diffs, 1):
                st.markdown(f"&nbsp;&nbsp;**{i}.** {d}")
        else:
            st.markdown("#### 🗂️ 鉴别诊断")
            st.caption("无其他需鉴别的诊断")

        st.markdown("")

        # 临床建议
        recs = report.get("recommendations", [])
        if recs:
            st.markdown("#### 💡 临床建议")
            for r in recs:
                st.markdown(f"&nbsp;&nbsp;→ {r}")

    st.divider()

    # ── 患者说明（白话解释，默认展开）──
    st.markdown("#### 👤 患者友好说明")
    with st.expander("展开查看通俗解读", expanded=True):
        explanation = report.get("patient_explanation", "N/A")
        # 给解释加一个浅色引用框，视觉上区别于代码块
        st.markdown(
            f"<div style='background:#1e3a5f;border-left:4px solid #4a9eff;"
            f"padding:12px 16px;border-radius:4px;line-height:1.8;'>{explanation}</div>",
            unsafe_allow_html=True
        )

    # ── 文献检索（Agno Multi-Agent 模式才有）──
    if "literature_context" in report:
        st.markdown("#### 📚 相关文献检索结果")
        with st.expander("展开查看 AI 检索到的医学文献"):
            st.markdown(report["literature_context"])

    # ── RAG 知识库参考 ──
    if rag_context:
        st.markdown("#### 🗄️ RAG 知识库参考")
        with st.expander(f"展开查看 {len(rag_context)} 条匹配医学知识"):
            for i, ctx in enumerate(rag_context, 1):
                st.markdown(
                    f"<div style='background:#1a2f1a;border-left:3px solid #4caf50;"
                    f"padding:10px 14px;margin-bottom:8px;border-radius:4px;"
                    f"font-size:0.88em;line-height:1.7;'>"
                    f"<b>参考 [{i}]</b>&nbsp;&nbsp;{ctx}</div>",
                    unsafe_allow_html=True
                )

    # ── 底部模型信息和时间戳 ──
    st.divider()
    col_model, col_time = st.columns(2)
    col_model.caption(f"🤖 分析模型：{report.get('model_used', 'N/A')}")
    col_time.caption(f"⏱ 分析完成时间：{report.get('analysis_timestamp', 'N/A')}")



# ============================================================
# 8. 主程序入口
# ============================================================

def main():
    """
    Streamlit 应用主函数。
    所有 UI 组件在这里组装。
    """

    # --- 页面配置（必须是第一个 Streamlit 调用） ---
    st.set_page_config(
        page_title="Medical Imaging Agent",
        page_icon="🏥",
        layout="wide",         # 使用宽布局
        initial_sidebar_state="expanded"
    )

    # ---- 初始化 Session State ----
    # 优先级：.env 文件 > session_state 手动输入
    # os.getenv() 读取的是 load_dotenv() 注入的环境变量
    if "google_api_key" not in st.session_state:
        # 依次尝试读取常见平台的环境变量名
        env_key = (
            os.getenv("DASHSCOPE_API_KEY") or   # 阿里百炼
            os.getenv("SILICONFLOW_API_KEY") or  # 硅基流动
            os.getenv("GEMINI_API_KEY") or        # Google Gemini
            os.getenv("CLOUD_API_KEY")            # 通用备用名
        )
        st.session_state.google_api_key = env_key  # None 或从 .env 读到的值
    if "cloud_model" not in st.session_state:
        st.session_state.cloud_model = os.getenv("CLOUD_MODEL", "qwen-vl-max")
    if "cloud_base_url" not in st.session_state:
        st.session_state.cloud_base_url = os.getenv(
            "CLOUD_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "current_report" not in st.session_state:
        st.session_state.current_report = None
    if "current_rag_context" not in st.session_state:
        st.session_state.current_rag_context = []

    # ---- 初始化 RAG 知识库（首次加载，之后缓存） ----
    vectorstore = None
    if RAG_AVAILABLE:
        with st.spinner("🗄️ 初始化 RAG 知识库（首次启动需下载嵌入模型，约 80MB）..."):
            vectorstore = build_rag_knowledge_base()

    # ================================================================
    # 侧边栏
    # ================================================================
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.divider()

        # ---- 模型选择 ----
        st.markdown("### 🤖 Model Selection")
        model_choice = st.radio(
            "选择分析模型",
            options=["🖥️ 本地模型 (Ollama)", "☁️ 云端模型 (API)"],
            index=0,
            help="本地模型无需 API Key，完全离线；云端模型填入任意平台 API Key 即可使用"
        )

        use_ollama = "本地" in model_choice

        if use_ollama:
            # 本地 Ollama 配置
            st.success("✅ 使用本地 Ollama 模型，无需 API Key")

            # ----------------------------------------------------------------
            # 动态获取本地已安装的 Ollama 模型列表
            # 用 requests 直接请求 Ollama 的 HTTP API，比 SDK 更稳定
            # Ollama 默认监听 http://localhost:11434
            # ----------------------------------------------------------------
            def get_local_ollama_models():
                """
                通过 HTTP 请求获取本地 Ollama 已安装模型列表。
                返回：(模型名列表, 错误信息或None)
                """
                try:
                    # GET /api/tags 返回所有已下载的模型
                    resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
                    if resp.status_code == 200:
                        data = resp.json()
                        # 每个模型对象里的 "name" 字段就是模型名，如 "llava:7b"
                        model_names = [m["name"] for m in data.get("models", [])]
                        return model_names, None
                    else:
                        return [], f"HTTP {resp.status_code}"
                except requests.exceptions.ConnectionError:
                    # 连接被拒绝 = Ollama 服务没启动
                    return [], "connection_refused"
                except Exception as e:
                    return [], str(e)

            local_models, ollama_error = get_local_ollama_models()

            if ollama_error == "connection_refused":
                # Ollama 服务未启动
                st.error(
                    "❌ **Ollama 服务未运行**\n\n"
                    "请执行以下任一操作后**刷新页面**：\n"
                    "- 点击系统托盘 Ollama 图标\n"
                    "- 或在终端运行：`ollama serve`"
                )
                ollama_model = None  # 服务未运行，不能选模型

            elif ollama_error:
                st.warning(f"⚠️ 获取模型列表出错：{ollama_error}")
                ollama_model = None

            elif not local_models:
                st.warning(
                    "⚠️ **未找到已安装的模型**\n\n"
                    "请在终端运行：`ollama pull moondream`（轻量）或 `ollama pull llava:7b`"
                )
                ollama_model = None

            else:
                # ✅ 成功获取到本地模型列表
                # 优先推荐视觉模型（名字含 llava / moondream / vision 的）
                vision_models = [m for m in local_models
                                  if any(kw in m.lower() for kw in ["llava", "moondream", "vision", "phi3"])]
                other_models = [m for m in local_models if m not in vision_models]

                # 视觉模型排前面，其他模型排后面
                sorted_models = vision_models + other_models

                # 显示模型数量提示
                if vision_models:
                    st.success(f"✅ Ollama 服务运行中，找到 {len(local_models)} 个模型（{len(vision_models)} 个视觉模型）")
                else:
                    st.warning(
                        f"⚠️ 找到 {len(local_models)} 个模型，但没有视觉模型\n"
                        "建议安装：`ollama pull llava:7b`"
                    )

                # 下拉菜单：只显示你本地实际安装的模型
                ollama_model = st.selectbox(
                    "选择本地视觉模型",
                    options=sorted_models,
                    index=0,
                    help=(
                        "列表来自你本地已安装的 Ollama 模型。\n"
                        "如需安装更多：ollama pull llava:13b / moondream 等"
                    )
                )

                # 如果选了非视觉模型，给出警告
                if ollama_model not in vision_models:
                    st.warning(f"⚠️ {ollama_model} 可能不支持图像输入，建议选择含 'llava' 的视觉模型")
        else:
            # 云端模式：通用 OpenAI 兼容格式，支持任意平台
            ollama_model = None

            # ---- 平台预设（快速填入）----
            st.markdown("**☁️ 云端 API 配置**")
            preset = st.selectbox(
                "平台预设（可自定义）",
                options=[
                    "阿里百炼 DashScope",
                    "硅基流动 SiliconFlow",
                    "火山方舟 豆包",
                    "Gemini (需代理/VPN)",
                    "自定义",
                ],
                index=0,   # 默认选中阿里百炼
                help="选择预设后自动填入 Base URL，再输入对应平台的 API Key 即可"
            )

            # 预设 base_url 和模型名映射
            PRESETS = {
                "Gemini (需代理/VPN)": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "model": "gemini-2.0-flash",
                    "key_hint": "AIzaSy...",
                    "note": "需要 VPN 访问。key 从 aistudio.google.com 获取，每天免费 1500 次"
                },
                "硅基流动 SiliconFlow": {
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
                    "key_hint": "sk-...",
                    "note": "国内直连。注册：cloud.siliconflow.cn，视觉模型需付费额度"
                },
                "火山方舟 豆包": {
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "model": "doubao-vision-pro-32k",
                    "key_hint": "填入火山方舟 API Key",
                    "note": "国内直连。注册：console.volcengine.com，新用户有免费额度"
                },
                "阿里百炼 DashScope": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-vl-max",
                    "key_hint": "sk-...",
                    "note": "国内直连。注册：bailian.console.aliyun.com，新用户有免费额度"
                },
                "自定义": {
                    "base_url": "",
                    "model": "",
                    "key_hint": "填入 API Key",
                    "note": "填入任意 OpenAI 兼容平台的 Base URL 和模型名"
                }
            }

            p = PRESETS.get(preset, PRESETS["自定义"])
            if p["note"]:
                st.caption(p["note"])

            # Base URL 输入（预设自动填入，可手动修改）
            cloud_base_url = st.text_input(
                "Base URL",
                value=st.session_state.get("cloud_base_url", p["base_url"]),
                placeholder="https://api.example.com/v1",
                help="API 基础地址，不含 /chat/completions。切换预设后自动填入"
            )
            st.session_state.cloud_base_url = cloud_base_url

            # 模型名输入
            cloud_model_input = st.text_input(
                "模型名称",
                value=st.session_state.get("cloud_model", p["model"]),
                placeholder="模型ID，如 gemini-2.0-flash",
                help="填入对应平台的模型 ID"
            )
            st.session_state.cloud_model = cloud_model_input

            # API Key 输入
            st.markdown("**API Key**")
            if not st.session_state.google_api_key:
                api_key_input = st.text_input(
                    "输入 API Key",
                    type="password",
                    placeholder=p["key_hint"]
                )
                st.caption("💡 也可在 .env 文件中写入 DASHSCOPE_API_KEY=sk-xxx，下次启动自动读取")
                if api_key_input:
                    st.session_state.google_api_key = api_key_input
                    st.success("✅ API Key 已保存")
                    st.rerun()
            else:
                # 判断 key 是来自 .env 还是手动输入，给出不同提示
                from_env = bool(
                    os.getenv("DASHSCOPE_API_KEY") or
                    os.getenv("SILICONFLOW_API_KEY") or
                    os.getenv("GEMINI_API_KEY") or
                    os.getenv("CLOUD_API_KEY")
                )
                if from_env:
                    st.success("✅ API Key 已从 .env 自动读取")
                else:
                    st.success("✅ API Key 已配置")
                if st.button("🔄 重置 API Key"):
                    st.session_state.google_api_key = None
                    st.rerun()

            # 显示当前配置摘要
            if st.session_state.google_api_key and cloud_base_url and cloud_model_input:
                st.caption(f"📡 {cloud_base_url.split('/')[2]} | {cloud_model_input}")

        st.divider()

        # ---- RAG 状态 ----
        st.markdown("### 🗄️ RAG Knowledge Base")
        if RAG_AVAILABLE and vectorstore is not None:
            st.success(f"✅ 知识库就绪\n{len(MEDICAL_KNOWLEDGE_BASE)} 条医学知识")
        elif not RAG_AVAILABLE:
            st.warning("⚠️ RAG 未安装\n运行: pip install langchain langchain-community chromadb sentence-transformers")
        else:
            st.error("❌ 知识库初始化失败")

        st.divider()

        # ---- 历史记录 ----
        if st.session_state.analysis_history:
            st.markdown("### 📂 Analysis History")
            st.caption(f"共 {len(st.session_state.analysis_history)} 条历史记录")

            for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
                # 显示最近 5 条
                with st.expander(f"#{len(st.session_state.analysis_history)-i} - {hist['timestamp']}"):
                    st.write(f"**诊断**: {hist['diagnosis']}")
                    st.write(f"**严重度**: {hist['severity']}")

            if st.button("🗑️ 清除历史"):
                st.session_state.analysis_history = []
                st.rerun()

        st.divider()

        # ---- 免责声明 ----
        st.warning(
            "⚠️ **免责声明**\n\n"
            "本工具仅供教育和研究目的。所有分析结果必须由具备资质的医疗专业人员审核。"
            "请勿仅凭本工具的分析做出临床决策。"
        )

    # ================================================================
    # 主内容区
    # ================================================================
    st.title("🏥 AI 医学影像智能诊断系统")
    st.markdown("*本地模型 / 云端大模型 · RAG医学知识库 · 结构化报告导出*")
    st.divider()

    # ---- 上传区 ----
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📁 上传医学影像图像",
            type=["jpg", "jpeg", "png"],
            help="支持格式：JPG, JPEG, PNG | DICOM 需先转换为 PNG"
        )

    with col_info:
        st.markdown("**支持的图像类型**")
        st.markdown("• X-Ray（胸片、骨骼）\n• MRI\n• CT Scan\n• Ultrasound\n• 其他放射影像")

    # ---- 如果上传了文件 ----
    if uploaded_file is not None:

        # 读取图像
        image = PILImage.open(uploaded_file)

        # 显示图像（居中，限制宽度）
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

            analyze_btn = st.button(
                "🔍 开始分析",
                type="primary",
                use_container_width=True,
                disabled=(
                    (use_ollama and not ollama_model) or          # 本地模式但模型未就绪
                    (not use_ollama and not st.session_state.google_api_key)  # 云端模式但无Key
                )
            )

        if use_ollama and not ollama_model:
            st.warning("请先确保 Ollama 服务运行并安装视觉模型")
        elif not use_ollama and not st.session_state.google_api_key:
            st.warning("请先在侧边栏填入 API Key")

        # ---- 点击分析按钮 ----
        if analyze_btn:
            spinner_msg = (
                "🔄 AI 正在分析影像... CPU推理需要 1-3 分钟，请耐心等待"
                if use_ollama else
                "🔄 云端 AI 正在分析影像... 通常 5-15 秒内完成"
            )
            with st.spinner(spinner_msg):

                img_bytes_io = io.BytesIO()
                image_to_save = image.convert("RGB")
                image_to_save.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                if use_ollama:
                    report = analyze_with_ollama(img_bytes, model=ollama_model)
                    report["model_used"] = f"Ollama {ollama_model} (Local)"
                else:
                    # 从 session_state 读取用户在侧边栏填入的配置
                    cloud_model = st.session_state.get("cloud_model", "gemini-2.0-flash")
                    cloud_base_url = st.session_state.get(
                        "cloud_base_url",
                        "https://generativelanguage.googleapis.com/v1beta/openai"
                    )
                    report = analyze_with_cloud(
                        img_bytes,
                        api_key=st.session_state.google_api_key,
                        model=cloud_model,
                        base_url=cloud_base_url
                    )

                # RAG 检索：用主要诊断作为检索词
                rag_context = []
                if vectorstore and RAG_AVAILABLE:
                    primary_dx = report.get("primary_diagnosis", "")
                    image_type = report.get("image_type", "")
                    search_query = f"{primary_dx} {image_type}".strip()
                    if search_query and search_query != "Error":
                        rag_context = search_medical_knowledge(vectorstore, search_query, k=3)

                # 保存到 Session State（用于下载按钮）
                st.session_state.current_report = report
                st.session_state.current_rag_context = rag_context

                # 添加到历史记录
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "filename": uploaded_file.name,
                    "diagnosis": report.get("primary_diagnosis", "N/A"),
                    "severity": report.get("severity", "N/A"),
                    "model": report.get("model_used", "Unknown")
                })

        # ---- 显示分析结果 ----
        if st.session_state.current_report:
            st.divider()
            st.markdown("## 📋 影像分析报告")

            render_report_ui(
                st.session_state.current_report,
                st.session_state.current_rag_context
            )

            # PDF 下载按钮
            st.divider()
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])

            with col_dl1:
                # JSON 下载（原始数据，方便开发者查看）
                json_str = json.dumps(st.session_state.current_report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 下载 JSON 报告",
                    data=json_str,
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with col_dl2:
                # PDF 下载
                if PDF_AVAILABLE:
                    pdf_bytes = generate_pdf_report(
                        st.session_state.current_report,
                        st.session_state.current_rag_context
                    )
                    if pdf_bytes:
                        st.download_button(
                            label="📄 下载 PDF 报告",
                            data=pdf_bytes,
                            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.info("安装 reportlab 以启用 PDF 导出:\npip install reportlab")

    else:
        # 未上传文件时显示引导信息
        st.info("👆 请上传医学影像图像开始分析")

        # 展示技术栈说明（美化项目说明）
        st.markdown("---")
        st.markdown("### 🛠️ 技术架构")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**🖥️ 本地模型**\nOllama + LLaVA 7B\n无需 API Key")
        with col2:
            st.markdown("**🗄️ RAG 知识库**\nLangChain + ChromaDB\n医学文献检索")
        with col3:
            st.markdown("**🤖 Multi-Agent**\nAgno Framework\n（Gemini 模式）")
        with col4:
            st.markdown("**📄 结构化输出**\nPydantic + PDF\n标准报告导出")


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    main()