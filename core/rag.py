"""
core/rag.py
===========
RAG（检索增强生成）知识库模块。

职责：
  1. 加载知识文本（PubMed JSON 优先，内置知识兜底）
  2. 用 sentence-transformers 嵌入，构建 ChromaDB 向量库
  3. 提供相似度检索接口

RAG 工作流：
  离线构建：文本 → sentence-transformers → 384维向量 → ChromaDB 持久化
  在线检索：诊断词 → 向量化 → 余弦相似度搜索 → top-k 结果

注意：
  build_rag_knowledge_base 带有 @st.cache_resource 装饰器，
  由 app.py 调用，保证整个 Streamlit 生命周期内只初始化一次。
  core 模块本身不导入 streamlit，保持 UI 无关性。
"""

import os
import json
from pathlib import Path

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ── 内置知识（保底 fallback）────────────────────────────────
_BUILTIN_KNOWLEDGE = [
    "Chest X-ray is the most common imaging study used in emergency medicine. Normal chest X-ray shows clear lung fields, normal cardiac silhouette (less than 50% of thoracic diameter), and sharp costophrenic angles.",
    "Pneumonia on chest X-ray appears as consolidation or infiltrates in the lung parenchyma. Common findings include air space opacity, air bronchograms, and lobar or segmental involvement.",
    "Pleural effusion on chest X-ray presents as blunting of the costophrenic angle, homogeneous opacity in the lower lung fields, and mediastinal shift away from the effusion in large cases.",
    "Pneumothorax is identified on chest X-ray by absence of lung markings peripheral to the visceral pleural line. Tension pneumothorax shows mediastinal shift toward the opposite side.",
    "CT scan of the head is the first-line imaging for suspected intracranial hemorrhage. Acute blood appears hyperdense on non-contrast CT.",
    "MRI is superior to CT for evaluation of soft tissue, posterior fossa structures, and spinal cord. T1-weighted images show fat as bright and fluid as dark.",
    "Bone fractures on X-ray appear as lucent lines through cortical bone. Stress fractures may require MRI or bone scan for detection.",
    "Pulmonary edema on chest X-ray shows bilateral perihilar infiltrates in a 'bat wing' pattern, Kerley B lines.",
    "Abdominal X-ray can identify bowel obstruction, free air under the diaphragm indicating perforation.",
    "Ultrasound is preferred for gallbladder evaluation. Gallstones appear as echogenic foci with posterior acoustic shadowing.",
    "DEXA scan: T-score above -1.0 is normal; -1.0 to -2.5 is osteopenia; below -2.5 is osteoporosis.",
    "Brain MRI DWI shows acute ischemic changes within minutes to hours as bright areas.",
    "Lung nodule on CT: under 6mm low risk, 6-8mm intermediate risk, over 8mm high risk per Fleischner guidelines.",
    "Cardiac MRI late gadolinium enhancement indicates myocardial fibrosis or scar.",
    "PET scan uses FDG to detect metabolically active tissue. Cancer cells show increased FDG uptake.",
]


def load_knowledge_base() -> tuple[list[str], str]:
    """
    加载医学知识文本。

    查找路径：项目根目录下的 medical_knowledge_base.json
    优先级：PubMed JSON > 内置知识

    返回：
      (texts, source_description)
      texts              : 知识文本列表
      source_description : 来源描述，用于侧边栏展示
    """
    # __file__ 是 core/rag.py，父目录是 core/，再父目录是项目根
    root = Path(__file__).parent.parent
    json_path = root / "medical_knowledge_base.json"

    if json_path.exists():
        try:
            texts = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(texts, list) and len(texts) > 0:
                return texts, f"PubMed文献库（{len(texts)}条）"
        except Exception:
            pass

    return _BUILTIN_KNOWLEDGE, f"内置知识库（{len(_BUILTIN_KNOWLEDGE)}条）"


# 模块级加载，只执行一次
MEDICAL_KNOWLEDGE_BASE, KB_SOURCE = load_knowledge_base()


def build_rag_knowledge_base() -> tuple:
    if not RAG_AVAILABLE:
        return None, "RAG库未安装"
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        persist_dir = str(Path(__file__).parent.parent / "medical_kb_store")

        # 关键改动：目录已存在就直接加载，不重新构建
        if Path(persist_dir).exists():
            vectorstore = Chroma(
                collection_name="medical_knowledge",
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
        else:
            vectorstore = Chroma.from_texts(
                texts=MEDICAL_KNOWLEDGE_BASE,
                embedding=embeddings,
                collection_name="medical_knowledge",
                persist_directory=persist_dir
            )
        return vectorstore, KB_SOURCE
    except Exception as e:
        return None, f"初始化失败：{e}"


def search_medical_knowledge(vectorstore, query: str, k: int = 3) -> list[str]:
    """
    在向量库中检索最相关的 k 条医学知识。

    参数：
      vectorstore : ChromaDB 实例
      query       : 检索词（通常是主要诊断 + 影像类型）
      k           : 返回 top-k 条结果（默认3，可在侧边栏调节）

    返回：
      相关文本列表，空列表表示检索失败或知识库不可用
    """
    if vectorstore is None:
        return []
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception:
        return []