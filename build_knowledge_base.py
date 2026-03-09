"""
从 PubMed 抓取放射学 / 医学影像相关文献摘要，构建本地 RAG 知识库。

运行完成后会生成 medical_knowledge_base.json，主程序会自动加载它。
PubMed 是美国国立医学图书馆的免费文献数据库，无需注册，无需 API Key。
Bio.Entrez 是 Biopython 提供的 PubMed 接口封装库。

工作原理：
  1. 用关键词在 PubMed 搜索文章 ID 列表（esearch）
  2. 批量获取文章详情，提取标题 + 摘要（efetch）
  3. 清洗文本，保存为 JSON 文件
  4. 主程序加载 JSON，构建 ChromaDB 向量库
"""

import json
import time
import os
from pathlib import Path

# biopython 提供 Bio.Entrez 模块，封装了 NCBI/PubMed 的 E-utilities API
try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("❌ 请先安装 biopython：pip install biopython")
    exit(1)

# ── PubMed API 要求填写邮箱（不会发邮件，只用于滥用溯源）──
Entrez.email = "your_email@example.com"

# ── 搜索关键词列表 ──
# 每个关键词会搜索并抓取 MAX_PER_QUERY 篇文章
# 覆盖主要影像学领域，和系统能处理的图像类型对应
SEARCH_QUERIES = [
    "chest X-ray radiology findings diagnosis",
    "chest CT pulmonary findings",
    "brain MRI diagnosis radiology",
    "abdominal CT findings diagnosis",
    "musculoskeletal X-ray fracture diagnosis",
    "pneumonia chest radiograph diagnosis",
    "pleural effusion radiology",
    "pulmonary nodule CT evaluation",
    "stroke brain imaging MRI CT",
    "cardiac imaging echocardiography",
    "ultrasound abdomen liver gallbladder",
    "bone density DEXA osteoporosis",
]

MAX_PER_QUERY = 40        # 每个关键词抓取数量，总计约 480 条
OUTPUT_FILE = "medical_knowledge_base.json"
DELAY_SECONDS = 0.4       # PubMed 限速：每秒最多 3 次请求，0.4秒间隔安全


def fetch_pubmed_articles(query: str, max_results: int) -> list[dict]:
    """
    用一个关键词搜索 PubMed 并返回文章列表。

    两步调用：
      esearch：输入关键词，返回匹配的 PMID 列表（PubMed 文章 ID）
      efetch ：输入 PMID 列表，返回文章详情（标题、摘要等）
    """
    articles = []

    try:
        # ── 第一步：搜索，获取 PMID 列表 ──
        # retmax 控制最多返回多少个 ID
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"   # 按相关性排序，确保抓到高质量文章
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return []

        # ── 第二步：批量获取文章详情 ──
        # rettype="abstract" 只获取摘要，比全文小得多
        # retmode="xml" 返回结构化 XML，方便解析
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),   # 逗号分隔的 ID 字符串
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        # ── 第三步：解析 XML，提取标题和摘要 ──
        for record in records.get("PubmedArticle", []):
            try:
                article = record["MedlineCitation"]["Article"]

                # 标题（ArticleTitle 可能是 StringElement）
                title = str(article.get("ArticleTitle", "")).strip()

                # 摘要（Abstract.AbstractText 可以是列表，每段有不同标签）
                abstract_data = article.get("Abstract", {}).get("AbstractText", [])
                if isinstance(abstract_data, list):
                    # 结构化摘要：Background/Methods/Results/Conclusion 各一段
                    abstract = " ".join(str(s) for s in abstract_data)
                else:
                    abstract = str(abstract_data)
                abstract = abstract.strip()

                # 过滤：标题和摘要都必须有内容，摘要不能太短（低于50字符的没价值）
                if title and abstract and len(abstract) > 50:
                    articles.append({
                        "title": title,
                        "abstract": abstract,
                        "query": query    # 记录来源关键词，方便调试
                    })

            except Exception:
                # 单篇文章解析失败不影响其他文章
                continue

        time.sleep(DELAY_SECONDS)   # 遵守 PubMed 频率限制

    except Exception as e:
        print(f"  ⚠ 关键词 '{query}' 抓取失败: {e}")

    return articles


def build_knowledge_texts(articles: list[dict]) -> list[str]:
    """
    把文章列表转换为 RAG 知识文本片段。

    格式：[标题] 摘要内容
    这种格式让嵌入模型同时捕获标题语义和摘要内容。
    """
    texts = []
    seen = set()   # 去重，避免同一篇文章因为多个关键词被重复添加

    for art in articles:
        title = art["title"]
        abstract = art["abstract"]

        # 用标题做去重键（同一篇文章可能被多个关键词搜到）
        key = title[:80].lower()
        if key in seen:
            continue
        seen.add(key)

        # 拼接为知识片段
        text = f"[{title}] {abstract}"

        # 限制每条文本长度（过长的摘要截断到1000字符）
        # ChromaDB 对单条文本有大小限制，也避免单条占用过多向量维度
        if len(text) > 1000:
            text = text[:997] + "..."

        texts.append(text)

    return texts


def main():
    print("=" * 60)
    print("PubMed 医学文献知识库构建工具")
    print("=" * 60)
    print(f"搜索领域：{len(SEARCH_QUERIES)} 个关键词")
    print(f"每个关键词抓取：{MAX_PER_QUERY} 篇")
    print(f"预计总条数：~{len(SEARCH_QUERIES) * MAX_PER_QUERY} 篇（去重后会少一些）")
    print(f"输出文件：{OUTPUT_FILE}")
    print()

    all_articles = []

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"[{i}/{len(SEARCH_QUERIES)}] 搜索：{query}")
        articles = fetch_pubmed_articles(query, MAX_PER_QUERY)
        all_articles.extend(articles)
        print(f"  ✓ 获取 {len(articles)} 篇，累计 {len(all_articles)} 篇")

    print()
    print(f"抓取完成，共 {len(all_articles)} 篇原始文章")

    # 转换为知识文本片段（去重）
    knowledge_texts = build_knowledge_texts(all_articles)
    print(f"去重后有效知识片段：{len(knowledge_texts)} 条")

    # 保存为 JSON（列表格式，主程序直接 json.load 读取）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_texts, f, ensure_ascii=False, indent=2)

    print()
    print(f"✅ 知识库已保存到 {OUTPUT_FILE}")
    print(f"   文件大小：{Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB")
    print()
    print("下一步：重启 Streamlit，主程序会自动加载这个知识库")
    print("        侧边栏 RAG 状态会从 '15条' 变成实际数量")


if __name__ == "__main__":
    main()