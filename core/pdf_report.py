"""
core/pdf_report.py
==================
PDF 诊断报告生成模块（reportlab）。

设计：在内存缓冲区生成 PDF，不写磁盘，
返回 bytes 供 Streamlit 下载按钮直接使用。
"""

import io

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def generate_pdf_report(report: dict, rag_context: list) -> bytes | None:
    """
    将分析结果生成 PDF 格式的诊断报告。

    参数：
      report      : 影像分析结果字典
      rag_context : RAG 检索到的参考知识列表

    返回：
      PDF 二进制数据（bytes），PDF_AVAILABLE=False 时返回 None
    """
    if not PDF_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=20*mm, rightMargin=20*mm
    )

    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('Title',   parent=styles['Title'],
                                   fontSize=18, textColor=colors.HexColor('#1a5276'), spaceAfter=6)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                   fontSize=12, textColor=colors.HexColor('#2980b9'),
                                   spaceBefore=12, spaceAfter=4)
    body_style    = ParagraphStyle('Body',    parent=styles['Normal'], fontSize=10, spaceAfter=3)
    small_style   = ParagraphStyle('Small',   parent=styles['Normal'],
                                   fontSize=8, textColor=colors.HexColor('#555'))
    warn_style    = ParagraphStyle('Warn',    parent=styles['Normal'],
                                   fontSize=9, textColor=colors.HexColor('#7f8c8d'))

    story = []

    # ── 标题 ──
    story.append(Paragraph("🏥 AI医学影像诊断报告", title_style))
    story.append(Paragraph(
        f"生成时间：{report.get('analysis_timestamp', 'N/A')} | 模型：{report.get('model_used', 'N/A')}",
        body_style
    ))
    story.append(Spacer(1, 5*mm))

    # ── 影像信息表格 ──
    story.append(Paragraph("影像信息", heading_style))
    table = Table([
        ["字段", "内容"],
        ["影像类型",   report.get("image_type", "N/A")],
        ["检查部位",   report.get("anatomical_region", "N/A")],
        ["图像质量",   report.get("image_quality", "N/A")],
        ["严重程度",   report.get("severity", "N/A")],
        ["置信度",     report.get("confidence_level", "N/A")],
    ], colWidths=[60*mm, 110*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    story.append(table)
    story.append(Spacer(1, 4*mm))

    # ── 主要发现 ──
    story.append(Paragraph("主要发现", heading_style))
    for f in report.get("key_findings", []):
        story.append(Paragraph(f"• {f}", body_style))

    # ── 异常发现 ──
    if report.get("abnormalities"):
        story.append(Paragraph("异常发现", heading_style))
        for a in report.get("abnormalities", []):
            story.append(Paragraph(f"⚠ {a}", body_style))

    # ── 诊断评估 ──
    story.append(Paragraph("诊断评估", heading_style))
    story.append(Paragraph(f"<b>主要诊断：</b>{report.get('primary_diagnosis', 'N/A')}", body_style))
    for i, d in enumerate(report.get("differential_diagnoses", []), 1):
        story.append(Paragraph(f"  鉴别{i}：{d}", body_style))

    # ── Tool Use 综合建议 ──
    if report.get("tool_use_summary"):
        story.append(Paragraph("Tool Use Agent 综合建议", heading_style))
        story.append(Paragraph(report["tool_use_summary"], body_style))

    # ── 患者说明 ──
    story.append(Paragraph("患者友好说明", heading_style))
    story.append(Paragraph(report.get("patient_explanation", "N/A"), body_style))

    # ── 建议 ──
    story.append(Paragraph("临床建议", heading_style))
    for r in report.get("recommendations", []):
        story.append(Paragraph(f"→ {r}", body_style))

    # ── RAG 参考知识 ──
    if rag_context:
        story.append(Paragraph("RAG 参考知识", heading_style))
        for i, ctx in enumerate(rag_context, 1):
            story.append(Paragraph(f"[{i}] {ctx[:300]}...", small_style))
            story.append(Spacer(1, 2*mm))

    # ── 免责声明 ──
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "⚠ 免责声明：本报告仅供教育和研究目的，不构成医疗建议。"
        "所有分析结果必须由具备资质的医疗专业人员审核，请勿仅凭本报告做出临床决策。",
        warn_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()