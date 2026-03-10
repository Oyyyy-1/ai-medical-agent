"""
Pydantic 数据模型定义。

为什么用 Pydantic：
  大模型输出是字符串，解析成 dict 后字段可能缺失或类型不符。
  Pydantic BaseModel 强制字段类型，让下游（PDF生成、UI渲染）
  拿到的数据结构始终稳定，不会因为某个字段缺失而崩溃。
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class MedicalReport(BaseModel):
    """
    结构化医学诊断报告数据模型。

    字段说明：
      image_type           : 影像类型，如 X-ray / MRI / CT / Ultrasound
      anatomical_region    : 解剖区域，如 胸部正侧位 / 头部横断面
      image_quality        : 图像质量评估（技术层面）
      key_findings         : 主要影像发现列表
      abnormalities        : 异常发现列表（无异常时为 ["未见明显异常"]）
      primary_diagnosis    : 主要诊断结论
      differential_diagnoses: 鉴别诊断列表
      severity             : 严重程度 Normal/Mild/Moderate/Severe
      confidence_level     : AI 置信度 Low/Medium/High
      patient_explanation  : 患者友好的通俗解释
      recommendations      : 临床建议列表
      analysis_timestamp   : 分析完成时间戳
    """
    image_type:             str
    anatomical_region:      str
    image_quality:          str
    key_findings:           List[str]
    abnormalities:          List[str]
    primary_diagnosis:      str
    differential_diagnoses: List[str]
    severity:               str
    confidence_level:       str
    patient_explanation:    str
    recommendations:        List[str]
    analysis_timestamp:     str