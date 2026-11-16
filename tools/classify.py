"""工具：板块分类（规则 + 可选 LLM 复核）。包装 classifier 模块。"""
from __future__ import annotations

from typing import Any

import classifier
from sectors import SECTORS

from .schemas import ClassifyNewsInput


def classify_news(input: ClassifyNewsInput) -> dict[str, Any]:
    """判断一条新闻是否属于目标板块。

    返回 {"matched", "confidence", "score", "matched_keywords", "reason",
    "llm_checked", "llm_kept"}。use_llm=True 且命中低置信度/高风险条件时，
    会用传入的 LLM verifier 复核；复核结果不保留则 matched=False。
    """
    rules = SECTORS.get(input.sector)
    if rules is None:
        return {
            "matched": False,
            "confidence": 0.0,
            "score": 0.0,
            "matched_keywords": [],
            "reason": f"未知板块「{input.sector}」",
            "llm_checked": False,
            "llm_kept": None,
        }

    news = {"标题": input.title, "新闻内容": input.content}
    result = classifier.classify_news_item(news, input.sector, list(rules))

    matched = result.matched
    llm_checked = False
    llm_kept: bool | None = None
    reason = result.reason

    if input.use_llm and result.matched and input_llm_verifier is not None:
        if classifier.should_validate_with_llm(input.sector, result):
            validation = classifier.validate_classification_with_llm(
                news, input.sector, result, verifier=input_llm_verifier
            )
            llm_checked = True
            llm_kept = validation.should_keep
            matched = validation.should_keep
            reason = validation.reason

    return {
        "matched": matched,
        "confidence": result.confidence,
        "score": result.score,
        "matched_keywords": list(result.matched_keywords),
        "reason": reason,
        "llm_checked": llm_checked,
        "llm_kept": llm_kept,
    }


# Agent 运行时装配点：graph 层在调用前注入 verifier，工具本身不感知供应商。
input_llm_verifier: Any = None


def set_llm_verifier(verifier: Any) -> None:
    """注入 LLM 复核器（如 ChatCompletionsLLMVerifier 实例）。"""
    global input_llm_verifier
    input_llm_verifier = verifier
