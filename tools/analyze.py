"""工具：利好/利空分析与重要性排序。包装 analysis 模块。"""
from __future__ import annotations

from typing import Any

import pandas as pd

import analysis

from .schemas import AnalyzeSentimentInput, RankNewsInput


def analyze_sentiment(input: AnalyzeSentimentInput) -> dict[str, Any]:
    """按规则评估一条新闻对目标板块的方向（利好/利空/中性）与重要性。

    返回 analysis 字典：含 overall_sentiment、importance、各板块评估等。
    纯规则计算，不调用 LLM，可安全批量使用。
    """
    news = {"标题": input.title, "新闻内容": input.content}
    result = analysis.analyze_news_item(news, list(input.sectors))
    return {"analysis": result, "error": None}


def rank_news(input: RankNewsInput) -> dict[str, Any]:
    """按重要性降序排序新闻（政策>产业事件>一般资讯>个股异动），同分按时间倒序。

    items 中每条需带 "analysis" 键（analyze_sentiment 的输出）。
    返回 {"count", "items"}，保持原字段并保留 analysis。
    """
    if not input.items:
        return {"count": 0, "items": []}

    rows = []
    for item in input.items:
        row = {k: v for k, v in item.items() if k != "analysis"}
        row["analysis"] = item.get("analysis", {})
        rows.append(row)

    df = pd.DataFrame(rows)
    ranked = analysis.sort_news_by_importance(df)
    return {"count": len(ranked), "items": ranked.to_dict(orient="records")}
