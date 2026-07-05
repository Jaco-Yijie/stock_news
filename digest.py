"""每日板块日报：用 LLM 把板块当天的新闻归纳成 3-5 句中文摘要。"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd


MAX_DIGEST_ITEMS = 15
MAX_CONTENT_CHARS = 120

DIGEST_SYSTEM_PROMPT = (
    "你是A股行业新闻编辑。只根据给定的新闻归纳，"
    "不得编造事实、数字或引入给定新闻之外的信息。"
)


def build_digest_records(news_df: pd.DataFrame) -> list[dict[str, str]]:
    if news_df is None or news_df.empty:
        return []
    records: list[dict[str, str]] = []
    for row in news_df.head(MAX_DIGEST_ITEMS).to_dict("records"):
        records.append(
            {
                "title": str(row.get("标题", "")),
                "time": str(row.get("发布时间", "")),
                "source": str(row.get("来源媒体", "")),
                "summary": str(row.get("新闻内容", ""))[:MAX_CONTENT_CHARS],
            }
        )
    return records


def build_digest_prompt(sector: str, records: list[dict[str, str]]) -> str:
    payload = {
        "task": (
            f"用 3 到 5 句中文归纳「{sector}」板块今天的关键动态，"
            "指出整体偏利好、偏利空还是中性，并说明主要原因。"
            "直接输出连贯段落，不要列表符号，不要输出任务说明。"
        ),
        "news": records,
    }
    return json.dumps(payload, ensure_ascii=False)


def generate_sector_digest(
    verifier: Any,
    sector: str,
    records: list[dict[str, str]],
) -> str:
    if not records:
        return ""
    text = verifier.complete(DIGEST_SYSTEM_PROMPT, build_digest_prompt(sector, records))
    return " ".join(str(text or "").split())
