"""工具：抓取板块新闻。包装 fetcher.fetch_sector_news，输出 JSON 友好结构。"""
from __future__ import annotations

from typing import Any

import fetcher
from sectors import SECTORS

from .schemas import FetchSectorNewsInput


def fetch_sector_news(input: FetchSectorNewsInput) -> dict[str, Any]:
    """抓取指定板块的最新新闻并去重。

    返回 {"sector", "count", "items", "warnings", "error"}；
    error 非空表示抓取失败，items 可能为空列表。
    """
    if input.sector not in SECTORS:
        return {
            "sector": input.sector,
            "count": 0,
            "items": [],
            "warnings": [],
            "error": f"未知板块「{input.sector}」，可选：{'、'.join(sorted(SECTORS))}",
        }

    result = fetcher.fetch_sector_news(input.sector, timeout=input.timeout)
    df = result.data
    items = df.to_dict(orient="records") if df is not None and not df.empty else []
    return {
        "sector": input.sector,
        "count": len(items),
        "items": items,
        "warnings": list(result.warnings),
        "error": result.error,
    }
