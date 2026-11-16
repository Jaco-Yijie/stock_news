"""工具：增量过滤。与历史缓存比对，只保留没推送过的新闻。"""
from __future__ import annotations

from typing import Any

import pandas as pd

import news_store

from .schemas import FilterNewNewsInput


def filter_new_news(input: FilterNewNewsInput) -> dict[str, Any]:
    """按链接判重，剔除缓存中已存在的新闻。

    items 为展示层格式（标题/新闻内容/发布时间/原文链接）。
    返回 {"total_in", "kept", "dropped", "items", "error"}，items 保持输入格式。
    缓存读取失败时保守处理：全部保留，并在 error 中说明原因。
    """
    if not input.items:
        return {"total_in": 0, "kept": 0, "dropped": 0, "items": [], "error": None}

    cache = news_store.read_cache()
    if cache.error:
        return {
            "total_in": len(input.items),
            "kept": len(input.items),
            "dropped": 0,
            "items": input.items,
            "error": f"缓存不可用，未做增量过滤：{cache.error}",
        }

    # 缓存内部是英文列（link/title/...），先转到缓存格式再判重，输出转回展示格式
    fetched_cache = news_store.display_to_cache("", pd.DataFrame(input.items))
    filtered = news_store.filter_incremental_news(cache.data, fetched_cache)
    if filtered.empty:
        return {"total_in": len(input.items), "kept": 0,
                "dropped": len(input.items), "items": [], "error": None}

    kept_links = {str(v) for v in filtered["link"]}
    kept = [
        item for item in input.items
        if str(item.get("原文链接", "")) in kept_links
    ]
    # 无链接的行按保守策略保留（与 filter_incremental_news 行为一致）
    if not kept and all(not str(i.get("原文链接", "")) for i in input.items):
        kept = input.items

    return {
        "total_in": len(input.items),
        "kept": len(kept),
        "dropped": len(input.items) - len(kept),
        "items": kept,
        "error": None,
    }
