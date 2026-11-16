"""工具：历史检索。在缓存中查旧闻，支持"这条是否推送过/是否有增量信息"的判断。"""
from __future__ import annotations

from typing import Any

import news_store

from .schemas import SearchHistoryInput

_TEXT_COLUMNS = ("title", "content", "标题", "新闻内容")


def search_history(input: SearchHistoryInput) -> dict[str, Any]:
    """在历史缓存中按关键词检索已收录新闻，按发布时间倒序返回。

    返回 {"query", "count", "items", "error"}。用于：
    - 推送前确认某条新闻是否已推送过（避免重复）；
    - 判断新抓到的"旧闻"是否有增量信息。
    """
    query = input.query.strip()
    if not query:
        return {"query": input.query, "count": 0, "items": [], "error": "query 为空"}

    cache = news_store.read_cache()
    if cache.error:
        return {"query": query, "count": 0, "items": [], "error": cache.error}
    if cache.data.empty:
        return {"query": query, "count": 0, "items": [], "error": None}

    df = cache.data
    keyword = query.casefold()
    mask = False
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.casefold().str.contains(keyword, regex=False)
    hits = df[mask] if not isinstance(mask, bool) else df.iloc[0:0]

    time_col = next((c for c in ("publish_time", "发布时间") if c in hits.columns), None)
    if time_col:
        hits = hits.sort_values(time_col, ascending=False)

    items = hits.head(input.limit).to_dict(orient="records")
    return {"query": query, "count": len(hits), "items": items, "error": None}
