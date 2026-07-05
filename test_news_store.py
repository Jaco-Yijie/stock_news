from __future__ import annotations

import pandas as pd

from news_store import filter_incremental_news


def _cache_row(link: str, publish_time: str, sector: str = "半导体芯片") -> dict:
    return {
        "news_type": "sector_news",
        "sector": sector,
        "title": f"新闻 {link}",
        "source": "测试",
        "publish_time": publish_time,
        "link": link,
        "keyword": "芯片",
        "content": "",
        "event_category": "",
        "related_sectors": "",
        "reason": "",
        "fetched_at": "2026-07-05T00:00:00Z",
    }


def test_incremental_keeps_out_of_order_news() -> None:
    """上游返回乱序时，发布时间早于缓存最新时间的新链接也不能漏。"""
    existing = pd.DataFrame([_cache_row("http://e.com/new", "2026-07-05 10:00:00")])
    fetched = pd.DataFrame(
        [
            _cache_row("http://e.com/new", "2026-07-05 10:00:00"),
            _cache_row("http://e.com/older-but-unseen", "2026-07-05 08:00:00"),
        ]
    )
    result = filter_incremental_news(existing, fetched)
    assert list(result["link"]) == ["http://e.com/older-but-unseen"]


def test_incremental_dedupes_by_normalized_link() -> None:
    existing = pd.DataFrame([_cache_row("http://e.com/a/", "2026-07-05 10:00:00")])
    fetched = pd.DataFrame(
        [_cache_row("http://e.com/a?from=share", "2026-07-05 11:00:00")]
    )
    result = filter_incremental_news(existing, fetched)
    assert result.empty


def test_incremental_keeps_rows_without_link_for_later_dedup() -> None:
    existing = pd.DataFrame([_cache_row("http://e.com/a", "2026-07-05 10:00:00")])
    fetched = pd.DataFrame([_cache_row("", "2026-07-05 11:00:00")])
    result = filter_incremental_news(existing, fetched)
    assert len(result) == 1


if __name__ == "__main__":
    test_incremental_keeps_out_of_order_news()
    test_incremental_dedupes_by_normalized_link()
    test_incremental_keeps_rows_without_link_for_later_dedup()
    print("test_news_store.py: ok")
