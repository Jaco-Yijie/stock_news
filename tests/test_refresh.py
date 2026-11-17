from __future__ import annotations

import pandas as pd

import refresh
import refresh_job
from news_store import CacheReadResult, display_to_cache, empty_cache_frame


def _sector_news(sector, title, slug):
    return display_to_cache(sector, pd.DataFrame([{
        "标题": title, "原文链接": f"https://example.com/{slug}",
    }]), fetched_at="2026-11-16T00:00:00Z")


def test_full_refresh_preserves_unselected_and_failed_sectors(monkeypatch):
    existing = pd.concat([
        _sector_news("光伏", "旧光伏行业新闻", "pv-old"),
        _sector_news("风电", "待保留的风电政策", "wind-old"),
        _sector_news("机器人", "机器人行业原有数据", "bot-old"),
    ], ignore_index=True)
    fresh = _sector_news("光伏", "新光伏项目中标", "pv-new")
    monkeypatch.setattr(refresh, "read_cache", lambda: CacheReadResult(existing))
    monkeypatch.setattr(refresh, "fetch_refresh_cache", lambda *args, **kwargs: (
        fresh, {"风电": ["抓取超时"]}, {},
    ))

    outcome = refresh.run_full_refresh(["光伏", "风电"], {}, {}, {}, preserve_unselected=True)
    assert set(outcome.cache["title"]) == {"新光伏项目中标", "待保留的风电政策", "机器人行业原有数据"}
    assert outcome.base_cache is existing and not outcome.fetch_failed


def test_empty_full_refresh_cannot_replace_existing_data(monkeypatch):
    monkeypatch.setattr(refresh, "fetch_refresh_cache", lambda *args, **kwargs: (
        empty_cache_frame(), {}, {},
    ))
    assert refresh.run_full_refresh(["光伏"], {}, {}, {}).fetch_failed


def test_incremental_refilter_exception_does_not_discard_snapshot(monkeypatch):
    existing = _sector_news("光伏", "应保留的历史新闻", "old")
    monkeypatch.setattr(refresh, "read_cache", lambda: CacheReadResult(existing))
    monkeypatch.setattr(refresh, "refilter_external_event_cache", lambda *args, **kwargs: (
        empty_cache_frame(), ("高风险板块缓存重过滤失败",),
    ))
    outcome = refresh.run_incremental_refresh(["光伏"], {}, {}, {})
    assert outcome.read_error and outcome.cache is existing


def test_cloud_refresh_without_supabase_fails_before_fetching(monkeypatch, capsys):
    monkeypatch.setattr(refresh_job, "cache_backend_name", lambda: "本地文件")
    assert refresh_job.main(["--require-supabase"]) == 1
    assert "缺少 Supabase" in capsys.readouterr().out
