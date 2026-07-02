"""抓取编排逻辑：被 Streamlit 页面和后台定时任务共用，不依赖 Streamlit。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from classifier import LLMVerifier
from fetcher import fetch_external_event_news, fetch_sector_news
from news_store import (
    combine_cache_frames,
    deduplicate_cache,
    display_to_cache,
    filter_incremental_news,
    merge_cache,
    read_cache,
    refilter_external_event_cache,
)
from time_utils import utc_now_iso


# 板块/事件类别层面的抓取并发数；每个板块内部的关键词还有一层并发，
# 两层相乘就是对上游接口的最大并发请求数，不宜设得过大。
MAX_GROUP_FETCH_WORKERS = 3


@dataclass
class RefreshOutcome:
    cache: pd.DataFrame
    added_count: int = 0
    sector_warnings: dict[str, list[str]] = field(default_factory=dict)
    event_warnings: dict[str, list[str]] = field(default_factory=dict)
    fetch_failed: bool = False


def fetch_selected_sector_cache(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    llm_verifier: LLMVerifier | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fetched_frames: list[pd.DataFrame] = []
    warnings_by_sector: dict[str, list[str]] = {}
    fetched_at = utc_now_iso()
    if not selected_sectors:
        return combine_cache_frames(fetched_frames), warnings_by_sector

    max_workers = min(MAX_GROUP_FETCH_WORKERS, len(selected_sectors))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            sector: executor.submit(
                fetch_sector_news,
                sector,
                sectors_config[sector],
                llm_verifier=llm_verifier,
            )
            for sector in selected_sectors
        }
        for sector, future in futures.items():
            sector_warnings: list[str] = []
            try:
                result = future.result()
            except Exception as exc:
                warnings_by_sector[sector] = [f"{sector} 抓取失败：{exc}"]
                continue

            if result.error:
                sector_warnings.append(result.error)
            if result.warnings:
                sector_warnings.extend(result.warnings)
            if sector_warnings:
                warnings_by_sector[sector] = sector_warnings

            if not result.data.empty:
                fetched_frames.append(
                    display_to_cache(sector, result.data, fetched_at=fetched_at)
                )

    return combine_cache_frames(fetched_frames), warnings_by_sector


def fetch_external_event_cache(
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fetched_frames: list[pd.DataFrame] = []
    warnings_by_event: dict[str, list[str]] = {}
    fetched_at = utc_now_iso()
    if not external_events:
        return combine_cache_frames(fetched_frames), warnings_by_event

    max_workers = min(MAX_GROUP_FETCH_WORKERS, len(external_events))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            event_category: executor.submit(
                fetch_external_event_news,
                event_category,
                keywords,
                event_to_sectors=event_to_sectors,
                llm_verifier=llm_verifier,
            )
            for event_category, keywords in external_events.items()
        }
        for event_category, future in futures.items():
            event_warnings: list[str] = []
            try:
                result = future.result()
            except Exception as exc:
                warnings_by_event[event_category] = [f"{event_category} 抓取失败：{exc}"]
                continue

            if result.error:
                event_warnings.append(result.error)
            if result.warnings:
                event_warnings.extend(result.warnings)
            if event_warnings:
                warnings_by_event[event_category] = event_warnings

            if not result.data.empty:
                fetched_frames.append(
                    display_to_cache(
                        "",
                        result.data,
                        fetched_at=fetched_at,
                        news_type="external_event",
                    )
                )

    return combine_cache_frames(fetched_frames), warnings_by_event


def fetch_refresh_cache(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    fetched_frames: list[pd.DataFrame] = []
    sector_warnings: dict[str, list[str]] = {}

    if selected_sectors:
        sector_cache, sector_warnings = fetch_selected_sector_cache(
            selected_sectors,
            sectors_config,
            llm_verifier=llm_verifier,
        )
        if not sector_cache.empty:
            fetched_frames.append(sector_cache)

    external_cache, external_warnings = fetch_external_event_cache(
        external_events,
        event_to_sectors,
        llm_verifier=llm_verifier,
    )
    if not external_cache.empty:
        fetched_frames.append(external_cache)

    return combine_cache_frames(fetched_frames), sector_warnings, external_warnings


def run_incremental_refresh(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None = None,
) -> RefreshOutcome:
    existing_result = read_cache()
    existing_cache, cache_refilter_warnings = refilter_external_event_cache(
        existing_result.data,
        external_events,
        sectors_config=sectors_config,
        llm_verifier=llm_verifier,
    )
    fetched_cache, warnings_by_sector, warnings_by_event = fetch_refresh_cache(
        selected_sectors,
        sectors_config,
        external_events,
        event_to_sectors,
        llm_verifier=llm_verifier,
    )
    incremental_cache = filter_incremental_news(existing_cache, fetched_cache)
    merged_cache, _ = merge_cache(existing_cache, incremental_cache)
    merged_cache, merge_refilter_warnings = refilter_external_event_cache(
        merged_cache,
        external_events,
        sectors_config=sectors_config,
        llm_verifier=llm_verifier,
    )
    added_count = max(len(merged_cache) - len(existing_cache), 0)
    cache_warnings = [*cache_refilter_warnings, *merge_refilter_warnings]
    if cache_warnings:
        warnings_by_event.setdefault("缓存重过滤", []).extend(cache_warnings)

    return RefreshOutcome(
        cache=merged_cache,
        added_count=added_count,
        sector_warnings=warnings_by_sector,
        event_warnings=warnings_by_event,
    )


def run_full_refresh(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None = None,
) -> RefreshOutcome:
    fetched_cache, warnings_by_sector, warnings_by_event = fetch_refresh_cache(
        selected_sectors,
        sectors_config,
        external_events,
        event_to_sectors,
        llm_verifier=llm_verifier,
    )
    if fetched_cache.empty and (warnings_by_sector or warnings_by_event):
        return RefreshOutcome(
            cache=fetched_cache,
            sector_warnings=warnings_by_sector,
            event_warnings=warnings_by_event,
            fetch_failed=True,
        )

    rebuilt_cache = deduplicate_cache(fetched_cache)
    rebuilt_cache, cache_refilter_warnings = refilter_external_event_cache(
        rebuilt_cache,
        external_events,
        sectors_config=sectors_config,
        llm_verifier=llm_verifier,
    )
    if cache_refilter_warnings:
        warnings_by_event.setdefault("缓存重过滤", []).extend(cache_refilter_warnings)

    return RefreshOutcome(
        cache=rebuilt_cache,
        added_count=len(rebuilt_cache),
        sector_warnings=warnings_by_sector,
        event_warnings=warnings_by_event,
    )


def prune_old_cache(cache_df: pd.DataFrame, retention_days: int) -> pd.DataFrame:
    """清理超过保留期的旧新闻，避免缓存无限膨胀。

    以发布时间为准；发布时间无法解析时退回抓取时间；两者都无法解析则保留。
    """
    if retention_days <= 0 or cache_df.empty:
        return cache_df

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=retention_days)
    publish_times = pd.to_datetime(cache_df["publish_time"], errors="coerce")
    fetched_times = pd.to_datetime(
        cache_df["fetched_at"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    keep_mask = (
        publish_times.ge(cutoff)
        | (publish_times.isna() & fetched_times.ge(cutoff))
        | (publish_times.isna() & fetched_times.isna())
    )
    return cache_df[keep_mask].reset_index(drop=True)
