from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from fetcher import deduplicate_news
from paths import DATA_DIR


CACHE_PATH = DATA_DIR / "news_cache.csv"
CACHE_COLUMNS = [
    "news_type",
    "sector",
    "title",
    "source",
    "publish_time",
    "link",
    "keyword",
    "content",
    "event_category",
    "related_sectors",
    "reason",
    "fetched_at",
]
DISPLAY_COLUMNS = [
    "news_type",
    "sector",
    "标题",
    "来源媒体",
    "发布时间",
    "原文链接",
    "匹配关键词",
    "新闻内容",
    "event_category",
    "related_sectors",
    "reason",
]


@dataclass(frozen=True)
class CacheReadResult:
    data: pd.DataFrame
    error: str | None = None


def empty_cache_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_COLUMNS)


def empty_display_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DISPLAY_COLUMNS)


def _normalize_cache_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_cache_frame()

    normalized = df.copy()
    for column in CACHE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized = normalized[CACHE_COLUMNS]
    for column in CACHE_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)
    normalized["news_type"] = normalized["news_type"].replace("", "sector_news")

    return normalized.reset_index(drop=True)


def read_cache(path: Path = CACHE_PATH) -> CacheReadResult:
    if not path.exists() or path.stat().st_size == 0:
        return CacheReadResult(data=empty_cache_frame())

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        return CacheReadResult(data=empty_cache_frame(), error=f"读取本地缓存失败：{exc}")

    return CacheReadResult(data=_normalize_cache_frame(df))


def save_cache(df: pd.DataFrame, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_cache_frame(df)
    tmp_path = path.with_name(f"{path.name}.tmp")
    normalized.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def clear_cache(path: Path = CACHE_PATH) -> None:
    if path.exists():
        path.unlink()


def cache_metadata(df: pd.DataFrame) -> dict[str, str | int]:
    normalized = _normalize_cache_frame(df)
    if normalized.empty:
        return {
            "total": 0,
            "latest_fetched_at": "无",
            "latest_publish_time": "无",
        }

    fetched_times = pd.to_datetime(normalized["fetched_at"], errors="coerce")
    publish_times = pd.to_datetime(normalized["publish_time"], errors="coerce")
    latest_fetched_at = fetched_times.max()
    latest_publish_time = publish_times.max()

    return {
        "total": len(normalized),
        "latest_fetched_at": (
            latest_fetched_at.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(latest_fetched_at)
            else "无"
        ),
        "latest_publish_time": (
            latest_publish_time.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(latest_publish_time)
            else "无"
        ),
    }


def cache_to_display(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_cache_frame(df)
    if normalized.empty:
        return empty_display_frame()

    display_df = pd.DataFrame(
        {
            "news_type": normalized["news_type"],
            "sector": normalized["sector"],
            "标题": normalized["title"],
            "来源媒体": normalized["source"],
            "发布时间": normalized["publish_time"],
            "原文链接": normalized["link"],
            "匹配关键词": normalized["keyword"],
            "新闻内容": normalized["content"],
            "event_category": normalized["event_category"],
            "related_sectors": normalized["related_sectors"],
            "reason": normalized["reason"],
        }
    )
    return display_df[DISPLAY_COLUMNS].reset_index(drop=True)


def display_to_cache(
    sector: str,
    news_df: pd.DataFrame,
    fetched_at: str | None = None,
    news_type: str = "sector_news",
    event_category: str = "",
    related_sectors: str = "",
    reason: str = "",
) -> pd.DataFrame:
    if fetched_at is None:
        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if news_df is None or news_df.empty:
        return empty_cache_frame()

    def get_column(column: str) -> pd.Series:
        if column not in news_df.columns:
            return pd.Series([""] * len(news_df), index=news_df.index, dtype="object")
        return news_df[column].fillna("").astype(str)

    def get_column_or_default(column: str, default: str) -> pd.Series:
        if column not in news_df.columns:
            return pd.Series([default] * len(news_df), index=news_df.index, dtype="object")
        return news_df[column].fillna(default).replace("", default).astype(str)

    cache_df = pd.DataFrame(
        {
            "news_type": get_column_or_default("news_type", news_type),
            "sector": get_column_or_default("sector", sector),
            "title": get_column("标题"),
            "source": get_column("来源媒体"),
            "publish_time": get_column("发布时间"),
            "link": get_column("原文链接"),
            "keyword": get_column("匹配关键词"),
            "content": get_column("新闻内容"),
            "event_category": get_column_or_default("event_category", event_category),
            "related_sectors": get_column_or_default("related_sectors", related_sectors),
            "reason": get_column_or_default("reason", reason),
            "fetched_at": fetched_at,
        }
    )
    return _normalize_cache_frame(cache_df)


def combine_cache_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    normalized_frames = [
        _normalize_cache_frame(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not normalized_frames:
        return empty_cache_frame()
    return _normalize_cache_frame(pd.concat(normalized_frames, ignore_index=True))


def deduplicate_cache(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_cache_frame(df)
    if normalized.empty:
        return normalized

    deduped_frames: list[pd.DataFrame] = []
    sector_news_df = normalized[normalized["news_type"].ne("external_event")]
    for sector, sector_df in sector_news_df.groupby("sector", sort=False):
        display_df = cache_to_display(sector_df)
        deduped_display = deduplicate_news(display_df.drop(columns=["sector"]))
        deduped_frames.append(
            display_to_cache(sector, deduped_display, news_type="sector_news")
        )

    external_df = normalized[normalized["news_type"].eq("external_event")]
    if not external_df.empty:
        external_display = cache_to_display(external_df)
        deduped_external = deduplicate_news(external_display.drop(columns=["sector"]))
        deduped_frames.append(
            display_to_cache("", deduped_external, news_type="external_event")
        )

    return combine_cache_frames(deduped_frames)


def _incremental_group_key(row: pd.Series) -> str:
    news_type = str(row.get("news_type", "sector_news") or "sector_news")
    if news_type == "external_event":
        category = str(row.get("event_category", "") or "未分类")
        return f"{news_type}:{category}"
    sector = str(row.get("sector", "") or "未分类")
    return f"{news_type}:{sector}"


def filter_incremental_news(
    existing_cache: pd.DataFrame,
    fetched_cache: pd.DataFrame,
) -> pd.DataFrame:
    existing = _normalize_cache_frame(existing_cache)
    fetched = _normalize_cache_frame(fetched_cache)
    if fetched.empty:
        return fetched
    if existing.empty:
        return fetched

    existing_with_time = existing.copy()
    existing_with_time["_publish_dt"] = pd.to_datetime(
        existing_with_time["publish_time"], errors="coerce"
    )
    existing_with_time["_incremental_group"] = existing_with_time.apply(
        _incremental_group_key, axis=1
    )
    latest_by_group = existing_with_time.groupby("_incremental_group")["_publish_dt"].max()

    fetched_with_time = fetched.copy()
    fetched_with_time["_publish_dt"] = pd.to_datetime(
        fetched_with_time["publish_time"], errors="coerce"
    )
    fetched_with_time["_incremental_group"] = fetched_with_time.apply(
        _incremental_group_key, axis=1
    )

    keep_mask = []
    for _, row in fetched_with_time.iterrows():
        publish_dt = row["_publish_dt"]
        latest_dt = latest_by_group.get(row["_incremental_group"], pd.NaT)
        if pd.isna(publish_dt) or pd.isna(latest_dt):
            keep_mask.append(True)
        else:
            keep_mask.append(publish_dt > latest_dt)

    kept = fetched_with_time[keep_mask].drop(columns=["_publish_dt", "_incremental_group"])
    return _normalize_cache_frame(kept)


def merge_cache(
    existing_cache: pd.DataFrame,
    new_cache: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    existing_deduped = deduplicate_cache(existing_cache)
    merged = deduplicate_cache(combine_cache_frames([existing_deduped, new_cache]))
    added_count = max(len(merged) - len(existing_deduped), 0)
    return merged, added_count
