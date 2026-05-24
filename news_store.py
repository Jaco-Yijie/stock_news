from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from classifier import LLMValidationCache, LLMVerifier
from fetcher import deduplicate_news, filter_external_event_news, high_risk_sector_rules
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
        return CacheReadResult(
            data=empty_cache_frame(),
            error=f"读取本地缓存失败：{type(exc).__name__}",
        )

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


def refilter_external_event_cache(
    df: pd.DataFrame,
    external_events: dict[str, list[Any]],
    sectors_config: dict[str, Any] | None = None,
    llm_verifier: LLMVerifier | None = None,
    llm_cache: LLMValidationCache | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    warnings: list[str] = []
    try:
        normalized = _normalize_cache_frame(df)
    except Exception as exc:
        return empty_cache_frame(), (f"缓存重过滤失败，已丢弃缓存数据：{exc}",)

    if normalized.empty:
        return normalized, ()

    if df is not None and not df.empty and "news_type" not in df.columns:
        warnings.append("缓存缺少 news_type 字段，无法识别外部事件缓存，已按普通新闻保留。")

    kept_frames: list[pd.DataFrame] = []
    sector_df = normalized[normalized["news_type"].ne("external_event")]
    if not sector_df.empty:
        for sector, group_df in sector_df.groupby("sector", sort=False):
            sector_name = str(sector or "").strip()
            sector_config = (
                sectors_config.get(sector_name)
                if isinstance(sectors_config, dict)
                else None
            )
            sector_rules = high_risk_sector_rules(sector_name, sector_config)
            if not sector_rules:
                kept_frames.append(group_df)
                continue

            try:
                filtered_df = filter_external_event_news(
                    group_df,
                    sector_name,
                    sector_rules,
                    llm_verifier=llm_verifier,
                    llm_cache=llm_cache,
                )
            except Exception as exc:
                warnings.append(
                    f"高风险板块缓存「{sector_name}」重过滤失败，"
                    f"已丢弃 {len(group_df)} 条缓存新闻：{exc}"
                )
                continue

            kept_count = len(filtered_df) if not filtered_df.empty else 0
            dropped_count = len(group_df) - kept_count
            if dropped_count > 0:
                warnings.append(
                    f"高风险板块缓存「{sector_name}」已按新规则过滤 {dropped_count} 条旧缓存新闻。"
                )
            if kept_count > 0:
                kept_frames.append(_normalize_cache_frame(filtered_df))

    external_df = normalized[normalized["news_type"].eq("external_event")]
    if external_df.empty:
        return combine_cache_frames(kept_frames), tuple(warnings)

    if not isinstance(external_events, dict):
        warnings.append("外部事件规则配置无效，已丢弃缓存中的外部事件新闻。")
        return combine_cache_frames(kept_frames), tuple(warnings)

    for event_category, category_df in external_df.groupby("event_category", sort=False):
        category_name = str(event_category or "").strip()
        if not category_name or category_name not in external_events:
            warnings.append(
                f"外部事件缓存类别「{category_name or '未分类'}」缺少规则，"
                f"已丢弃 {len(category_df)} 条缓存新闻。"
            )
            continue

        try:
            filtered_df = filter_external_event_news(
                category_df,
                category_name,
                external_events[category_name],
                llm_verifier=llm_verifier,
                llm_cache=llm_cache,
            )
        except Exception as exc:
            warnings.append(
                f"外部事件缓存类别「{category_name}」重过滤失败，"
                f"已丢弃 {len(category_df)} 条缓存新闻：{exc}"
            )
            continue

        kept_count = len(filtered_df) if not filtered_df.empty else 0
        dropped_count = len(category_df) - kept_count
        if dropped_count > 0:
            warnings.append(
                f"外部事件缓存类别「{category_name}」已按新规则过滤 {dropped_count} 条旧缓存新闻。"
            )
        if kept_count > 0:
            kept_frames.append(_normalize_cache_frame(filtered_df))

    return combine_cache_frames(kept_frames), tuple(warnings)


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
