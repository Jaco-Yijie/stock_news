from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from classifier import LLMVerifier, extract_rule_keywords
from config_store import (
    add_event_category,
    add_event_keyword,
    add_keyword,
    add_sector,
    extract_sector_keywords,
    remove_event_category,
    remove_event_keyword,
    remove_keyword,
    remove_sector,
    reset_events_config,
    reset_sectors_config,
    try_load_events_config,
    try_load_sectors_config,
    update_event_related_sectors,
)
from fetcher import (
    SectorResult,
    deduplicate_news,
    fetch_external_event_news,
    fetch_sector_news,
)
from llm_provider import load_llm_verifier_from_env
from news_store import (
    cache_metadata,
    cache_to_display,
    clear_cache,
    combine_cache_frames,
    deduplicate_cache,
    display_to_cache,
    filter_incremental_news,
    merge_cache,
    read_cache,
    refilter_external_event_cache,
    save_cache,
)
from sectors import SECTORS as DEFAULT_SECTORS
from time_utils import format_utc8_time, utc_now_iso


st.set_page_config(page_title="A股板块新闻", page_icon="📰", layout="wide")
DEFAULT_SELECTED_SECTORS = ("半导体芯片",)
TIME_RANGE_OPTIONS = ("全部", "今天", "近 3 天", "本周")
SECTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "科技": (
        "半导体芯片",
        "人工智能",
        "算力数据中心",
        "信创软件",
        "网络安全",
        "消费电子",
        "量子计算",
    ),
    "新兴产业": ("商业航天", "低空经济", "机器人", "脑机接口"),
    "新能源": ("新能源汽车", "光伏", "风电", "储能", "电力设备"),
    "医药": ("创新药", "医药医疗", "CXO"),
    "消费": ("白酒消费", "家电", "旅游酒店", "传媒游戏"),
    "金融地产": ("银行", "证券", "保险", "房地产"),
    "周期资源": (
        "有色金属",
        "黄金",
        "煤炭",
        "钢铁",
        "化工",
        "农业",
        "航运港口",
        "物流快递",
        "环保水务",
    ),
}


def get_llm_verifier() -> tuple[LLMVerifier | None, str | None]:
    return load_llm_verifier_from_env()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-soft: #f3f4f6;
            --border: #e5e7eb;
            --border-strong: #d1d5db;
            --text: #111827;
            --muted: #6b7280;
            --muted-soft: #9ca3af;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --success: #0f766e;
            --warning: #92400e;
            --warning-bg: #fffbeb;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            height: 1rem !important;
            min-height: 1rem !important;
            background: transparent !important;
            z-index: 990 !important;
        }

        div[data-testid="stDecoration"] {
            display: none !important;
            height: 0 !important;
        }

        div[data-testid="stToolbar"] {
            top: 0.25rem !important;
            right: 0.75rem !important;
            z-index: 991 !important;
        }

        section.main {
            padding-top: 0 !important;
        }

        .block-container {
            padding-top: 1rem !important;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }

        main .block-container {
            padding-top: 1rem !important;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
            min-width: 292px !important;
            max-width: 320px !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
            letter-spacing: 0;
            font-size: 14px;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 1rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.5rem;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: #e0f2fe;
            border: 1px solid #bae6fd;
            color: #0f172a;
            font-weight: 600;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: #0f172a;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: #ffffff;
            border-color: var(--border);
            border-radius: 8px;
            box-shadow: none;
        }

        [data-baseweb="select"] > div:hover,
        [data-baseweb="input"] > div:hover {
            border-color: var(--border-strong);
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        [data-testid="stCheckbox"] {
            min-height: 1.75rem;
        }

        [data-testid="stCheckbox"] label {
            gap: 0.35rem;
            color: var(--text);
            font-size: 0.9rem;
        }

        [data-testid="stCheckbox"] div[role="checkbox"] {
            border-color: #cbd5e1 !important;
        }

        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        div.stButton > button {
            width: 100%;
            border: 1px solid var(--border-strong);
            border-radius: 8px;
            background: #ffffff;
            color: #111827;
            box-shadow: none;
            font-weight: 600;
        }

        div.stButton > button:hover {
            border-color: #9ca3af;
            background: #f9fafb;
            color: #111827;
        }

        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1.5rem;
            padding: 0.2rem 0 1rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }

        .dashboard-title {
            margin: 0;
            font-size: 24px;
            line-height: 1.2;
            letter-spacing: 0;
            color: var(--text);
            font-weight: 650;
        }

        .dashboard-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
            min-width: 620px;
        }

        .metric-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 0.72rem 0.82rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
            white-space: nowrap;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.06rem;
            font-weight: 700;
        }

        .sector-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 0.8rem 0 1rem;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .sector-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.82rem 0.95rem;
            border-bottom: 1px solid var(--border);
            background: #fbfdff;
        }

        .sector-name {
            color: var(--text);
            font-size: 18px;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        .sector-meta {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #f9fafb;
            color: var(--muted);
            padding: 0.18rem 0.52rem;
            font-size: 0.75rem;
            white-space: nowrap;
        }

        .pill.warn {
            border-color: #fde68a;
            background: var(--warning-bg);
            color: var(--warning);
        }

        .section-warning {
            color: var(--warning);
            background: var(--warning-bg);
            border-bottom: 1px solid #fde68a;
            padding: 0.62rem 0.95rem;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }

        .news-list {
            padding: 0.3rem 0;
        }

        .news-card {
            border-bottom: 1px solid var(--border);
            background: var(--surface);
            padding: 14px 16px;
            transition: background-color 120ms ease;
        }

        .news-card:last-child {
            border-bottom: 0;
        }

        .news-card:hover {
            background: #f9fafb;
        }

        .news-title {
            margin: 0 0 6px;
            color: var(--text);
            font-size: 18px;
            line-height: 1.45;
            font-weight: 600;
            overflow-wrap: anywhere;
            word-break: break-word;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .news-meta {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 5px 8px;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
        }

        .keyword-tag {
            border-radius: 6px;
            background: var(--accent-soft);
            border: 1px solid #dbeafe;
            color: #1d4ed8;
            padding: 1px 6px;
            font-size: 12px;
            font-weight: 500;
        }

        .news-link {
            margin-left: auto;
            color: var(--accent) !important;
            text-decoration: none !important;
            font-weight: 600;
            white-space: nowrap;
        }

        .news-link:hover {
            text-decoration: underline !important;
        }

        .empty-state {
            padding: 0.9rem 0.95rem;
            color: var(--muted);
            background: #ffffff;
        }

        @media (max-width: 900px) {
            .dashboard-header {
                flex-direction: column;
            }

            .metric-grid {
                grid-template-columns: 1fr;
                min-width: 0;
                width: 100%;
            }

            .sector-heading {
                align-items: flex-start;
                flex-direction: column;
            }

            .news-link {
                margin-left: 0;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 0.9rem 0.75rem 2rem;
            }

            [data-testid="stSidebar"] {
                min-width: 260px !important;
                max-width: 300px !important;
            }

            .dashboard-header {
                gap: 0.75rem;
                padding-bottom: 0.85rem;
            }

            .dashboard-title {
                font-size: 22px;
            }

            .dashboard-subtitle {
                font-size: 13px;
            }

            .metric-card {
                padding: 10px 12px;
            }

            .metric-label {
                font-size: 12px;
            }

            .metric-value {
                font-size: 15px;
            }

            .sector-heading {
                padding: 12px;
                gap: 8px;
            }

            .sector-name {
                font-size: 17px;
            }

            .pill {
                font-size: 12px;
                padding: 2px 7px;
            }

            .news-card {
                padding: 12px;
            }

            .news-title {
                font-size: 16px;
                line-height: 1.45;
                -webkit-line-clamp: 3;
            }

            .news-meta {
                align-items: flex-start;
                gap: 4px 8px;
                font-size: 12px;
            }

            .keyword-tag {
                font-size: 12px;
                padding: 1px 5px;
            }

            .news-link {
                flex-basis: 100%;
                margin-left: 0;
                padding-top: 2px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def llm_verifier_cache_key(llm_verifier: LLMVerifier | None) -> str:
    if llm_verifier is None:
        return "none"
    return "|".join(
        str(getattr(llm_verifier, attr, ""))
        for attr in ("provider", "model", "prompt_version", "rule_version")
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_display_data(
    cache_version: int,
    sectors_config: dict[str, Any],
    external_events: dict[str, Any],
    llm_cache_key: str,
    _llm_verifier: LLMVerifier | None,
):
    """读取缓存并完成重过滤、格式转换、去重。

    整条流水线只在缓存版本或配置变化时重算，
    普通页面交互（筛选、搜索、翻页等）直接复用结果。
    """
    cache_result = read_cache()
    cache_df, refilter_warnings = refilter_external_event_cache(
        cache_result.data,
        external_events,
        sectors_config=sectors_config,
        llm_verifier=_llm_verifier,
    )
    display_df = cache_to_display(cache_df)
    sector_display_df = deduplicate_display_news(
        display_df[display_df["news_type"].ne("external_event")].reset_index(drop=True),
        group_column="sector",
    )
    external_display_df = deduplicate_display_news(
        display_df[display_df["news_type"].eq("external_event")].reset_index(drop=True)
    )
    return (
        display_df,
        sector_display_df,
        external_display_df,
        cache_metadata(cache_df),
        cache_result.error,
        refilter_warnings,
    )


def parse_keyword_input(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").replace("，", ",").split(",")
        if item.strip()
    ]


def fetch_selected_sector_cache(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    llm_verifier: LLMVerifier | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fetched_frames: list[pd.DataFrame] = []
    warnings_by_sector: dict[str, list[str]] = {}
    fetched_at = utc_now_iso()

    for sector in selected_sectors:
        sector_warnings: list[str] = []
        try:
            result = fetch_sector_news(
                sector,
                sectors_config[sector],
                llm_verifier=llm_verifier,
            )
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
            fetched_frames.append(display_to_cache(sector, result.data, fetched_at=fetched_at))

    return combine_cache_frames(fetched_frames), warnings_by_sector


def fetch_external_event_cache(
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fetched_frames: list[pd.DataFrame] = []
    warnings_by_event: dict[str, list[str]] = {}
    fetched_at = utc_now_iso()

    for event_category, keywords in external_events.items():
        event_warnings: list[str] = []
        try:
            result = fetch_external_event_news(
                event_category,
                keywords,
                event_to_sectors=event_to_sectors,
                llm_verifier=llm_verifier,
            )
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


def collect_sources(news_df: pd.DataFrame) -> list[str]:
    sources: set[str] = set()
    if "来源媒体" not in news_df.columns or news_df.empty:
        return []
    sources.update(source for source in news_df["来源媒体"].dropna().astype(str) if source)
    return sorted(sources)


def filter_news_by_time(news_df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    if news_df.empty or time_range == "全部":
        return news_df.reset_index(drop=True)
    if "发布时间" not in news_df.columns:
        return news_df.iloc[0:0].reset_index(drop=True)

    publish_times = pd.to_datetime(news_df["发布时间"], errors="coerce")
    now = pd.Timestamp.now()
    if time_range == "今天":
        start_at = now.normalize()
    elif time_range == "近 3 天":
        start_at = now - pd.Timedelta(days=3)
    elif time_range == "本周":
        start_at = now.normalize() - pd.Timedelta(days=now.weekday())
    else:
        return news_df.reset_index(drop=True)

    try:
        keep_mask = publish_times.ge(start_at)
    except TypeError:
        publish_times = pd.to_datetime(
            news_df["发布时间"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
        keep_mask = publish_times.ge(start_at)

    return news_df[keep_mask.fillna(False)].reset_index(drop=True)


def deduplicate_display_news(
    news_df: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    if news_df.empty:
        return news_df.reset_index(drop=True)
    if not group_column or group_column not in news_df.columns:
        return deduplicate_news(news_df).reset_index(drop=True)

    deduped_frames = [
        deduplicate_news(group_df)
        for _, group_df in news_df.groupby(group_column, sort=False)
    ]
    if not deduped_frames:
        return news_df.iloc[0:0].reset_index(drop=True)
    return pd.concat(deduped_frames, ignore_index=True).reset_index(drop=True)


def grouped_sectors(sectors_config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    grouped: list[tuple[str, list[str]]] = []
    assigned_sectors: set[str] = set()

    for group_name, sector_names in SECTOR_GROUPS.items():
        sectors = [sector for sector in sector_names if sector in sectors_config]
        if sectors:
            grouped.append((group_name, sectors))
            assigned_sectors.update(sectors)

    remaining_sectors = [sector for sector in sectors_config if sector not in assigned_sectors]
    if remaining_sectors:
        grouped.append(("其他", remaining_sectors))

    return grouped


def sector_matches_query(sector: str, sector_keywords: Any, query: str) -> bool:
    if not query:
        return True
    return query in sector.casefold() or any(
        query in item.casefold() for item in extract_sector_keywords(sector_keywords)
    )


def render_sector_selector(sectors_config: dict[str, Any], sector_query: str) -> list[str]:
    normalized_query = sector_query.strip().casefold()
    visible_sectors = [
        sector
        for sector, sector_keywords in sectors_config.items()
        if sector_matches_query(sector, sector_keywords, normalized_query)
    ]
    selected_count = sum(
        1
        for sector in sectors_config
        if st.session_state.get(f"sector_selected::{sector}", False)
    )
    st.sidebar.caption(
        f"板块选择：已选 {selected_count} · 可见 {len(visible_sectors)} · 共 {len(sectors_config)}"
    )

    visible_set = set(visible_sectors)
    has_visible_group = False
    for group_name, sectors in grouped_sectors(sectors_config):
        visible_group_sectors = [sector for sector in sectors if sector in visible_set]
        if not visible_group_sectors:
            continue

        has_visible_group = True
        selected_in_group = sum(
            1
            for sector in sectors
            if st.session_state.get(f"sector_selected::{sector}", False)
        )
        expanded = bool(normalized_query) or selected_in_group > 0
        with st.sidebar.expander(
            f"{group_name}（{selected_in_group}/{len(sectors)}）",
            expanded=expanded,
        ):
            for sector in visible_group_sectors:
                st.checkbox(sector, key=f"sector_selected::{sector}")

    if not has_visible_group:
        st.sidebar.caption("没有匹配的板块。")

    return visible_sectors


def filter_news(news_df: pd.DataFrame, keyword: str, selected_sources: list[str]) -> pd.DataFrame:
    filtered_df = news_df

    if keyword:
        keyword = keyword.strip().casefold()
        def match_column(column: str) -> pd.Series:
            if column not in filtered_df.columns:
                return pd.Series([""] * len(filtered_df), index=filtered_df.index)
            return filtered_df[column].fillna("").astype(str)

        match_text = (
            match_column("标题")
            + " "
            + match_column("匹配关键词")
            + " "
            + match_column("来源媒体")
            + " "
            + match_column("event_category")
            + " "
            + match_column("related_sectors")
        ).str.casefold()
        filtered_df = filtered_df[match_text.str.contains(keyword, regex=False)]

    if selected_sources:
        filtered_df = filtered_df[filtered_df["来源媒体"].isin(selected_sources)]

    return filtered_df.reset_index(drop=True)


def render_news_item(row: pd.Series) -> str:
    title = escape(str(row.get("标题", "")))
    source = escape(str(row.get("来源媒体", "")))
    publish_time = escape(str(row.get("发布时间", "")))
    keyword = escape(str(row.get("匹配关键词", "")))
    link = escape(str(row.get("原文链接", "")), quote=True)

    return f"""
        <article class="news-card">
            <div class="news-title">{title}</div>
            <div class="news-meta">
                <span>来源媒体：{source}</span>
                <span>发布时间：{publish_time}</span>
                <span class="keyword-tag">{keyword}</span>
                <a class="news-link" href="{link}" target="_blank" rel="noopener noreferrer">打开原文</a>
            </div>
        </article>
        """


def render_external_event_item(row: pd.Series) -> str:
    title = escape(str(row.get("标题", "")))
    source = escape(str(row.get("来源媒体", "")))
    publish_time = escape(str(row.get("发布时间", "")))
    keyword = escape(str(row.get("匹配关键词", "")))
    event_category = escape(str(row.get("event_category", "")))
    related_sectors = escape(str(row.get("related_sectors", "") or "未映射"))
    link = escape(str(row.get("原文链接", "")), quote=True)

    return f"""
        <article class="news-card">
            <div class="news-title">{title}</div>
            <div class="news-meta">
                <span>来源媒体：{source}</span>
                <span>发布时间：{publish_time}</span>
                <span class="keyword-tag">{event_category}</span>
                <span class="keyword-tag">{keyword}</span>
                <span>可能影响板块：{related_sectors}</span>
                <a class="news-link" href="{link}" target="_blank" rel="noopener noreferrer">打开原文</a>
            </div>
        </article>
        """


def show_sector_section(
    sector: str,
    result,
    keyword: str,
    selected_sources: list[str],
    max_items: int,
) -> None:
    filtered_df = filter_news(result.data, keyword, selected_sources)
    warning_count = len(result.warnings) + (1 if result.error else 0)
    warning_html = ""
    if result.warnings:
        warning_html += (
            '<div class="section-warning">部分关键词抓取失败：'
            + escape("；".join(result.warnings))
            + "</div>"
        )
    if result.error:
        warning_html += f'<div class="section-warning">{escape(result.error)}</div>'

    if result.error:
        body_html = warning_html
    elif filtered_df.empty:
        body_html = warning_html + '<div class="empty-state">当前筛选条件下暂无新闻。</div>'
    else:
        news_html = "".join(
            render_news_item(row) for _, row in filtered_df.head(max_items).iterrows()
        )
        body_html = warning_html + f'<div class="news-list">{news_html}</div>'

    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">{escape(sector)}</div>
                <div class="sector-meta">
                    <span class="pill">去重后 {len(result.data)} 条</span>
                    <span class="pill">当前显示 {len(filtered_df)} 条</span>
                    <span class="pill warn">warning {warning_count}</span>
                </div>
            </div>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_external_events_section(
    external_df: pd.DataFrame,
    warnings_by_event: dict[str, list[str]],
    keyword: str,
    selected_sources: list[str],
    max_items: int,
) -> None:
    filtered_df = filter_news(external_df, keyword, selected_sources)
    warning_items = [
        f"{event_category}: {'；'.join(warnings)}"
        for event_category, warnings in warnings_by_event.items()
        if warnings
    ]
    warning_html = ""
    if warning_items:
        warning_html = (
            '<div class="section-warning">部分外部事件抓取失败：'
            + escape("；".join(warning_items))
            + "</div>"
        )

    if filtered_df.empty:
        body_html = warning_html + '<div class="empty-state">当前筛选条件下暂无外部事件新闻。</div>'
    else:
        news_html = "".join(
            render_external_event_item(row)
            for _, row in filtered_df.head(max_items).iterrows()
        )
        body_html = warning_html + f'<div class="news-list">{news_html}</div>'

    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">外部事件</div>
                <div class="sector-meta">
                    <span class="pill">去重后 {len(external_df)} 条</span>
                    <span class="pill">当前显示 {len(filtered_df)} 条</span>
                    <span class="pill warn">warning {len(warning_items)}</span>
                </div>
            </div>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header(
    selected_count: int,
    displayed_total: int,
    cache_total: int,
    latest_cache_at: str,
    added_count: int,
) -> None:
    st.markdown(
        f"""
        <section class="dashboard-header">
            <div>
                <div class="dashboard-title">A股板块新闻</div>
                <div class="dashboard-subtitle">按板块聚合东方财富新闻，支持筛选与去重</div>
            </div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">覆盖板块数</div>
                    <div class="metric-value">{selected_count}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">缓存新闻总数</div>
                    <div class="metric-value">{cache_total}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">当前新闻数</div>
                    <div class="metric-value">{displayed_total}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">本次新增</div>
                    <div class="metric-value">{added_count}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最近缓存更新</div>
                    <div class="metric-value">{escape(latest_cache_at)}</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def set_config_notice(message: str) -> None:
    st.session_state.config_notice = message


def show_config_notice() -> None:
    notice = st.session_state.pop("config_notice", "")
    if notice:
        st.sidebar.success(notice)


def render_sector_config_manager(sectors_config: dict[str, Any]) -> None:
    with st.sidebar.expander("关键词管理", expanded=False):
        sector_options = list(sectors_config)
        if not sector_options:
            st.info("暂无板块配置。")
            return

        selected_sector = st.selectbox("选择板块", sector_options, key="sector_config_select")
        keywords = extract_sector_keywords(sectors_config.get(selected_sector, []))
        st.caption("当前关键词：" + ("、".join(keywords) if keywords else "无"))

        new_keyword = st.text_input("新增关键词", key="new_sector_keyword")
        if st.button("添加关键词", key="add_sector_keyword"):
            try:
                add_keyword(selected_sector, new_keyword)
                set_config_notice("关键词已保存。请点击“增量刷新”或“全量刷新”更新新闻。")
                st.rerun()
            except Exception as exc:
                st.warning(str(exc))

        if keywords:
            keyword_to_remove = st.selectbox(
                "删除关键词",
                keywords,
                key="remove_sector_keyword_select",
            )
            if st.button("删除选中关键词", key="remove_sector_keyword"):
                remove_keyword(selected_sector, keyword_to_remove)
                set_config_notice("关键词已删除。请点击“增量刷新”或“全量刷新”更新新闻。")
                st.rerun()

        st.divider()
        new_sector_name = st.text_input("新增板块名称", key="new_sector_name")
        new_sector_keywords = st.text_input(
            "新增板块关键词",
            placeholder="多个关键词用逗号分隔",
            key="new_sector_keywords",
        )
        if st.button("新增板块", key="add_sector"):
            keywords_to_add = parse_keyword_input(new_sector_keywords)
            if not keywords_to_add:
                st.warning("新增板块至少需要 1 个关键词。")
            else:
                try:
                    add_sector(new_sector_name, keywords_to_add)
                    set_config_notice("板块已保存。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()
                except Exception as exc:
                    st.warning(str(exc))

        custom_sectors = [sector for sector in sectors_config if sector not in DEFAULT_SECTORS]
        if custom_sectors:
            sector_to_remove = st.selectbox(
                "删除自定义板块",
                custom_sectors,
                key="remove_custom_sector_select",
            )
            confirm_remove_sector = st.checkbox(
                "确认删除该自定义板块",
                key="confirm_remove_custom_sector",
            )
            if st.button("删除自定义板块", key="remove_custom_sector"):
                if confirm_remove_sector:
                    remove_sector(sector_to_remove)
                    set_config_notice("自定义板块已删除。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()
                else:
                    st.warning("请先勾选确认项。")
        else:
            st.caption("当前没有自定义板块。")

        st.divider()
        confirm_reset = st.checkbox("确认恢复默认板块配置", key="confirm_reset_sectors")
        if st.button("恢复默认板块配置", key="reset_sector_config"):
            if confirm_reset:
                reset_sectors_config()
                set_config_notice("板块配置已恢复默认。请点击“增量刷新”或“全量刷新”更新新闻。")
                st.rerun()
            else:
                st.warning("请先勾选确认项。")


def render_event_config_manager(
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    sectors_config: dict[str, Any],
) -> None:
    with st.sidebar.expander("外部事件管理", expanded=False):
        event_options = list(external_events)
        if not event_options:
            st.info("暂无外部事件配置。")
        else:
            selected_event = st.selectbox("选择事件类别", event_options, key="event_config_select")
            event_keywords = extract_rule_keywords(external_events.get(selected_event, []))
            related_sectors = event_to_sectors.get(selected_event, [])
            st.caption("当前事件关键词：" + ("、".join(event_keywords) if event_keywords else "无"))
            st.caption("当前影响板块：" + ("、".join(related_sectors) if related_sectors else "未映射"))

            new_event_keyword = st.text_input("新增事件关键词", key="new_event_keyword")
            if st.button("添加事件关键词", key="add_event_keyword"):
                try:
                    add_event_keyword(selected_event, new_event_keyword)
                    set_config_notice("事件关键词已保存。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()
                except Exception as exc:
                    st.warning(str(exc))

            if event_keywords:
                event_keyword_to_remove = st.selectbox(
                    "删除事件关键词",
                    event_keywords,
                    key="remove_event_keyword_select",
                )
                if st.button("删除选中事件关键词", key="remove_event_keyword"):
                    remove_event_keyword(selected_event, event_keyword_to_remove)
                    set_config_notice("事件关键词已删除。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()

            related_options = sorted(set(sectors_config) | set(related_sectors))
            updated_related_sectors = st.multiselect(
                "编辑可能影响板块",
                options=related_options,
                default=[sector for sector in related_sectors if sector in related_options],
                key="event_related_sectors",
            )
            if st.button("保存影响板块", key="save_event_related_sectors"):
                update_event_related_sectors(selected_event, updated_related_sectors)
                set_config_notice("事件映射已保存。请点击“增量刷新”或“全量刷新”更新新闻。")
                st.rerun()

            confirm_remove_event = st.checkbox("确认删除该事件类别", key="confirm_remove_event")
            if st.button("删除事件类别", key="remove_event_category"):
                if confirm_remove_event:
                    remove_event_category(selected_event)
                    set_config_notice("事件类别已删除。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()
                else:
                    st.warning("请先勾选确认项。")

        st.divider()
        new_event_name = st.text_input("新增事件类别", key="new_event_name")
        new_event_keywords = st.text_input(
            "新增事件关键词",
            placeholder="多个关键词用逗号分隔",
            key="new_event_keywords",
        )
        new_event_related = st.multiselect(
            "新增事件影响板块",
            options=list(sectors_config),
            key="new_event_related",
        )
        if st.button("新增事件类别", key="add_event_category"):
            keywords_to_add = parse_keyword_input(new_event_keywords)
            if not keywords_to_add:
                st.warning("新增事件类别至少需要 1 个关键词。")
            else:
                try:
                    add_event_category(new_event_name, keywords_to_add, new_event_related)
                    set_config_notice("事件类别已保存。请点击“增量刷新”或“全量刷新”更新新闻。")
                    st.rerun()
                except Exception as exc:
                    st.warning(str(exc))

        st.divider()
        confirm_reset_events = st.checkbox("确认恢复默认外部事件配置", key="confirm_reset_events")
        if st.button("恢复默认事件配置", key="reset_event_config"):
            if confirm_reset_events:
                reset_events_config()
                set_config_notice("外部事件配置已恢复默认。请点击“增量刷新”或“全量刷新”更新新闻。")
                st.rerun()
            else:
                st.warning("请先勾选确认项。")


def main() -> None:
    inject_styles()

    if "cache_version" not in st.session_state:
        st.session_state.cache_version = 0
    if "last_added_count" not in st.session_state:
        st.session_state.last_added_count = 0
    if "last_sector_warnings" not in st.session_state:
        st.session_state.last_sector_warnings = {}
    if "last_external_warnings" not in st.session_state:
        st.session_state.last_external_warnings = {}

    sectors_config, sectors_config_error = try_load_sectors_config()
    events_config, events_config_error = try_load_events_config()
    external_events = events_config["external_events"]
    event_to_sectors = events_config["event_to_sectors"]
    llm_verifier, llm_notice = get_llm_verifier()

    for sector in sectors_config:
        st.session_state.setdefault(
            f"sector_selected::{sector}",
            sector in DEFAULT_SELECTED_SECTORS,
        )

    st.sidebar.header("筛选")
    show_config_notice()
    if sectors_config_error:
        st.sidebar.warning(sectors_config_error)
    if events_config_error:
        st.sidebar.warning(events_config_error)

    sector_query = st.sidebar.text_input("板块搜索", placeholder="输入板块或关键词")
    select_col, clear_col = st.sidebar.columns(2)
    if select_col.button("全选"):
        for sector in sectors_config:
            st.session_state[f"sector_selected::{sector}"] = True
    if clear_col.button("清空"):
        for sector in sectors_config:
            st.session_state[f"sector_selected::{sector}"] = False

    render_sector_selector(sectors_config, sector_query)

    selected_sectors = [
        sector
        for sector in sectors_config
        if st.session_state.get(f"sector_selected::{sector}", False)
    ]
    display_scope = st.sidebar.radio(
        "新闻类型",
        options=["全部", "板块新闻", "外部事件"],
    )
    show_sector_news = display_scope in ("全部", "板块新闻")
    show_external_events = display_scope in ("全部", "外部事件")
    time_range = st.sidebar.radio("时间范围", options=TIME_RANGE_OPTIONS)
    if llm_notice:
        st.sidebar.warning(llm_notice)
    keyword = st.sidebar.text_input("关键词搜索", placeholder="标题、命中关键词、来源媒体")
    max_items = st.sidebar.slider("每个板块最多展示条数", min_value=5, max_value=100, value=30)

    st.sidebar.divider()
    render_sector_config_manager(sectors_config)
    render_event_config_manager(external_events, event_to_sectors, sectors_config)

    st.sidebar.divider()
    st.sidebar.subheader("数据更新")
    if st.sidebar.button("增量刷新"):
        if selected_sectors or show_external_events:
            with st.spinner("正在增量抓取新新闻，并合并到本地缓存..."):
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
                save_cache(merged_cache)
                st.session_state.last_added_count = added_count
                st.session_state.last_sector_warnings = warnings_by_sector
                st.session_state.last_external_warnings = warnings_by_event
                st.session_state.cache_version += 1
                st.success(f"增量刷新完成，新增 {added_count} 条新闻。")
        else:
            st.sidebar.warning("请至少选择一个板块，或切换为显示外部事件后再刷新。")

    if st.sidebar.button("全量刷新"):
        if selected_sectors or show_external_events:
            with st.spinner("正在全量抓取所选板块，并重建本地缓存..."):
                fetched_cache, warnings_by_sector, warnings_by_event = fetch_refresh_cache(
                    selected_sectors,
                    sectors_config,
                    external_events,
                    event_to_sectors,
                    llm_verifier=llm_verifier,
                )
                if fetched_cache.empty and (warnings_by_sector or warnings_by_event):
                    st.session_state.last_added_count = 0
                    st.session_state.last_sector_warnings = warnings_by_sector
                    st.session_state.last_external_warnings = warnings_by_event
                    st.warning("全量刷新未获取到有效数据，已保留原本地缓存。")
                else:
                    rebuilt_cache = deduplicate_cache(fetched_cache)
                    rebuilt_cache, cache_refilter_warnings = refilter_external_event_cache(
                        rebuilt_cache,
                        external_events,
                        sectors_config=sectors_config,
                        llm_verifier=llm_verifier,
                    )
                    if cache_refilter_warnings:
                        warnings_by_event.setdefault("缓存重过滤", []).extend(
                            cache_refilter_warnings
                        )
                    save_cache(rebuilt_cache)
                    st.session_state.last_added_count = len(rebuilt_cache)
                    st.session_state.last_sector_warnings = warnings_by_sector
                    st.session_state.last_external_warnings = warnings_by_event
                    st.session_state.cache_version += 1
                    st.success(f"全量刷新完成，缓存已重建为 {len(rebuilt_cache)} 条新闻。")
        else:
            st.sidebar.warning("请至少选择一个板块，或切换为显示外部事件后再全量刷新。")

    confirm_clear_cache = st.sidebar.checkbox("确认要清空本地缓存")
    if st.sidebar.button("确认清空本地缓存"):
        if confirm_clear_cache:
            clear_cache()
            st.session_state.last_added_count = 0
            st.session_state.last_sector_warnings = {}
            st.session_state.last_external_warnings = {}
            st.session_state.cache_version += 1
            st.warning("本地缓存已清空。")
        else:
            st.sidebar.warning("请先勾选确认项，避免误清空缓存。")

    if show_sector_news and not selected_sectors and not show_external_events:
        st.info("请至少选择一个板块。")
        return

    (
        raw_cache_display_df,
        sector_deduped_df,
        external_deduped_df,
        metadata,
        cache_error,
        cache_refilter_warnings,
    ) = load_display_data(
        st.session_state.cache_version,
        sectors_config,
        external_events,
        llm_verifier_cache_key(llm_verifier),
        llm_verifier,
    )
    if cache_error:
        st.warning(cache_error + "。页面将显示空缓存，可点击全量刷新重建。")
    if cache_refilter_warnings:
        st.warning("；".join(cache_refilter_warnings))

    sector_cache_display_df = filter_news_by_time(sector_deduped_df, time_range)
    external_display_df = filter_news_by_time(external_deduped_df, time_range)
    selected_cache_display_df = sector_cache_display_df[
        sector_cache_display_df["sector"].isin(selected_sectors)
    ].reset_index(drop=True)

    source_frames: list[pd.DataFrame] = []
    if show_sector_news:
        source_frames.append(selected_cache_display_df)
    if show_external_events:
        source_frames.append(external_display_df)
    source_df = (
        pd.concat(source_frames, ignore_index=True)
        if source_frames
        else pd.DataFrame()
    )
    available_sources = collect_sources(source_df)
    selected_sources = st.sidebar.multiselect("来源媒体", options=available_sources)

    results: dict[str, SectorResult] = {}
    warnings_by_sector = st.session_state.get("last_sector_warnings", {})
    for sector in selected_sectors if show_sector_news else []:
        sector_df = selected_cache_display_df[
            selected_cache_display_df["sector"] == sector
        ].reset_index(drop=True)
        results[sector] = SectorResult(
            data=sector_df,
            warnings=tuple(warnings_by_sector.get(sector, [])),
        )

    displayed_total = 0
    for sector in selected_sectors if show_sector_news else []:
        filtered_df = filter_news(results[sector].data, keyword, selected_sources)
        displayed_total += min(len(filtered_df), max_items)
    if show_external_events:
        filtered_external_df = filter_news(external_display_df, keyword, selected_sources)
        displayed_total += min(len(filtered_external_df), max_items)

    latest_cache_at = format_utc8_time(metadata["latest_fetched_at"])
    render_dashboard_header(
        selected_count=len(selected_sectors),
        displayed_total=displayed_total,
        cache_total=int(metadata["total"]),
        latest_cache_at=latest_cache_at,
        added_count=int(st.session_state.last_added_count),
    )

    st.caption("本地缓存已启用")
    st.caption(f"缓存更新时间：{latest_cache_at}")
    if raw_cache_display_df.empty:
        st.info("本地缓存暂无新闻。点击左侧“增量刷新”进行首次抓取，或点击“全量刷新”重建缓存。")
    elif sector_cache_display_df.empty and external_display_df.empty:
        st.info("当前时间范围下暂无新闻。")

    if show_external_events:
        show_external_events_section(
            external_display_df,
            st.session_state.get("last_external_warnings", {}),
            keyword,
            selected_sources,
            max_items,
        )

    if show_sector_news:
        if not selected_sectors:
            st.info("未选择板块，当前只显示外部事件。")
        for sector in selected_sectors:
            show_sector_section(
                sector,
                results[sector],
                keyword,
                selected_sources,
                max_items,
            )


if __name__ == "__main__":
    main()
