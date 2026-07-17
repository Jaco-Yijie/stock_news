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
from analysis import (
    IMPACT_LABELS,
    LOW_CONFIDENCE_REASON,
    SENTIMENT_LABELS,
    analyze_display_frame,
    group_similar_news,
    parse_publish_times,
    select_hot_news,
    sentiment_counts,
    sort_news_by_importance,
)
from analysis_rules import CATEGORY_LABELS
from digest import build_digest_records, generate_sector_digest
from fetcher import SectorResult, deduplicate_news
from llm_provider import load_llm_verifier_from_env
from news_store import (
    cache_fingerprint,
    cache_metadata,
    cache_to_display,
    clear_cache,
    read_cache,
    refilter_external_event_cache,
    save_cache,
)
from refresh import run_full_refresh, run_incremental_refresh
from sectors import SECTORS as DEFAULT_SECTORS
from stock_search import (
    StockProfile,
    fetch_stock_profile,
    format_market_cap,
    load_stock_list,
    match_sectors_for_stock,
    search_stocks,
)
from time_utils import now_utc8_naive


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
        /* ============================================================
           设计语言：中文财经日报
           纸面灰白底 + 暖墨字 + 市场红（品牌色即"利好红"）+ 盘面绿
           报头与栏目标题用宋体，数字一律等宽对齐（红涨绿跌）
           ============================================================ */
        :root {
            --paper: #f6f5f2;
            --surface: #ffffff;
            --surface-soft: #f1efeb;
            --line: #e5e1d8;
            --line-strong: #cfc9be;
            --ink: #23201c;
            --ink-soft: #4c463f;
            --muted: #6e675e;
            --faint: #9b938a;
            --red: #b5372e;
            --red-deep: #96291f;
            --red-soft: #f7ebe8;
            --red-line: #e4c4bf;
            --green: #2e7d4f;
            --green-soft: #eaf2ec;
            --green-line: #c4dbcc;
            --amber: #96660f;
            --amber-soft: #f7efdb;
            --amber-line: #e6d3a8;
            --serif: "Songti SC", "Noto Serif SC", "STSong", "SimSun", serif;
        }

        .stApp {
            background: var(--paper);
            color: var(--ink);
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
            background: #fbfaf7;
            border-right: 1px solid var(--line);
            min-width: 292px !important;
            max-width: 320px !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
            letter-spacing: 0;
            font-size: 14px;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-family: var(--serif);
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.5rem;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: var(--red-soft);
            border: 1px solid var(--red-line);
            color: var(--red-deep);
            font-weight: 600;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: var(--red-deep);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: var(--surface);
            border-color: var(--line);
            border-radius: 3px;
            box-shadow: none;
        }

        [data-baseweb="select"] > div:hover,
        [data-baseweb="input"] > div:hover {
            border-color: var(--line-strong);
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--red);
            border-color: var(--red);
        }

        [data-testid="stCheckbox"] {
            min-height: 1.75rem;
        }

        [data-testid="stCheckbox"] label {
            gap: 0.35rem;
            color: var(--ink);
            font-size: 0.9rem;
        }

        [data-testid="stCheckbox"] div[role="checkbox"] {
            border-color: var(--line-strong) !important;
        }

        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
            background-color: var(--red) !important;
            border-color: var(--red) !important;
        }

        div.stButton > button {
            width: 100%;
            border: 1px solid var(--line-strong);
            border-radius: 3px;
            background: var(--surface);
            color: var(--ink);
            box-shadow: none;
            font-weight: 600;
            transition: border-color 120ms ease, color 120ms ease;
        }

        div.stButton > button:hover {
            border-color: var(--red);
            background: var(--surface);
            color: var(--red-deep);
        }

        div.stButton > button:focus-visible,
        a:focus-visible {
            outline: 2px solid var(--red);
            outline-offset: 2px;
        }

        /* ---- 报头 ---- */
        .masthead {
            border-bottom: 4px double var(--line-strong);
            margin-bottom: 1.1rem;
        }

        .masthead-top {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            padding: 0.2rem 0 0.7rem;
        }

        .masthead-title {
            font-family: var(--serif);
            font-size: 30px;
            font-weight: 900;
            letter-spacing: 0.06em;
            line-height: 1.2;
            color: var(--ink);
        }

        .masthead-date {
            font-size: 13px;
            color: var(--muted);
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }

        .ticker-strip {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            border-top: 1px solid var(--line);
            padding: 0.6rem 0 0.7rem;
        }

        .ticker-strip .stat {
            display: inline-flex;
            align-items: baseline;
            gap: 0.45rem;
            padding: 0 1.15rem;
            border-right: 1px solid var(--line);
        }

        .ticker-strip .stat:first-child {
            padding-left: 0;
        }

        .ticker-strip .stat:last-child {
            border-right: 0;
        }

        .stat-label {
            font-size: 0.78rem;
            color: var(--muted);
            white-space: nowrap;
        }

        .stat-num {
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }

        .stat-num.up { color: var(--red); }
        .stat-num.down { color: var(--green); }

        /* ---- 个股信息卡片的指标格 ---- */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0;
            border-top: 1px solid var(--line);
        }

        .metric-card {
            padding: 0.6rem 0.95rem;
            border-right: 1px solid var(--line);
        }

        .metric-card:last-child {
            border-right: 0;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            margin-bottom: 0.15rem;
            white-space: nowrap;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        /* ---- 板块区块 ---- */
        .sector-section {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 3px;
            margin: 0.8rem 0 1rem;
            overflow: hidden;
        }

        .sector-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.78rem 0.95rem;
            border-bottom: 1px solid var(--line);
        }

        .sector-name {
            font-family: var(--serif);
            color: var(--ink);
            font-size: 19px;
            font-weight: 700;
            overflow-wrap: anywhere;
            border-left: 3px solid var(--red);
            padding-left: 10px;
            line-height: 1.3;
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
            border: 1px solid var(--line);
            background: var(--paper);
            color: var(--muted);
            padding: 0.18rem 0.52rem;
            font-size: 0.75rem;
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }

        .pill.warn {
            border-color: var(--amber-line);
            background: var(--amber-soft);
            color: var(--amber);
        }

        .section-warning {
            color: var(--amber);
            background: var(--amber-soft);
            border-bottom: 1px solid var(--amber-line);
            padding: 0.62rem 0.95rem;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }

        .news-list {
            padding: 0.3rem 0;
        }

        .news-card {
            border-bottom: 1px solid var(--line);
            background: var(--surface);
            padding: 14px 16px;
            transition: background-color 120ms ease;
        }

        .news-card:last-child {
            border-bottom: 0;
        }

        .news-card:hover {
            background: #fbfaf7;
        }

        .news-title {
            margin: 0 0 6px;
            color: var(--ink);
            font-size: 17px;
            line-height: 1.5;
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
            font-variant-numeric: tabular-nums;
        }

        .keyword-tag {
            border-radius: 3px;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            color: var(--ink-soft);
            padding: 1px 6px;
            font-size: 12px;
            font-weight: 500;
        }

        .news-link {
            margin-left: auto;
            color: var(--red) !important;
            text-decoration: none !important;
            font-weight: 600;
            white-space: nowrap;
        }

        .news-link:hover {
            color: var(--red-deep) !important;
            text-decoration: underline !important;
        }

        .empty-state {
            padding: 0.9rem 0.95rem;
            color: var(--muted);
            background: var(--surface);
        }

        /* ---- 标签：红=利好，绿=利空，琥珀=高影响 ---- */
        .tag {
            display: inline-flex;
            align-items: center;
            border-radius: 3px;
            padding: 1px 8px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.6;
            border: 1px solid transparent;
            white-space: nowrap;
        }

        .tag-pos { background: var(--red-soft); color: var(--red-deep); border-color: var(--red-line); }
        .tag-neg { background: var(--green-soft); color: var(--green); border-color: var(--green-line); }
        .tag-neu { background: var(--surface-soft); color: var(--muted); border-color: var(--line); }
        .tag-impact-high { background: var(--amber-soft); color: var(--amber); border-color: var(--amber-line); }
        .tag-impact-medium { background: var(--surface-soft); color: var(--muted); border-color: var(--line); }
        .tag-impact-low { background: transparent; color: var(--faint); border-color: var(--line); }
        .tag-hot { background: var(--red); color: #ffffff; border-color: var(--red); }
        .tag-cat-policy { background: transparent; color: var(--ink-soft); border-color: var(--line-strong); }
        .tag-cat-stock_move { background: transparent; color: var(--faint); border-color: var(--line); }
        .tag-low-conf { background: var(--amber-soft); color: var(--amber); border-color: var(--amber-line); }

        details.related-reports {
            margin-top: 6px;
            font-size: 12px;
            color: var(--muted);
        }

        details.related-reports summary {
            cursor: pointer;
            color: var(--red);
            font-weight: 500;
        }

        .related-item {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            padding: 4px 0 0 12px;
            color: var(--muted);
        }

        .related-item a {
            color: var(--ink-soft) !important;
            text-decoration: none !important;
        }

        .related-item a:hover {
            color: var(--red) !important;
            text-decoration: underline !important;
        }

        .news-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 0 0 6px;
        }

        .news-reason {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 12px;
            flex-wrap: wrap;
            color: var(--ink-soft);
            font-size: 13px;
            margin-top: 7px;
        }

        .section-title {
            font-family: var(--serif);
            font-size: 20px;
            font-weight: 700;
            color: var(--ink);
            margin: 1.2rem 0 0.15rem;
        }

        .hot-card {
            border: 1px solid var(--line);
            border-left: 3px solid var(--red);
            border-radius: 3px;
            background: var(--surface);
            padding: 14px 16px;
            margin-bottom: 0.6rem;
        }

        .hot-title {
            font-size: 18px;
            font-weight: 650;
            color: var(--ink);
            margin: 6px 0 4px;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .hot-sectors {
            color: var(--ink-soft);
            font-size: 13px;
            margin-top: 3px;
        }

        /* ---- 侧边栏结构：眉题分组 + 轻量操作 + 扁平分组列表 ---- */
        [data-testid="stSidebar"] .side-label {
            border-top: 1px solid var(--line);
            margin-top: 1.15rem;
            padding-top: 0.95rem;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.16em;
            color: var(--faint);
        }

        [data-testid="stSidebar"] .sector-status {
            font-size: 12px;
            color: var(--muted);
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        [data-testid="stSidebar"] .st-key-sector_toolbar div.stButton > button {
            width: 100%;
            min-height: 1.7rem;
            padding: 0.2rem 0.3rem;
            border: none;
            background: transparent;
            color: var(--red);
            font-size: 12.5px;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .st-key-sector_toolbar div.stButton > button:hover {
            color: var(--red-deep);
            background: transparent;
            text-decoration: underline;
        }

        [data-testid="stSidebar"] .st-key-sector_groups [data-testid="stExpander"] {
            border: 0;
            border-bottom: 1px solid var(--line);
            border-radius: 0;
            background: transparent;
        }

        [data-testid="stSidebar"] .st-key-sector_groups [data-testid="stExpander"] details {
            border: none;
            background: transparent;
        }

        [data-testid="stSidebar"] .st-key-sector_groups [data-testid="stExpander"] summary {
            padding: 0.5rem 0.15rem;
        }

        [data-testid="stSidebar"] .st-key-sector_groups [data-testid="stExpander"] summary:hover {
            color: var(--red-deep);
        }

        [data-testid="stSidebar"] .st-key-sector_groups [data-testid="stCheckbox"] {
            min-height: 1.55rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.25rem 0.9rem;
            flex-wrap: wrap;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: var(--faint);
            font-size: 12.5px;
        }

        @media (prefers-reduced-motion: reduce) {
            .news-card,
            div.stButton > button {
                transition: none;
            }
        }

        @media (max-width: 900px) {
            .masthead-top {
                flex-direction: column;
                gap: 0.2rem;
            }

            .ticker-strip .stat {
                padding: 0 0.8rem;
            }

            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .metric-card:nth-child(2n) {
                border-right: 0;
            }

            .metric-card:nth-child(-n+2) {
                border-bottom: 1px solid var(--line);
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

            .masthead-title {
                font-size: 24px;
            }

            .ticker-strip {
                gap: 0.3rem 0;
            }

            .stat-label {
                font-size: 11px;
            }

            .stat-num {
                font-size: 1rem;
            }

            .hot-title {
                font-size: 16px;
            }

            .news-reason {
                font-size: 12px;
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
                font-size: 15px;
                line-height: 1.5;
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


@st.cache_data(ttl=60, show_spinner=False)
def get_cache_fingerprint(cache_version: int) -> str:
    """每分钟最多探测一次数据指纹；后台抓取写入新数据后，
    指纹变化会自动触发下方展示缓存重新加载。"""
    return cache_fingerprint()


@st.cache_data(ttl=3600, show_spinner=False)
def load_display_data(
    cache_version: int,
    fingerprint: str,
    sectors_config: dict[str, Any],
    external_events: dict[str, Any],
    llm_cache_key: str,
    _llm_verifier: LLMVerifier | None,
):
    """读取缓存并完成重过滤、格式转换、去重。

    整条流水线只在缓存数据、缓存版本或配置变化时重算，
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
    # 利好/利空判断为派生数据，随展示流水线一起缓存，不写入新闻缓存
    sector_display_df = sector_display_df.copy()
    sector_display_df["analysis"] = analyze_display_frame(sector_display_df)
    external_display_df = external_display_df.copy()
    external_display_df["analysis"] = analyze_display_frame(external_display_df)
    return (
        display_df,
        sector_display_df,
        external_display_df,
        cache_metadata(cache_df),
        cache_result.error,
        refilter_warnings,
    )


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_stock_list() -> list[tuple[str, str]]:
    """A 股代码-名称列表，每天最多拉取一次；失败时退回内置列表。"""
    return load_stock_list()


@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_profile(code: str, name: str) -> StockProfile:
    """个股基本信息与实时行情，5 分钟内复用。"""
    return fetch_stock_profile(code, name)


def parse_keyword_input(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").replace("，", ",").split(",")
        if item.strip()
    ]


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
    now = pd.Timestamp(now_utc8_naive())
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
    # 状态 + 全选/清空 放在同一行：状态是信息，操作降为文字钮。
    # 先处理按钮点击再计算已选数，保证状态行反映本次操作的结果。
    with st.sidebar.container(key="sector_toolbar"):
        status_col, select_col, clear_col = st.columns(
            [1.6, 0.7, 0.7], vertical_alignment="center"
        )
        if select_col.button("全选", key="select_all_sectors"):
            for sector in sectors_config:
                st.session_state[f"sector_selected::{sector}"] = True
        if clear_col.button("清空", key="clear_all_sectors"):
            for sector in sectors_config:
                st.session_state[f"sector_selected::{sector}"] = False

        selected_count = sum(
            1
            for sector in sectors_config
            if st.session_state.get(f"sector_selected::{sector}", False)
        )
        if normalized_query:
            status_text = f"匹配 {len(visible_sectors)} · 已选 {selected_count}"
        else:
            status_text = f"已选 {selected_count} / {len(sectors_config)} 个板块"
        status_col.markdown(
            f'<div class="sector-status">{escape(status_text)}</div>',
            unsafe_allow_html=True,
        )

    visible_set = set(visible_sectors)
    has_visible_group = False
    with st.sidebar.container(key="sector_groups"):
        for group_name, sectors in grouped_sectors(sectors_config):
            visible_group_sectors = [
                sector for sector in sectors if sector in visible_set
            ]
            if not visible_group_sectors:
                continue

            has_visible_group = True
            selected_in_group = sum(
                1
                for sector in sectors
                if st.session_state.get(f"sector_selected::{sector}", False)
            )
            # 有选中的分组加粗，展开状态和计数一起把"选了什么"变得可扫读
            label = f"{group_name} · {selected_in_group}/{len(sectors)}"
            if selected_in_group:
                label = f"**{label}**"
            expanded = bool(normalized_query) or selected_in_group > 0
            with st.expander(label, expanded=expanded):
                for sector in visible_group_sectors:
                    st.checkbox(sector, key=f"sector_selected::{sector}")

    if not has_visible_group:
        st.sidebar.caption("没有匹配的板块，换个词试试。")

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


LOW_CONFIDENCE_TAG = '<span class="tag tag-low-conf">低置信</span>'
DIVERGENT_TAG = '<span class="tag tag-neu">板块分化</span>'


def sentiment_tag_html(sentiment: str, prefix: str = "") -> str:
    css_class = {"positive": "tag-pos", "negative": "tag-neg"}.get(sentiment, "tag-neu")
    label = SENTIMENT_LABELS.get(sentiment, "中性")
    if prefix:
        label = f"{prefix}·{label}"
    return f'<span class="tag {css_class}">{escape(label)}</span>'


def sentiment_tags_html(
    sentiment: str,
    low_confidence: bool = False,
    divergent: bool = False,
    prefix: str = "",
) -> str:
    html = sentiment_tag_html(sentiment, prefix)
    if sentiment == "neutral":
        if divergent:
            html += DIVERGENT_TAG
        elif low_confidence:
            html += LOW_CONFIDENCE_TAG
    return html


def impact_tag_html(impact_level: str) -> str:
    label = IMPACT_LABELS.get(impact_level, "低影响")
    title_attr = ""
    if impact_level == "high":
        title_attr = ' title="依据政策、宏观事件或产业信号词判定"'
    return f'<span class="tag tag-impact-{escape(impact_level)}"{title_attr}>{label}</span>'


def _row_analysis(row: pd.Series) -> dict[str, Any]:
    analysis = row.get("analysis")
    if isinstance(analysis, dict):
        return analysis
    return {
        "sentiment": "neutral",
        "impact_level": "low",
        "reason": LOW_CONFIDENCE_REASON,
        "confidence": 0,
        "divergent": False,
        "sector_assessments": {},
    }


def _is_low_confidence(analysis: dict[str, Any]) -> bool:
    return int(analysis.get("confidence", 0)) < 45


def news_tags_html(row: pd.Series, hot_links: set[str] | None = None) -> str:
    analysis = _row_analysis(row)
    low_confidence = _is_low_confidence(analysis)
    tags: list[str] = []

    if str(row.get("news_type", "")) == "external_event":
        assessments = analysis.get("sector_assessments") or {}
        shown = 0
        for sector, assessment in assessments.items():
            tags.append(
                sentiment_tag_html(assessment.get("sentiment", "neutral"), prefix=sector)
            )
            shown += 1
            if shown >= 3:
                break
        if not assessments or analysis.get("divergent"):
            tags.insert(
                0,
                sentiment_tags_html(
                    analysis.get("sentiment", "neutral"),
                    low_confidence=low_confidence,
                    divergent=bool(analysis.get("divergent")),
                ),
            )
    else:
        sector_name = str(row.get("sector", "") or "").strip()
        assessment = (analysis.get("sector_assessments") or {}).get(sector_name)
        sentiment = (
            assessment.get("sentiment") if assessment else analysis.get("sentiment", "neutral")
        )
        tags.append(
            sentiment_tags_html(
                sentiment,
                low_confidence=low_confidence,
                divergent=bool(analysis.get("divergent")),
            )
        )

    tags.append(impact_tag_html(analysis.get("impact_level", "low")))
    category = str(analysis.get("category", "general"))
    if category in ("policy", "stock_move"):
        tags.append(
            f'<span class="tag tag-cat-{category}">{CATEGORY_LABELS.get(category, "")}</span>'
        )
    link = str(row.get("原文链接", ""))
    if hot_links and link in hot_links:
        tags.append('<span class="tag tag-hot">今日重点</span>')
    return f'<div class="news-tags">{"".join(tags)}</div>'


def _row_reason(row: pd.Series) -> str:
    analysis = _row_analysis(row)
    if str(row.get("news_type", "")) != "external_event":
        sector_name = str(row.get("sector", "") or "").strip()
        assessment = (analysis.get("sector_assessments") or {}).get(sector_name)
        if assessment:
            return str(assessment.get("reason", ""))
    return str(analysis.get("reason", ""))


def related_reports_html(related_rows: list[pd.Series]) -> str:
    if not related_rows:
        return ""
    items = []
    for row in related_rows[:8]:
        title = escape(str(row.get("标题", "")))
        source = escape(str(row.get("来源媒体", "")))
        publish_time = escape(str(row.get("发布时间", "")))
        link = escape(str(row.get("原文链接", "")), quote=True)
        items.append(
            f'<div class="related-item">'
            f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>'
            f"<span>{source} · {publish_time}</span></div>"
        )
    return (
        f'<details class="related-reports">'
        f"<summary>另有 {len(related_rows)} 条相关报道</summary>"
        + "".join(items)
        + "</details>"
    )


def render_news_item(
    row: pd.Series,
    hot_links: set[str] | None = None,
    related_rows: list[pd.Series] | None = None,
) -> str:
    title = escape(str(row.get("标题", "")))
    source = escape(str(row.get("来源媒体", "")))
    publish_time = escape(str(row.get("发布时间", "")))
    keyword = escape(str(row.get("匹配关键词", "")))
    link = escape(str(row.get("原文链接", "")), quote=True)
    reason = escape(_row_reason(row))

    return f"""
        <article class="news-card">
            <div class="news-title">{title}</div>
            {news_tags_html(row, hot_links)}
            <div class="news-meta">
                <span>{source}</span>
                <span>{publish_time}</span>
                <span class="keyword-tag">{keyword}</span>
            </div>
            <div class="news-reason">
                <span>判断：{reason}</span>
                <a class="news-link" href="{link}" target="_blank" rel="noopener noreferrer">查看原文</a>
            </div>
            {related_reports_html(related_rows or [])}
        </article>
        """


def render_external_event_item(
    row: pd.Series,
    hot_links: set[str] | None = None,
    related_rows: list[pd.Series] | None = None,
) -> str:
    title = escape(str(row.get("标题", "")))
    source = escape(str(row.get("来源媒体", "")))
    publish_time = escape(str(row.get("发布时间", "")))
    keyword = escape(str(row.get("匹配关键词", "")))
    event_category = escape(str(row.get("event_category", "")))
    related_sectors = escape(str(row.get("related_sectors", "") or "未映射"))
    link = escape(str(row.get("原文链接", "")), quote=True)
    reason = escape(_row_reason(row))

    return f"""
        <article class="news-card">
            <div class="news-title">{title}</div>
            {news_tags_html(row, hot_links)}
            <div class="news-meta">
                <span>{source}</span>
                <span>{publish_time}</span>
                <span class="keyword-tag">{event_category}</span>
                <span class="keyword-tag">{keyword}</span>
                <span>可能影响板块：{related_sectors}</span>
            </div>
            <div class="news-reason">
                <span>判断：{reason}</span>
                <a class="news-link" href="{link}" target="_blank" rel="noopener noreferrer">查看原文</a>
            </div>
            {related_reports_html(related_rows or [])}
        </article>
        """


def render_hot_news_item(row: pd.Series) -> str:
    analysis = _row_analysis(row)
    title = escape(str(row.get("标题", "")))
    source = escape(str(row.get("来源媒体", "")))
    publish_time = escape(str(row.get("发布时间", "")))
    link = escape(str(row.get("原文链接", "")), quote=True)
    reason = escape(str(analysis.get("reason", "")))
    sectors = list((analysis.get("sector_assessments") or {}))
    sectors_text = escape("、".join(sectors) if sectors else "暂无明确映射")

    tags = (
        sentiment_tags_html(
            analysis.get("sentiment", "neutral"),
            low_confidence=_is_low_confidence(analysis),
            divergent=bool(analysis.get("divergent")),
        )
        + impact_tag_html(analysis.get("impact_level", "low"))
    )

    return f"""
        <article class="hot-card">
            <div class="news-tags">{tags}</div>
            <div class="hot-title">{title}</div>
            <div class="news-meta">
                <span>来源：{source}</span>
                <span>{publish_time}</span>
            </div>
            <div class="hot-sectors">影响板块：{sectors_text}</div>
            <div class="news-reason">
                <span>判断：{reason}</span>
                <a class="news-link" href="{link}" target="_blank" rel="noopener noreferrer">查看原文</a>
            </div>
        </article>
        """


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_sector_digest(
    digest_date: str,
    sector: str,
    records: list[dict[str, str]],
    llm_cache_key: str,
    _llm_verifier: LLMVerifier | None,
) -> str:
    return generate_sector_digest(_llm_verifier, sector, records)


MAX_DIGEST_SECTORS = 5


def render_daily_digest_section(
    llm_verifier: LLMVerifier | None,
    selected_sectors: list[str],
    sector_deduped_df: pd.DataFrame,
) -> None:
    with st.expander("今日板块日报（LLM 生成）", expanded=False):
        if llm_verifier is None:
            st.info(
                "未配置 LLM，无法生成日报。在环境变量或 Secrets 中配置 "
                "LLM_VERIFY_PROVIDER 及豆包 / DeepSeek 密钥后可用。"
            )
            return
        if not selected_sectors:
            st.info("请先选择板块。")
            return

        if st.button("生成 / 更新今日日报", key="generate_digest"):
            st.session_state.digest_requested = True
        if not st.session_state.get("digest_requested"):
            st.caption(
                f"点击按钮按板块归纳今天的新闻（最多 {MAX_DIGEST_SECTORS} 个板块，"
                "结果缓存 6 小时）。"
            )
            return

        today = now_utc8_naive().date()
        publish_times = parse_publish_times(sector_deduped_df)
        today_df = sector_deduped_df[publish_times.dt.date.eq(today).fillna(False)]

        for sector in selected_sectors[:MAX_DIGEST_SECTORS]:
            sector_df = today_df[today_df["sector"] == sector]
            if sector_df.empty:
                st.markdown(f"**{sector}**：今日暂无相关新闻。")
                continue
            records = build_digest_records(sort_news_by_importance(sector_df))
            try:
                with st.spinner(f"正在归纳「{sector}」..."):
                    digest_text = cached_sector_digest(
                        str(today),
                        sector,
                        records,
                        llm_verifier_cache_key(llm_verifier),
                        llm_verifier,
                    )
            except Exception as exc:
                st.warning(f"「{sector}」日报生成失败：{exc}")
                continue
            if digest_text:
                st.markdown(f"**{sector}**：{digest_text}")
            else:
                st.markdown(f"**{sector}**：LLM 未返回有效内容。")
        if len(selected_sectors) > MAX_DIGEST_SECTORS:
            st.caption(f"已选板块较多，仅生成前 {MAX_DIGEST_SECTORS} 个板块的日报以控制成本。")


def show_hot_news_section(hot_df: pd.DataFrame, used_fallback: bool) -> None:
    st.markdown('<div class="section-title">今日重点新闻</div>', unsafe_allow_html=True)
    st.caption("范围：全部板块与相关外部事件，与关注板块相关的新闻优先。")
    if hot_df is None or hot_df.empty:
        st.info("今日暂无足够重要的新闻。可以尝试刷新数据或选择更多板块。")
        return
    if used_fallback:
        st.caption("今日重点新闻不足，已补充最近 24 小时的重要新闻。")
    cards = "".join(render_hot_news_item(row) for _, row in hot_df.iterrows())
    st.markdown(cards, unsafe_allow_html=True)


def sentiment_pills_html(analyses) -> str:
    counts = sentiment_counts(analyses)
    return (
        f'<span class="pill">利好 {counts["positive"]}</span>'
        f'<span class="pill">中性 {counts["neutral"]}</span>'
        f'<span class="pill">利空 {counts["negative"]}</span>'
    )


EMPTY_FILTER_HINT = (
    '<div class="empty-state">当前筛选条件下暂无相关新闻。'
    "可以尝试扩大时间范围或选择更多板块。</div>"
)


def show_sector_section(
    sector: str,
    result,
    keyword: str,
    selected_sources: list[str],
    max_items: int,
    hot_links: set[str] | None = None,
) -> None:
    filtered_df = filter_news(result.data, keyword, selected_sources)
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
        body_html = warning_html + EMPTY_FILTER_HINT
    else:
        sorted_df = sort_news_by_importance(filtered_df)
        clusters = group_similar_news(sorted_df.head(max_items * 3))
        news_html = "".join(
            render_news_item(
                sorted_df.iloc[primary],
                hot_links,
                [sorted_df.iloc[position] for position in related],
            )
            for primary, related in clusters[:max_items]
        )
        body_html = warning_html + f'<div class="news-list">{news_html}</div>'

    sentiment_pills = (
        sentiment_pills_html(filtered_df["analysis"])
        if "analysis" in filtered_df.columns
        else ""
    )
    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">{escape(sector)}相关新闻</div>
                <div class="sector-meta">
                    <span class="pill">共 {len(filtered_df)} 条</span>
                    {sentiment_pills}
                </div>
            </div>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _format_quote_change(quote: dict[str, str]) -> tuple[str, str]:
    """返回（涨跌幅文本, tag 样式类）。A 股习惯：红涨绿跌。"""
    raw = quote.get("涨幅", "")
    try:
        change = float(raw)
    except (TypeError, ValueError):
        return "", "tag-neu"
    if change > 0:
        return f"+{change:.2f}%", "tag-pos"
    if change < 0:
        return f"{change:.2f}%", "tag-neg"
    return "0.00%", "tag-neu"


def render_stock_profile_card(profile: StockProfile, related_sectors: list[str]) -> None:
    quote_tags = ""
    latest = str(profile.quote.get("最新", "") or "").strip()
    if latest:
        change_text, change_class = _format_quote_change(profile.quote)
        quote_tags = f'<span class="pill">最新价 {escape(latest)}</span>'
        if change_text:
            quote_tags += f'<span class="tag {change_class}">{escape(change_text)}</span>'

    metrics: list[tuple[str, str]] = [
        ("所属行业", profile.industry or "-"),
        ("总市值", format_market_cap(profile.info.get("总市值", ""))),
        ("流通市值", format_market_cap(profile.info.get("流通市值", ""))),
        ("上市时间", profile.info.get("上市时间", "-") or "-"),
    ]
    metric_html = "".join(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(str(value))}</div>
        </div>
        """
        for label, value in metrics
    )

    if related_sectors:
        sector_pills = "".join(
            f'<span class="keyword-tag">{escape(sector)}</span>'
            for sector in related_sectors
        )
        sectors_html = (
            f'<div class="news-meta" style="margin-top:0.6rem;">关联板块：{sector_pills}</div>'
        )
    else:
        sectors_html = (
            '<div class="news-meta" style="margin-top:0.6rem;">'
            "未能识别关联板块，下方仅展示直接提及该股票的新闻。</div>"
        )

    warning_html = ""
    if profile.error:
        warning_html = f'<div class="section-warning">{escape(profile.error)}</div>'

    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">{escape(profile.name)}（{escape(profile.code)}）</div>
                <div class="sector-meta">{quote_tags}</div>
            </div>
            {warning_html}
            <div class="metric-grid">{metric_html}</div>
            {sectors_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_stock_mention_news(
    stock_name: str,
    sector_time_df: pd.DataFrame,
    external_time_df: pd.DataFrame,
    selected_sources: list[str],
    max_items: int,
    hot_links: set[str] | None,
) -> None:
    frames = [df for df in (sector_time_df, external_time_df) if not df.empty]
    combined_df = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    mention_df = filter_news(combined_df, stock_name, selected_sources)
    if not mention_df.empty and "原文链接" in mention_df.columns:
        mention_df = mention_df.drop_duplicates(subset=["原文链接"]).reset_index(drop=True)

    if mention_df.empty:
        body_html = (
            '<div class="empty-state">缓存新闻中暂未直接提及该股票，'
            "可查看下方关联板块新闻。</div>"
        )
    else:
        sorted_df = sort_news_by_importance(mention_df)
        news_html = "".join(
            render_news_item(sorted_df.iloc[position], hot_links)
            for position in range(min(len(sorted_df), max_items))
        )
        body_html = f'<div class="news-list">{news_html}</div>'

    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">{escape(stock_name)}直接相关新闻</div>
                <div class="sector-meta">
                    <span class="pill">共 {len(mention_df)} 条</span>
                </div>
            </div>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_stock_search_section(
    stock_query: str,
    sector_time_df: pd.DataFrame,
    external_time_df: pd.DataFrame,
    sectors_config: dict[str, Any],
    selected_sources: list[str],
    max_items: int,
    hot_links: set[str] | None,
) -> None:
    st.markdown('<div class="section-title">股票搜索</div>', unsafe_allow_html=True)
    matches = search_stocks(stock_query, cached_stock_list())
    if not matches:
        st.info(f"未找到与“{stock_query}”匹配的股票，请检查代码或名称。")
        return

    match = matches[0]
    if len(matches) > 1:
        match = st.selectbox(
            "匹配到多只股票，请选择",
            options=matches,
            format_func=lambda item: item.label,
        )

    with st.spinner(f"正在获取 {match.name} 的信息…"):
        profile = cached_stock_profile(match.code, match.name)
    related_sectors = match_sectors_for_stock(
        match.name, profile.industry, sectors_config
    )

    render_stock_profile_card(profile, related_sectors)
    show_stock_mention_news(
        match.name,
        sector_time_df,
        external_time_df,
        selected_sources,
        max_items,
        hot_links,
    )

    for sector in related_sectors:
        sector_df = sector_time_df[
            sector_time_df["sector"] == sector
        ].reset_index(drop=True) if "sector" in sector_time_df.columns else sector_time_df.iloc[0:0]
        show_sector_section(
            sector,
            SectorResult(data=sector_df),
            "",
            selected_sources,
            max_items,
            hot_links,
        )


def show_external_events_section(
    external_df: pd.DataFrame,
    warnings_by_event: dict[str, list[str]],
    keyword: str,
    selected_sources: list[str],
    max_items: int,
    hot_links: set[str] | None = None,
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
        body_html = warning_html + EMPTY_FILTER_HINT
    else:
        sorted_df = sort_news_by_importance(filtered_df)
        clusters = group_similar_news(sorted_df.head(max_items * 3))
        news_html = "".join(
            render_external_event_item(
                sorted_df.iloc[primary],
                hot_links,
                [sorted_df.iloc[position] for position in related],
            )
            for primary, related in clusters[:max_items]
        )
        body_html = warning_html + f'<div class="news-list">{news_html}</div>'

    sentiment_pills = (
        sentiment_pills_html(filtered_df["analysis"])
        if "analysis" in filtered_df.columns
        else ""
    )
    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <div class="sector-name">外部宏观风险事件</div>
                <div class="sector-meta">
                    <span class="pill">共 {len(filtered_df)} 条</span>
                    {sentiment_pills}
                </div>
            </div>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header(
    today_total: int,
    hot_count: int,
    today_counts: dict[str, int],
) -> None:
    now = now_utc8_naive()
    weekday_names = "一二三四五六日"
    date_text = (
        f"{now.year}年{now.month}月{now.day}日 · 星期{weekday_names[now.weekday()]}"
    )
    stats = [
        ("今日新闻", today_total, ""),
        ("今日热点", hot_count, ""),
        ("利好", today_counts.get("positive", 0), "up"),
        ("中性", today_counts.get("neutral", 0), ""),
        ("利空", today_counts.get("negative", 0), "down"),
    ]
    stats_html = "".join(
        f"""
        <span class="stat">
            <span class="stat-label">{escape(label)}</span>
            <span class="stat-num {css_class}">{value}</span>
        </span>
        """
        for label, value, css_class in stats
    )
    st.markdown(
        f"""
        <section class="masthead">
            <div class="masthead-top">
                <div class="masthead-title">A股板块新闻</div>
                <div class="masthead-date">{escape(date_text)}</div>
            </div>
            <div class="ticker-strip">{stats_html}</div>
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
    with st.container():
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
    with st.container():
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


FILTER_STATE_DEFAULTS: dict[str, Any] = {
    "stock_query": "",
    "sector_query": "",
    "display_scope": "全部",
    "time_range": "全部",
    "keyword_search": "",
    "max_items": 30,
    "source_filter": [],
}


def reset_all_filters(sectors_config: dict[str, Any]) -> None:
    for sector in sectors_config:
        st.session_state[f"sector_selected::{sector}"] = sector in DEFAULT_SELECTED_SECTORS
    for key, value in FILTER_STATE_DEFAULTS.items():
        st.session_state[key] = value


def _record_refresh_status(added_count: int) -> None:
    st.session_state.last_added_count = added_count
    st.session_state.last_refresh_at = now_utc8_naive().strftime("%H:%M")


def render_data_refresh_controls(
    selected_sectors: list[str],
    sectors_config: dict[str, Any],
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
    llm_verifier: LLMVerifier | None,
    show_external_events: bool,
) -> None:
    if st.button("重新加载缓存"):
        st.session_state.cache_version += 1
    st.caption("页面每分钟自动检测后台数据更新；点击可立即重新加载。")

    last_refresh_at = st.session_state.get("last_refresh_at", "")
    if last_refresh_at:
        st.caption(
            f"已成功更新 {int(st.session_state.get('last_added_count', 0))} 条新闻 ｜ "
            f"最近刷新时间：{last_refresh_at}"
        )

    if st.button("增量刷新"):
        if selected_sectors or show_external_events:
            with st.spinner("正在增量抓取新新闻，并合并到缓存..."):
                outcome = run_incremental_refresh(
                    selected_sectors,
                    sectors_config,
                    external_events,
                    event_to_sectors,
                    llm_verifier=llm_verifier,
                )
                if outcome.read_error:
                    st.error(
                        f"读取现有缓存失败，为避免误删数据本次未刷新：{outcome.read_error}"
                    )
                    st.stop()
                try:
                    save_cache(outcome.cache)
                except Exception as exc:
                    st.error(f"缓存保存失败，本次抓取结果未持久化：{exc}")
                else:
                    _record_refresh_status(outcome.added_count)
                    st.session_state.last_sector_warnings = outcome.sector_warnings
                    st.session_state.last_external_warnings = outcome.event_warnings
                    st.session_state.cache_version += 1
                    st.success(f"已成功更新 {outcome.added_count} 条新闻。")
        else:
            st.warning("请至少选择一个板块，或切换为显示外部事件后再刷新。")

    if st.button("全量刷新"):
        if selected_sectors or show_external_events:
            with st.spinner("正在全量抓取所选板块，并重建缓存..."):
                outcome = run_full_refresh(
                    selected_sectors,
                    sectors_config,
                    external_events,
                    event_to_sectors,
                    llm_verifier=llm_verifier,
                )
                if outcome.fetch_failed:
                    st.session_state.last_added_count = 0
                    st.session_state.last_sector_warnings = outcome.sector_warnings
                    st.session_state.last_external_warnings = outcome.event_warnings
                    st.warning("全量刷新未获取到有效数据，已保留原缓存。")
                else:
                    try:
                        save_cache(outcome.cache)
                    except Exception as exc:
                        st.error(f"缓存保存失败，本次抓取结果未持久化：{exc}")
                    else:
                        _record_refresh_status(outcome.added_count)
                        st.session_state.last_sector_warnings = outcome.sector_warnings
                        st.session_state.last_external_warnings = outcome.event_warnings
                        st.session_state.cache_version += 1
                        st.success(f"全量刷新完成，缓存已重建为 {outcome.added_count} 条新闻。")
        else:
            st.warning("请至少选择一个板块，或切换为显示外部事件后再全量刷新。")

    st.divider()
    confirm_clear_cache = st.checkbox("确认要清空缓存", key="confirm_clear_cache")
    if st.button("清空缓存"):
        if confirm_clear_cache:
            try:
                clear_cache()
            except Exception as exc:
                st.error(f"清空缓存失败：{exc}")
            else:
                st.session_state.last_added_count = 0
                st.session_state.last_sector_warnings = {}
                st.session_state.last_external_warnings = {}
                st.session_state.cache_version += 1
                st.warning("缓存已清空。")
        else:
            st.warning("请先勾选确认项，避免误清空缓存。")


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
    for state_key, state_value in FILTER_STATE_DEFAULTS.items():
        st.session_state.setdefault(state_key, state_value)

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

    st.sidebar.markdown('<div class="side-label">个股</div>', unsafe_allow_html=True)
    stock_query = st.sidebar.text_input(
        "股票搜索",
        placeholder="股票代码或名称，如 600519 / 贵州茅台",
        key="stock_query",
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="side-label">关注板块</div>', unsafe_allow_html=True)
    sector_query = st.sidebar.text_input(
        "板块搜索",
        placeholder="搜索板块或关键词",
        key="sector_query",
        label_visibility="collapsed",
    )
    render_sector_selector(sectors_config, sector_query)

    selected_sectors = [
        sector
        for sector in sectors_config
        if st.session_state.get(f"sector_selected::{sector}", False)
    ]
    if selected_sectors:
        st.sidebar.caption("已选：" + "、".join(selected_sectors))

    st.sidebar.markdown('<div class="side-label">新闻列表</div>', unsafe_allow_html=True)
    display_scope = st.sidebar.radio(
        "新闻类型",
        options=["全部", "板块新闻", "外部事件"],
        key="display_scope",
        horizontal=True,
    )
    show_sector_news = display_scope in ("全部", "板块新闻")
    show_external_events = display_scope in ("全部", "外部事件")
    time_range = st.sidebar.radio(
        "时间范围", options=TIME_RANGE_OPTIONS, key="time_range", horizontal=True
    )
    keyword = st.sidebar.text_input(
        "关键词搜索", placeholder="标题、命中关键词、来源媒体", key="keyword_search"
    )
    # 占位容器：来源媒体选项依赖缓存数据，数据加载后再填充，
    # 但视觉位置保持在关键词搜索之后、展示条数之前
    source_filter_container = st.sidebar.container()
    max_items = st.sidebar.slider(
        "每个板块最多展示条数", min_value=5, max_value=100, key="max_items"
    )

    st.sidebar.divider()
    # 重置放底部：先看筛选项，逃生舱最后；用回调避免在组件实例化后改状态
    st.sidebar.button(
        "重置全部筛选", on_click=reset_all_filters, args=(sectors_config,)
    )
    # 系统状态属于底部的高级设置区，不打断上方的筛选流
    if llm_notice:
        st.sidebar.caption(llm_notice)
    with st.sidebar.expander("数据管理与高级设置", expanded=False):
        keyword_tab, event_tab, data_tab = st.tabs(["关键词管理", "外部事件", "数据刷新"])
        with keyword_tab:
            render_sector_config_manager(sectors_config)
        with event_tab:
            render_event_config_manager(external_events, event_to_sectors, sectors_config)
        with data_tab:
            render_data_refresh_controls(
                selected_sectors,
                sectors_config,
                external_events,
                event_to_sectors,
                llm_verifier,
                show_external_events,
            )

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
        get_cache_fingerprint(st.session_state.cache_version),
        sectors_config,
        external_events,
        llm_verifier_cache_key(llm_verifier),
        llm_verifier,
    )
    if cache_error:
        st.warning(cache_error + "。页面将显示空缓存，可点击全量刷新重建。")
    # 常规的"已按新规则过滤旧缓存"属于后台正常行为，不展示；只提示真正的失败
    refilter_failures = [
        warning for warning in cache_refilter_warnings if "失败" in warning
    ]
    if refilter_failures:
        st.warning("；".join(refilter_failures))

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
    st.session_state.source_filter = [
        source
        for source in st.session_state.get("source_filter", [])
        if source in available_sources
    ]
    with source_filter_container:
        selected_sources = st.multiselect(
            "来源媒体",
            options=available_sources,
            key="source_filter",
            placeholder="选择来源媒体",
        )
        st.caption(f"缓存共 {int(metadata['total'])} 条新闻")

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

    # 今日热点：从去重后的全部新闻中选取，不受时间范围筛选影响
    hot_pool_frames: list[pd.DataFrame] = []
    if show_sector_news:
        hot_pool_frames.append(sector_deduped_df)
    if show_external_events:
        hot_pool_frames.append(external_deduped_df)
    hot_pool_df = (
        pd.concat(hot_pool_frames, ignore_index=True)
        if hot_pool_frames
        else pd.DataFrame()
    )
    hot_df, hot_used_fallback = select_hot_news(hot_pool_df, selected_sectors)
    hot_links = (
        set(hot_df["原文链接"].astype(str)) if not hot_df.empty else set()
    )

    # 今日新闻统计：当前关注范围内、发布日期为今天（UTC+8）的新闻
    today_scope_frames: list[pd.DataFrame] = []
    if show_sector_news and selected_sectors:
        today_scope_frames.append(
            sector_deduped_df[sector_deduped_df["sector"].isin(selected_sectors)]
        )
    if show_external_events:
        today_scope_frames.append(external_deduped_df)
    today_counts = {"positive": 0, "neutral": 0, "negative": 0}
    today_total = 0
    if today_scope_frames:
        today_scope_df = pd.concat(today_scope_frames, ignore_index=True)
        publish_times = parse_publish_times(today_scope_df)
        today_mask = publish_times.dt.date.eq(now_utc8_naive().date()).fillna(False)
        today_df = today_scope_df[today_mask]
        today_total = len(today_df)
        if "analysis" in today_df.columns:
            today_counts = sentiment_counts(today_df["analysis"])

    render_dashboard_header(
        today_total=today_total,
        hot_count=len(hot_df),
        today_counts=today_counts,
    )
    st.caption(
        "利好 / 利空 / 影响等级由规则与新闻文本自动生成，仅供信息参考，不构成投资建议。"
    )

    if raw_cache_display_df.empty:
        st.info("缓存暂无新闻。后台任务会定时抓取，也可以在“数据管理与高级设置”中手动刷新。")
    elif sector_cache_display_df.empty and external_display_df.empty:
        st.info("当前时间范围下暂无新闻。可以尝试扩大时间范围或选择更多板块。")

    if stock_query.strip():
        show_stock_search_section(
            stock_query.strip(),
            sector_cache_display_df,
            external_display_df,
            sectors_config,
            selected_sources,
            max_items,
            hot_links,
        )

    show_hot_news_section(hot_df, hot_used_fallback)

    if show_sector_news:
        render_daily_digest_section(llm_verifier, selected_sectors, sector_deduped_df)

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
                hot_links,
            )

    if show_external_events:
        show_external_events_section(
            external_display_df,
            st.session_state.get("last_external_warnings", {}),
            keyword,
            selected_sources,
            max_items,
            hot_links,
        )


if __name__ == "__main__":
    main()
