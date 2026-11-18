import hashlib
import hmac
import os
from collections import Counter
from html import escape
from typing import Any

import pandas as pd
import requests
import streamlit as st

from classifier import LLMVerifier, extract_rule_keywords
from config_store import (
    ConfigUnavailableError,
    add_event_category,
    add_event_keyword,
    add_keyword,
    add_sector,
    default_events_config,
    default_sectors_config,
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
    LOW_CONFIDENCE_THRESHOLD,
    SENTIMENT_LABELS,
    analyze_display_frame,
    group_similar_news,
    parse_publish_times,
    select_hot_news,
    sentiment_counts,
    sort_news_by_importance,
    target_sectors_for_row,
)
from analysis_rules import CATEGORY_LABELS
from digest import build_digest_records, generate_sector_digest
from fetcher import SectorResult, deduplicate_news
from llm_provider import load_llm_verifier_from_env
from news_links import safe_news_url
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
from time_utils import now_utc8_naive


st.set_page_config(
    page_title="板块新闻", page_icon="📰", layout="wide", initial_sidebar_state="collapsed"
)
DEFAULT_SELECTED_SECTORS = ("半导体芯片",)
TIME_RANGE_OPTIONS = ("全部", "今天", "近 3 天", "本周")


def get_llm_verifier() -> tuple[LLMVerifier | None, str | None]:
    return load_llm_verifier_from_env()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #fff;
            --soft: #f2f4f7;
            --line: #e1ded6;
            --ink: #23201c;
            --muted: #625e57;
            --brand: #1d3a5f;
            --brand-soft: #e8eef5;
            --positive: #a12634;
            --positive-soft: #fbedef;
            --neutral: #59616b;
            --neutral-soft: #eef0f2;
            --negative: #216246;
            --negative-soft: #e9f3ed;
            --serif: "Songti SC", STSong, SimSun, Georgia, serif;
            --sans: system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC",
                    "Microsoft YaHei", sans-serif;
        }
        .stApp { font-family: var(--sans); }
        .block-container { max-width: 1480px; padding: 3.75rem 2rem 3rem; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
        .stApp :is(button, a, summary, input, textarea, [tabindex]):focus-visible {
            outline: 2px solid var(--brand);
            outline-offset: 2px;
        }
        .stApp [data-baseweb="select"] > div:hover,
        .stApp [data-baseweb="input"] > div:hover { border-color: var(--brand); }
        .stApp [data-baseweb="select"] > div:focus-within,
        .stApp [data-baseweb="input"] > div:focus-within {
            box-shadow: 0 0 0 2px var(--brand);
        }
        .stApp :is(summary, [role="tab"]):hover { color: var(--brand); }

        .masthead {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: .5rem 1rem;
            flex-wrap: wrap;
            padding-bottom: .6rem;
            margin-bottom: .8rem;
            border-bottom: 3px double var(--brand);
        }
        .masthead .masthead-title {
            margin: 0;
            padding: 0;
            color: var(--brand);
            font: 800 clamp(1.45rem, 2.5vw, 2rem)/1.25 var(--serif);
        }
        .masthead-date, .news-time, .market-totals strong {
            font-variant-numeric: tabular-nums;
        }
        .masthead-date { color: var(--muted); font-size: .84rem; }
        .st-key-news_navigation [role="tab"] { min-height: 2.75rem; }
        .st-key-news_navigation [role="tab"]:focus-visible { outline-offset: -3px; }
        .st-key-news_navigation [aria-selected="true"] { font-weight: 700; }

        .briefing {
            height: 100%;
            padding: clamp(1rem, 2.2vw, 1.9rem);
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 4px solid var(--brand);
            border-radius: .75rem;
        }
        .briefing .briefing-label, .market-totals h3 {
            margin: 0;
            padding: 0;
            color: var(--brand);
            font: 750 .95rem/1.4 var(--sans);
        }
        .briefing .briefing-headline {
            margin: .48rem 0 .75rem;
            padding: 0;
            color: var(--ink);
            font: 800 clamp(1.4rem, 2.65vw, 2.25rem)/1.35 var(--serif);
            overflow-wrap: anywhere;
        }
        .briefing ul { margin: 0; padding-left: 1.2rem; color: var(--muted); }
        .briefing li + li { margin-top: .3rem; }
        .briefing li { overflow-wrap: anywhere; font-size: .9rem; }

        .st-key-market_mood, .st-key-follow_panel, .st-key-heat_panel {
            background: var(--surface);
            border-radius: .75rem;
        }
        .st-key-overview_body { margin-top: 1rem; }
        .market-totals dl { display: flex; flex-wrap: wrap; gap: .5rem 1.6rem; margin: .85rem 0; }
        .market-totals dt { color: var(--muted); font-size: .82rem; }
        .market-totals dd { margin: 0; color: var(--brand); font-size: 1.6rem; line-height: 1.25; }
        .mood-bar { display: flex; height: .55rem; border-radius: .3rem; overflow: hidden; background: var(--neutral-soft); }
        .market-totals .mood-bar { margin-bottom: .65rem; }
        .mood-bar > span { flex: var(--count) 1 0; background: var(--mood); min-width: 0; }
        .mood-positive { --mood: var(--positive); --mood-soft: var(--positive-soft); }
        .mood-neutral { --mood: var(--neutral); --mood-soft: var(--neutral-soft); }
        .mood-negative { --mood: var(--negative); --mood-soft: var(--negative-soft); }
        .st-key-overview_positive { --mood: var(--positive); --mood-soft: var(--positive-soft); }
        .st-key-overview_neutral { --mood: var(--neutral); --mood-soft: var(--neutral-soft); }
        .st-key-overview_negative { --mood: var(--negative); --mood-soft: var(--negative-soft); }
        .st-key-market_mood div.stButton button {
            color: var(--mood);
            background: var(--mood-soft);
            border-color: transparent;
            font-variant-numeric: tabular-nums;
            padding-inline: .25rem;
            min-height: 2.5rem;
        }
        .st-key-market_mood div.stButton button:is(:hover, [kind="primary"]) {
            border-color: var(--mood);
        }
        .st-key-heat_panel div.stButton button { font-variant-numeric: tabular-nums; }
        .st-key-heat_panel div.stButton button[kind="secondary"]:hover {
            background: var(--brand-soft);
        }
        .st-key-overview_flow [data-testid="stCaptionContainer"] { color: var(--muted); }

        .sector-heading {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: .5rem 1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: .7rem;
        }
        .sector-heading .sector-name { font: 750 1.22rem/1.4 var(--serif); color: var(--brand); }
        .sector-meta, .news-tags, .news-sectors, .news-meta, .news-top {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: .35rem .5rem;
        }
        .pill, .tag {
            border-radius: .3rem;
            padding: .13rem .45rem;
            color: var(--muted);
            background: var(--soft);
            font-size: .76rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .tag-sentiment { color: var(--mood); background: var(--mood-soft); font-weight: 700; }
        .tag-emphasis { color: var(--brand); background: var(--brand-soft); font-weight: 700; }
        .section-warning, .empty-state {
            border-radius: .45rem;
            background: var(--surface);
            padding: .8rem 1rem;
            color: var(--muted);
            overflow-wrap: anywhere;
        }
        .section-warning { border-inline-start: 3px solid var(--brand); margin-bottom: .65rem; }
        .news-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-inline-start: 4px solid var(--mood);
            border-radius: .6rem;
            padding: .8rem 1rem;
            margin-bottom: .65rem;
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .news-top { justify-content: space-between; }
        .news-time { color: var(--brand); font-size: .85rem; font-weight: 750; }
        .news-card .news-title {
            color: var(--ink);
            font: 700 1.13rem/1.45 var(--sans);
            padding: 0;
            margin: .4rem 0 .55rem;
        }
        .news-meta, .news-sectors { color: var(--muted); font-size: .8rem; }
        .news-meta { justify-content: space-between; margin-top: .55rem; }
        .news-reason {
            margin-top: .55rem;
            padding: .48rem .65rem;
            border-radius: .35rem;
            background: var(--soft);
            font-size: .85rem;
            line-height: 1.55;
        }
        .news-reason strong { margin-inline-end: .45rem; color: var(--brand); }
        .confidence { display: inline-flex; align-items: center; gap: .3rem; color: var(--muted); }
        .confidence-bars { display: inline-flex; gap: 2px; }
        .confidence-bars i { width: .36rem; height: .34rem; border-radius: 1px; background: #dce1e7; }
        .confidence-bars .filled { background: var(--brand); }
        .news-card a, .related-reports summary { color: var(--brand); font-weight: 650; }
        .news-card a:hover, .related-reports summary:hover { text-decoration: underline; }
        .related-reports { margin-top: .5rem; font-size: .8rem; color: var(--muted); }
        .related-reports summary { cursor: pointer; }
        .related-item { padding: .3rem .65rem 0; }

        @media (max-width: 860px) {
            :is(.st-key-overview_top, .st-key-overview_body, .st-key-main_filters) [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            :is(.st-key-overview_top, .st-key-overview_body, .st-key-main_filters) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100%;
                width: 100%;
            }
            .st-key-market_mood [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
            .st-key-market_mood [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 0;
                width: auto;
                min-width: 0 !important;
            }
        }
        @media (max-width: 600px) {
            .block-container { padding: 3.5rem .85rem 2rem; }
            .news-card { padding: .7rem .75rem; }
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


@st.cache_data(ttl=60, max_entries=16, show_spinner=False)
def get_cache_fingerprint(cache_version: int) -> str:
    """每分钟最多探测一次数据指纹；后台抓取写入新数据后，
    指纹变化会自动触发下方展示缓存重新加载。"""
    return cache_fingerprint()


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_display_data(
    cache_version: int,
    fingerprint: str,
    sectors_config: dict[str, Any],
    external_events: dict[str, Any],
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


def _row_sentiment(row: pd.Series) -> str:
    analysis = _row_analysis(row)
    sector = str(row.get("sector", "") or "").strip()
    assessment = (analysis.get("sector_assessments") or {}).get(sector)
    sentiment = (
        assessment.get("sentiment")
        if assessment and row.get("news_type") != "external_event"
        else analysis.get("sentiment")
    )
    return sentiment if sentiment in SENTIMENT_LABELS else "neutral"


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
        title = news_link_html(row.get("原文链接"), str(row.get("标题", "")), css_class="")
        source = escape(str(row.get("来源媒体", "")))
        publish_time = escape(str(row.get("发布时间", "")))
        items.append(
            f'<div class="related-item">'
            f"{title}"
            f"<span>{source} · {publish_time}</span></div>"
        )
    return (
        f'<details class="related-reports">'
        f"<summary>另有 {len(related_rows)} 条相关报道</summary>"
        + "".join(items)
        + "</details>"
    )


def news_link_html(value: object, label: str = "查看原文", css_class: str = "news-link") -> str:
    url = safe_news_url(value)
    text = escape(label)
    if not url:
        return text if not css_class else ""
    return (
        f'<a class="{css_class}" href="{escape(url, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">{text}</a>'
    )


def confidence_html(confidence: int) -> str:
    segments = 1 + (confidence >= LOW_CONFIDENCE_THRESHOLD) + (confidence >= 70)
    bars = "".join(f'<i class="{ "filled" if index < segments else "" }"></i>' for index in range(3))
    return (
        f'<span class="confidence" role="img" aria-label="置信度 {confidence} / 100" '
        f'title="置信度 {confidence} / 100">'
        f'<span aria-hidden="true">置信度</span>'
        f'<span class="confidence-bars" aria-hidden="true">{bars}</span></span>'
    )


def render_news_item(
    row: pd.Series,
    hot_links: set[str] | None = None,
    related_rows: list[pd.Series] | None = None,
) -> str:
    analysis = _row_analysis(row)
    sentiment = _row_sentiment(row)
    impact = analysis.get("impact_level", "low")
    impact_class = " tag-emphasis" if impact == "high" else ""
    tags = [
        f'<span class="tag tag-sentiment">{escape(SENTIMENT_LABELS[sentiment])}</span>',
        f'<span class="tag{impact_class}">{escape(IMPACT_LABELS.get(impact, "低影响"))}</span>',
    ]
    if analysis.get("divergent"):
        tags.append('<span class="tag">板块分化</span>')
    category = str(analysis.get("category", "general"))
    if category in ("policy", "stock_move"):
        tags.append(f'<span class="tag">{escape(CATEGORY_LABELS[category])}</span>')
    if hot_links and row.get("原文链接") in hot_links:
        tags.append('<span class="tag tag-emphasis">今日重点</span>')

    sectors = target_sectors_for_row(row)
    external = row.get("news_type") == "external_event"
    assessments = analysis.get("sector_assessments") or {}
    sector_tags = []
    for sector in sectors:
        direction = (assessments.get(sector) or {}).get("sentiment")
        if external and direction in SENTIMENT_LABELS:
            sector_tags.append(
                f'<span class="tag tag-sentiment mood-{direction}">'
                f'{escape(sector)} · {escape(SENTIMENT_LABELS[direction])}</span>'
            )
        else:
            sector_tags.append(f'<span class="tag">{escape(sector)}</span>')
    if external and not sector_tags:
        sector_tags.append('<span class="tag">未映射</span>')
    sectors_html = (
        f'<div class="news-sectors"><span>影响板块</span>{"".join(sector_tags)}</div>'
        if sector_tags else ""
    )
    metadata = " · ".join(
        str(value) for value in (
            row.get("来源媒体"),
            row.get("event_category") if external else None,
            row.get("匹配关键词"),
        ) if value
    )

    return f"""
        <article class="news-card mood-{sentiment}">
            <div class="news-top">
                <time class="news-time">{escape(str(row.get("发布时间") or "时间未知"))}</time>
                <div class="news-tags">{"".join(tags)}</div>
            </div>
            <h4 class="news-title">{escape(str(row.get("标题", "")))}</h4>
            {sectors_html}
            <div class="news-reason">
                <strong>判断</strong>{escape(_row_reason(row))}
            </div>
            <div class="news-meta">
                <span>{escape(metadata)}</span>
                {confidence_html(int(analysis.get("confidence", 0)))}
                {news_link_html(row.get("原文链接"))}
            </div>
            {related_reports_html(related_rows or [])}
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


def describe_llm_error(exc: Exception) -> str:
    """把 LLM 调用异常翻译成能直接照着处理的说明。"""
    if isinstance(exc, requests.Timeout):
        return (
            "调用超时。模型生成整段日报通常需要 10-60 秒，"
            "可在 Secrets 里调大 LLM_COMPLETE_TIMEOUT（当前默认 90 秒）。"
        )
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", "")
        if status == 401:
            return "鉴权失败（401）：API Key 无效或已过期。"
        if status == 404:
            return "接口不存在（404）：检查 *_ENDPOINT 与 *_MODEL 是否匹配。"
        if status == 429:
            return "触发限流（429）：账户额度或并发不足。"
        return f"模型接口请求失败（HTTP {status}）。"
    if isinstance(exc, requests.RequestException):
        return f"网络请求失败（{type(exc).__name__}）。"
    return f"生成失败（{type(exc).__name__}）。"


def render_daily_digest_section(
    llm_verifier: LLMVerifier | None,
    selected_sectors: list[str],
    sector_deduped_df: pd.DataFrame,
    *,
    can_generate: bool,
) -> None:
    with st.expander("今日板块日报（LLM 生成）", expanded=False):
        if llm_verifier is None:
            st.info(
                "未配置 LLM，无法生成日报。在环境变量或 Secrets 中配置 "
                "LLM_VERIFY_PROVIDER 及豆包 / DeepSeek 密钥后可用。"
            )
            return
        if not can_generate:
            st.caption("管理员登录后可生成日报。")
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
                st.warning(f"「{sector}」日报生成失败：{describe_llm_error(exc)}")
                continue
            if digest_text:
                st.markdown(f"**{sector}**：{digest_text}")
            else:
                st.markdown(f"**{sector}**：LLM 未返回有效内容。")
        if len(selected_sectors) > MAX_DIGEST_SECTORS:
            st.caption(f"已选板块较多，仅生成前 {MAX_DIGEST_SECTORS} 个板块的日报以控制成本。")


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
        body_html = warning_html + news_html

    sentiment_pills = (
        sentiment_pills_html(filtered_df["analysis"])
        if "analysis" in filtered_df.columns
        else ""
    )
    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <h3 class="sector-name">{escape(sector)}</h3>
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
            render_news_item(
                sorted_df.iloc[primary],
                hot_links,
                [sorted_df.iloc[position] for position in related],
            )
            for primary, related in clusters[:max_items]
        )
        body_html = warning_html + news_html

    sentiment_pills = (
        sentiment_pills_html(filtered_df["analysis"])
        if "analysis" in filtered_df.columns
        else ""
    )
    st.markdown(
        f"""
        <section class="sector-section">
            <div class="sector-heading">
                <h3 class="sector-name">外部宏观风险事件</h3>
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


def render_dashboard_header() -> None:
    now = now_utc8_naive()
    weekday = "一二三四五六日"[now.weekday()]
    date_text = f"{now:%Y.%m.%d} · 星期{weekday} · 北京时间"
    st.markdown(
        f"""
        <header class="masthead">
            <h1 class="masthead-title">板块新闻</h1>
            <time class="masthead-date">{escape(date_text)}</time>
        </header>
        """,
        unsafe_allow_html=True,
    )


def market_today(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news
    return news[parse_publish_times(news).dt.date.eq(now_utc8_naive().date())]


def filter_overview_sectors(news: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    if news.empty or not sectors:
        return news
    wanted = set(sectors)
    matches = [
        bool(wanted.intersection(target_sectors_for_row(row)))
        for row in news.to_dict("records")
    ]
    return news.loc[matches]


def render_market_brief(today: pd.DataFrame, hot_today: pd.DataFrame) -> None:
    headline = "今天还没有新闻"
    points = []
    if not hot_today.empty:
        lead = hot_today.iloc[0]
        headline = str(lead.get("标题") or "今日重点")
        direction = SENTIMENT_LABELS[_row_sentiment(lead)]
        impact = IMPACT_LABELS.get(_row_analysis(lead).get("impact_level"), "低影响")
        points.append(f"头条判断 · {direction} / {impact}：{_row_reason(lead)}")
        points.extend(
            f'{SENTIMENT_LABELS[_row_sentiment(row)]} · {row["标题"]}'
            for _, row in hot_today.iloc[1:3].iterrows()
        )
    elif not today.empty:
        headline = "今日已有新闻，暂无重点"
    else:
        points.append("后台会定时更新新闻，也可以在下方查看已有报道。")
    bullets = "".join(f"<li>{escape(point)}</li>" for point in points)
    st.markdown(
        f"""
        <section class="briefing">
            <h2 class="briefing-label">今日速览</h2>
            <p class="briefing-headline">{escape(headline)}</p>
            <ul>{bullets}</ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def toggle_overview_filter(key: str, value: str) -> None:
    st.session_state[key] = "" if st.session_state.get(key) == value else value


def render_market_mood(today: pd.DataFrame, hot_count: int) -> None:
    counts = sentiment_counts(today["analysis"]) if not today.empty else dict.fromkeys(SENTIMENT_LABELS, 0)
    segments = "".join(
        f'<span class="mood-{sentiment}" style="--count:{counts[sentiment]}"></span>'
        for sentiment in SENTIMENT_LABELS if counts[sentiment]
    )
    st.markdown(
        f"""
        <section class="market-totals">
            <h3>情绪分布</h3>
            <dl><div><dt>今日新闻</dt><dd><strong>{len(today)}</strong></dd></div>
                <div><dt>今日重点</dt><dd><strong>{hot_count}</strong></dd></div></dl>
            <div class="mood-bar" aria-hidden="true">{segments}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    for column, (sentiment, label) in zip(st.columns(3), SENTIMENT_LABELS.items()):
        column.button(
            f"{label} {counts[sentiment]}",
            key=f"overview_{sentiment}",
            width="stretch",
            type="primary" if st.session_state.overview_sentiment == sentiment else "secondary",
            on_click=toggle_overview_filter,
            args=("overview_sentiment", sentiment),
        )


def render_sector_heat(today: pd.DataFrame) -> None:
    st.subheader("板块热度榜")
    counts = Counter(
        sector
        for row in today.to_dict("records")
        for sector in set(target_sectors_for_row(row))
    )
    if not counts:
        st.caption("今天还没有板块热度数据。")
        return
    st.caption("按今日新闻条数排序")
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for rank, (sector, count) in enumerate(ranked[:6], 1):
        st.button(
            f"{rank:02d}　{sector} · {count} 条",
            key=f"overview_heat::{sector}",
            width="stretch",
            type="primary" if st.session_state.overview_sector == sector else "secondary",
            on_click=toggle_overview_filter,
            args=("overview_sector", sector),
        )


def render_overview_flow(
    pool: pd.DataFrame,
    sectors: list[str],
    time_range: str,
    keyword: str,
    sources: list[str],
    max_items: int,
    hot_links: set[str],
) -> None:
    st.subheader("重点新闻")
    active_sector = st.session_state.overview_sector
    active_sentiment = st.session_state.overview_sentiment
    if active_sector or active_sentiment:
        labels = [active_sector, SENTIMENT_LABELS.get(active_sentiment, "")]
        st.caption("概览筛选 · " + " · ".join(label for label in labels if label))
        st.button(
            "清除概览筛选", key="clear_overview_filters", on_click=clear_overview_filters
        )
    if pool.empty:
        st.info("缓存暂无此新闻类型的报道，后台任务会定时更新。")
        return

    display = filter_overview_sectors(pool, [active_sector] if active_sector else sectors)
    display = filter_news(filter_news_by_time(display, time_range), keyword, sources)
    if active_sentiment:
        display = display.loc[
            [_row_sentiment(row) == active_sentiment for _, row in display.iterrows()]
        ]
    if display.empty:
        st.info("当前筛选条件下暂无相关新闻。试试扩大时间范围或调整筛选条件。")
        return

    hot, used_fallback = select_hot_news(display, [active_sector] if active_sector else sectors)
    if used_fallback:
        st.caption("今日重点不足，已补充最近 24 小时的报道。")
    if not hot.empty:
        st.markdown(
            "".join(render_news_item(row, hot_links) for _, row in hot.iterrows()),
            unsafe_allow_html=True,
        )

    # 重点报道优先占据同组首位，其余报道继续沿用板块页的排序和事件折叠。
    hot_items = set(zip(hot["标题"], hot["原文链接"]))
    remaining = display.loc[
        [item not in hot_items for item in zip(display["标题"], display["原文链接"])]
    ]
    if remaining.empty or len(hot) >= max_items:
        return
    ordered = pd.concat(
        [hot, sort_news_by_importance(remaining).head(max_items * 3)], ignore_index=True
    )
    clusters = group_similar_news(ordered)
    additional = [(primary, related) for primary, related in clusters if primary >= len(hot)]
    if additional:
        st.subheader("更多新闻")
        st.markdown(
            "".join(
                render_news_item(
                    ordered.iloc[primary], hot_links,
                    [ordered.iloc[position] for position in related],
                )
                for primary, related in additional[:max_items - len(hot)]
            ),
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
    "display_scope": "全部",
    "time_range": "全部",
    "keyword_search": "",
    "max_items": 30,
    "source_filter": [],
    "overview_sector": "",
    "overview_sentiment": "",
}


def clear_overview_filters() -> None:
    st.session_state.overview_sector = ""
    st.session_state.overview_sentiment = ""


def clear_overview_sector() -> None:
    st.session_state.overview_sector = ""


def set_followed_sectors(sectors: list[str]) -> None:
    st.session_state.followed_sectors = sectors
    clear_overview_sector()


def reset_all_filters(sectors_config: dict[str, Any]) -> None:
    set_followed_sectors([sector for sector in DEFAULT_SELECTED_SECTORS if sector in sectors_config])
    for key, value in FILTER_STATE_DEFAULTS.items():
        st.session_state[key] = value


def render_sector_selector(sectors_config: dict[str, Any]) -> list[str]:
    options = list(sectors_config)
    current = st.session_state.setdefault(
        "followed_sectors", [sector for sector in DEFAULT_SELECTED_SECTORS if sector in sectors_config]
    )
    st.session_state.followed_sectors = [sector for sector in current if sector in sectors_config]
    selected = st.multiselect(
        "关注板块", options, key="followed_sectors", placeholder="搜索并选择板块",
        on_change=clear_overview_sector,
    )
    st.caption(f"已关注 {len(selected)} / {len(options)} 个板块")
    with st.container(horizontal=True, gap="small"):
        st.button(
            "全选", key="select_all_sectors", on_click=set_followed_sectors, args=(options,)
        )
        st.button("清空", key="clear_all_sectors", on_click=set_followed_sectors, args=([],))
    return selected


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
                    save_cache(outcome.cache, base_cache=outcome.base_cache)
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
                    preserve_unselected=True,
                )
                if outcome.read_error:
                    st.error("读取现有缓存失败，已保留原缓存。")
                elif outcome.fetch_failed:
                    st.session_state.last_added_count = 0
                    st.session_state.last_sector_warnings = outcome.sector_warnings
                    st.session_state.last_external_warnings = outcome.event_warnings
                    st.warning("全量刷新未获取到有效数据，已保留原缓存。")
                else:
                    try:
                        save_cache(outcome.cache, base_cache=outcome.base_cache)
                    except Exception as exc:
                        st.error(f"缓存保存失败，本次抓取结果未持久化：{exc}")
                    else:
                        _record_refresh_status(outcome.added_count)
                        st.session_state.last_sector_warnings = outcome.sector_warnings
                        st.session_state.last_external_warnings = outcome.event_warnings
                        st.session_state.cache_version += 1
                        st.success(f"全量刷新完成，获取 {outcome.added_count} 条新闻，"
                                   f"缓存共 {len(outcome.cache)} 条。")
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


def render_admin_login() -> bool:
    password = os.getenv("DASHBOARD_ADMIN_PASSWORD", "")
    if not password:
        try:
            password = str(st.secrets.get("DASHBOARD_ADMIN_PASSWORD", "") or "")
        except FileNotFoundError:
            pass
    if not password:
        st.caption("管理操作需要部署管理员配置口令。")
        return False

    expected = hashlib.sha256(password.encode()).digest()
    if hmac.compare_digest(st.session_state.get("admin_login_digest", b""), expected):
        if st.button("退出管理模式"):
            st.session_state.pop("admin_login_digest", None)
            st.session_state.pop("digest_requested", None)
            st.rerun()
        return True

    with st.form("admin_login", clear_on_submit=True):
        entered = st.text_input("管理员口令", type="password")
        if st.form_submit_button("登录"):
            if hmac.compare_digest(hashlib.sha256(entered.encode()).digest(), expected):
                st.session_state.admin_login_digest = expected
                st.rerun()
            st.error("管理员口令不正确。")
    return False


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

    # 远端配置暂时读不到时只临时显示默认配置，并暂停管理写操作：
    # 此时刷新会按默认规则误删自定义事件的缓存，编辑配置会覆盖远端已保存的内容。
    config_unavailable = False
    try:
        sectors_config, sectors_config_error = try_load_sectors_config(strict=True)
        events_config, events_config_error = try_load_events_config(strict=True)
    except ConfigUnavailableError as exc:
        config_unavailable = True
        sectors_config, sectors_config_error = default_sectors_config(), f"{exc}，已临时显示默认配置。"
        events_config, events_config_error = default_events_config(), None
    external_events = events_config["external_events"]
    event_to_sectors = events_config["event_to_sectors"]
    llm_verifier, llm_notice = get_llm_verifier()

    render_dashboard_header()
    with st.container(key="main_filters"):
        time_col, search_col = st.columns([1, 2])
        time_range = time_col.selectbox("时间范围", TIME_RANGE_OPTIONS, key="time_range")
        keyword = search_col.text_input(
            "关键词搜索", placeholder="标题、匹配关键词或来源…", key="keyword_search"
        )

    st.sidebar.header("筛选")
    show_config_notice()
    if sectors_config_error:
        st.sidebar.warning(sectors_config_error)
    if events_config_error:
        st.sidebar.warning(events_config_error)

    with st.sidebar.expander("更多筛选", expanded=False):
        display_scope = st.radio(
            "新闻类型",
            options=["全部", "板块新闻", "外部事件"],
            key="display_scope",
            horizontal=True,
        )
        source_filter_container = st.container()
        max_items = st.slider(
            "每组最多展示条数", min_value=5, max_value=100, key="max_items"
        )
    show_sector_news = display_scope in ("全部", "板块新闻")
    show_external_events = display_scope in ("全部", "外部事件")

    notices = st.container()
    tab_labels = ["市场概览"]
    if show_sector_news:
        tab_labels.append("板块新闻")
    if show_external_events:
        tab_labels.append("外部事件")
    tabs = dict(zip(tab_labels, st.tabs(tab_labels, key="news_navigation")))

    with tabs["市场概览"]:
        with st.container(key="overview_top"):
            brief_col, mood_col = st.columns([2, 1])
        with st.container(key="overview_body"):
            follow_col, flow_col, heat_col = st.columns([1, 2.7, 1.2])
        with follow_col.container(border=True, key="follow_panel"):
            selected_sectors = render_sector_selector(sectors_config)
        overview_footer = st.container()

    st.sidebar.button(
        "重置全部筛选", on_click=reset_all_filters, args=(sectors_config,)
    )
    # 系统状态属于底部的高级设置区，不打断上方的筛选流
    if llm_notice:
        st.sidebar.caption(llm_notice)
    with st.sidebar.expander("数据管理与高级设置", expanded=False):
        if st.button("重新加载缓存"):
            st.session_state.cache_version += 1
        st.caption("可立即读取后台更新的缓存。")
        admin = render_admin_login()
        if admin and config_unavailable:
            st.warning("远端配置暂时读取失败，管理操作已暂停。请稍后点击「重新加载缓存」重试。")
        elif admin:
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

    (
        _,
        sector_deduped_df,
        external_deduped_df,
        _,
        cache_error,
        cache_refilter_warnings,
    ) = load_display_data(
        st.session_state.cache_version,
        get_cache_fingerprint(st.session_state.cache_version),
        sectors_config,
        external_events,
    )
    with notices:
        if cache_error:
            st.warning(cache_error + "。页面将显示空缓存，管理员可在数据设置中全量刷新。")
        refilter_failures = [warning for warning in cache_refilter_warnings if "失败" in warning]
        if refilter_failures:
            st.warning("；".join(refilter_failures))

    sector_cache_display_df = filter_news_by_time(sector_deduped_df, time_range)
    external_display_df = filter_news_by_time(external_deduped_df, time_range)
    selected_cache_display_df = sector_cache_display_df[
        sector_cache_display_df["sector"].isin(selected_sectors)
    ].reset_index(drop=True)

    pool_frames = []
    if show_sector_news:
        pool_frames.append(sector_deduped_df)
    if show_external_events:
        pool_frames.append(external_deduped_df)
    overview_pool = pd.concat(pool_frames, ignore_index=True)
    available_sources = collect_sources(filter_news_by_time(overview_pool, time_range))
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

    today = market_today(overview_pool)
    followed_today = filter_overview_sectors(today, selected_sectors)
    hot_today, _ = select_hot_news(followed_today, selected_sectors)
    hot_links = set(hot_today["原文链接"].astype(str)) - {""}

    with brief_col:
        render_market_brief(followed_today, hot_today)
    with mood_col.container(border=True, key="market_mood"):
        render_market_mood(followed_today, len(hot_today))
        if selected_sectors:
            st.caption("统计范围：关注板块")
    with flow_col.container(key="overview_flow"):
        render_overview_flow(
            overview_pool, selected_sectors, time_range, keyword, selected_sources,
            max_items, hot_links,
        )
    with heat_col.container(border=True, key="heat_panel"):
        render_sector_heat(today)
        if show_sector_news:
            st.divider()
            render_daily_digest_section(
                llm_verifier, selected_sectors, sector_deduped_df,
                can_generate=admin and not config_unavailable,
            )
    with overview_footer:
        st.caption(
            "利好 / 利空 / 影响等级由规则与新闻文本自动生成，仅供信息参考，不构成投资建议。"
        )

    if show_sector_news:
        with tabs["板块新闻"]:
            st.subheader("板块新闻")
            st.caption(f"已选 {len(selected_sectors)} 个板块 · 时间范围：{time_range}")
            if not selected_sectors:
                st.info("在市场概览的「关注板块」中选择至少一个板块。")
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
        with tabs["外部事件"]:
            st.subheader("外部事件")
            st.caption(f"时间范围：{time_range}")
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
