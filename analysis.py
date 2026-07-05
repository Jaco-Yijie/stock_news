"""新闻利好/利空规则判断引擎与今日热点选择。

判断结果为派生数据，展示时即时计算，不写入新闻缓存，
因此对已有数据结构完全向后兼容，规则更新后旧新闻立即按新规则重判。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from analysis_rules import (
    CATEGORY_WEIGHTS,
    HIGH_IMPACT_SIGNALS,
    INDUSTRY_SIGNALS,
    MACRO_RULES,
    MEDIUM_IMPACT_SIGNALS,
    NEGATIVE_SIGNALS,
    POLICY_SIGNALS,
    POSITIVE_SIGNALS,
    STOCK_MOVE_SIGNALS,
)
from fetcher import _is_similar_title, _normalize_title_for_dedupe
from time_utils import now_utc8_naive


SENTIMENT_LABELS = {"positive": "利好", "neutral": "中性", "negative": "利空"}
IMPACT_LABELS = {"high": "高影响", "medium": "中影响", "low": "低影响"}
IMPACT_SCORES = {"high": 3.0, "medium": 2.0, "low": 1.0}
GENERIC_DIRECTION_THRESHOLD = 1.0
LOW_CONFIDENCE_THRESHOLD = 45
HOT_NEWS_LIMIT = 5
HOT_NEWS_MIN_TODAY = 3
LOW_CONFIDENCE_REASON = "判断依据不足"


def _match_signals(text: str, signals: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for keyword, weight in signals.items():
        if keyword.casefold() in text:
            score += weight
            hits.append(keyword)
    return score, hits


def _match_macro_rules(text: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    assessments: dict[str, tuple[str, str]] = {}
    hit_rules: list[str] = []
    for rule in MACRO_RULES:
        if not any(keyword.casefold() in text for keyword in rule["keywords"]):
            continue
        co_keywords = rule.get("co_keywords") or []
        if co_keywords and not any(keyword.casefold() in text for keyword in co_keywords):
            continue
        hit_rules.append(rule["name"])
        for sector, assessment in rule["sector_sentiments"].items():
            assessments.setdefault(sector, assessment)
    return assessments, hit_rules


def _news_category(text: str) -> str:
    """新闻类别：政策 > 产业 > 异动 > 一般资讯。

    政策信号优先；只命中盘面波动词的归为个股异动快讯（重要性最低）。
    """
    if any(keyword.casefold() in text for keyword in POLICY_SIGNALS):
        return "policy"
    if any(keyword.casefold() in text for keyword in INDUSTRY_SIGNALS):
        return "industry"
    if any(keyword.casefold() in text for keyword in STOCK_MOVE_SIGNALS):
        return "stock_move"
    return "general"


def _impact_level(text: str, macro_hit: bool, direction_strength: float) -> str:
    if any(keyword.casefold() in text for keyword in HIGH_IMPACT_SIGNALS):
        return "high"
    if (
        macro_hit
        or direction_strength >= 2.0
        or any(keyword.casefold() in text for keyword in MEDIUM_IMPACT_SIGNALS)
    ):
        return "medium"
    return "low"


def analyze_news_item(news: dict[str, Any], target_sectors: list[str]) -> dict[str, Any]:
    """按规则判断一条新闻的方向、影响与板块级评估。"""
    title = str(news.get("标题", "") or "")
    content = str(news.get("新闻内容", "") or "")
    text = f"{title} {content}".casefold()

    positive_score, positive_hits = _match_signals(text, POSITIVE_SIGNALS)
    negative_score, negative_hits = _match_signals(text, NEGATIVE_SIGNALS)
    macro_assessments, macro_hits = _match_macro_rules(text)

    direction_score = positive_score - negative_score
    if direction_score >= GENERIC_DIRECTION_THRESHOLD:
        generic_sentiment = "positive"
        generic_reason = "命中利好信号：" + "、".join(positive_hits[:4])
    elif direction_score <= -GENERIC_DIRECTION_THRESHOLD:
        generic_sentiment = "negative"
        generic_reason = "命中利空信号：" + "、".join(negative_hits[:4])
    else:
        generic_sentiment = "neutral"
        generic_reason = LOW_CONFIDENCE_REASON

    sector_assessments: dict[str, dict[str, str]] = {}
    for sector in target_sectors:
        sector = str(sector or "").strip()
        if not sector:
            continue
        if sector in macro_assessments:
            sentiment, reason = macro_assessments[sector]
        else:
            sentiment, reason = generic_sentiment, generic_reason
        sector_assessments[sector] = {"sentiment": sentiment, "reason": reason}

    directions = {item["sentiment"] for item in sector_assessments.values()}
    divergent = "positive" in directions and "negative" in directions
    if divergent:
        overall_sentiment = "neutral"
        overall_reason = "不同板块方向分化，整体按中性处理"
    elif "positive" in directions:
        overall_sentiment = "positive"
        overall_reason = next(
            item["reason"]
            for item in sector_assessments.values()
            if item["sentiment"] == "positive"
        )
    elif "negative" in directions:
        overall_sentiment = "negative"
        overall_reason = next(
            item["reason"]
            for item in sector_assessments.values()
            if item["sentiment"] == "negative"
        )
    elif sector_assessments:
        overall_sentiment = "neutral"
        overall_reason = next(iter(sector_assessments.values()))["reason"]
    else:
        overall_sentiment = generic_sentiment
        overall_reason = generic_reason

    signal_count = len(positive_hits) + len(negative_hits) + 2 * len(macro_hits)
    confidence = min(90, 20 + 20 * signal_count)

    # 依据不足时不强行给方向，统一按中性处理
    if overall_sentiment != "neutral" and confidence < LOW_CONFIDENCE_THRESHOLD:
        overall_sentiment = "neutral"
        overall_reason = LOW_CONFIDENCE_REASON

    direction_strength = max(abs(direction_score), 2.0 if macro_hits else 0.0)
    impact_level = _impact_level(text, bool(macro_hits), direction_strength)
    category = _news_category(text)
    importance = (
        CATEGORY_WEIGHTS.get(category, 1.0) * 2.0
        + IMPACT_SCORES.get(impact_level, 1.0)
        + confidence / 100.0
    )

    return {
        "sentiment": overall_sentiment,
        "impact_level": impact_level,
        "reason": overall_reason,
        "confidence": confidence,
        "divergent": divergent,
        "sector_assessments": sector_assessments,
        "macro_rules": macro_hits,
        "category": category,
        "importance": importance,
    }


def _related_sectors_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text == "未映射":
        return []
    return [item.strip() for item in text.replace("，", "、").split("、") if item.strip()]


def target_sectors_for_row(row: dict[str, Any]) -> list[str]:
    if str(row.get("news_type", "")) == "external_event":
        return _related_sectors_list(row.get("related_sectors"))
    sector = str(row.get("sector", "") or "").strip()
    return [sector] if sector else []


def analyze_display_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    """为展示 DataFrame 的每一行计算判断结果，返回与行对齐的列表。"""
    if df is None or df.empty:
        return []
    return [
        analyze_news_item(row, target_sectors_for_row(row))
        for row in df.to_dict("records")
    ]


def parse_publish_times(df: pd.DataFrame) -> pd.Series:
    """发布时间按北京时间（naive）解析。"""
    if df is None or df.empty or "发布时间" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df["发布时间"], errors="coerce")


def _hot_score(
    analysis: dict[str, Any],
    publish_time: pd.Timestamp,
    now: pd.Timestamp,
    related_to_selection: bool,
) -> float:
    # importance 已包含类别权重（政策>产业>资讯>异动）、影响等级与置信度
    score = float(analysis.get("importance", 2.0)) * 1.5
    if analysis["sentiment"] != "neutral" or analysis["divergent"]:
        score += 1.5
    if related_to_selection:
        score += 2.0
    if pd.notna(publish_time):
        hours_ago = max((now - publish_time).total_seconds() / 3600.0, 0.0)
        score += max(0.0, 1.5 - hours_ago / 16.0)
    return score


def select_hot_news(
    df: pd.DataFrame,
    selected_sectors: list[str],
    limit: int = HOT_NEWS_LIMIT,
) -> tuple[pd.DataFrame, bool]:
    """选出今日热点新闻。

    返回 (热点 DataFrame, 是否使用了最近 24 小时回退)。
    要求 df 带有 analysis 列（analyze_display_frame 的结果）。
    """
    if df is None or df.empty or "analysis" not in df.columns:
        return (df.iloc[0:0] if df is not None else pd.DataFrame()), False

    now = pd.Timestamp(now_utc8_naive())
    publish_times = parse_publish_times(df)
    selected_set = {str(s).strip() for s in selected_sectors if str(s).strip()}

    is_today = publish_times.dt.date.eq(now.date()).fillna(False)
    within_24h = publish_times.ge(now - pd.Timedelta(hours=24)).fillna(False)

    candidates: list[tuple[float, int]] = []
    fallback_candidates: list[tuple[float, int]] = []
    for position in range(len(df)):
        row = df.iloc[position]
        analysis = row["analysis"]
        if not isinstance(analysis, dict):
            continue

        row_sectors = set(target_sectors_for_row(row))
        related = bool(selected_set and (row_sectors & selected_set))
        # 外部宏观新闻只有与已选板块相关时才进入热点
        if str(row.get("news_type", "")) == "external_event" and selected_set and not related:
            continue

        score = _hot_score(analysis, publish_times.iloc[position], now, related)
        if bool(is_today.iloc[position]):
            candidates.append((score, position))
        elif bool(within_24h.iloc[position]):
            fallback_candidates.append((score, position))

    used_fallback = False
    if len(candidates) < HOT_NEWS_MIN_TODAY and fallback_candidates:
        used_fallback = True
        candidates.extend(fallback_candidates)

    candidates.sort(key=lambda item: item[0], reverse=True)

    # 同一事件的重复报道只保留一条（标准化标题 + 相似度去重）
    kept_positions: list[int] = []
    kept_titles: list[str] = []
    for _, position in candidates[:60]:
        title_key = _normalize_title_for_dedupe(df.iloc[position].get("标题", ""))
        if title_key and (title_key in kept_titles or _is_similar_title(title_key, kept_titles)):
            continue
        kept_positions.append(position)
        if title_key:
            kept_titles.append(title_key)
        if len(kept_positions) >= limit:
            break

    return df.iloc[kept_positions].reset_index(drop=True), used_fallback


def sort_news_by_importance(df: pd.DataFrame) -> pd.DataFrame:
    """按重要性降序排列（政策 > 产业事件 > 一般资讯 > 个股异动），同分按时间倒序。"""
    if df is None or df.empty or "analysis" not in df.columns:
        return df

    importance = df["analysis"].map(
        lambda item: float(item.get("importance", 0.0)) if isinstance(item, dict) else 0.0
    )
    publish_times = parse_publish_times(df)
    return (
        df.assign(_importance=importance, _publish=publish_times.values)
        .sort_values(["_importance", "_publish"], ascending=[False, False], na_position="last")
        .drop(columns=["_importance", "_publish"])
        .reset_index(drop=True)
    )


EVENT_CLUSTER_JACCARD = 0.5


def _title_bigrams(title_key: str) -> set[str]:
    if len(title_key) < 2:
        return {title_key} if title_key else set()
    return {title_key[i : i + 2] for i in range(len(title_key) - 1)}


def group_similar_news(df: pd.DataFrame) -> list[tuple[int, list[int]]]:
    """把同一事件的多条报道聚成组，返回 (主报道位置, 其余报道位置) 列表。

    输入应已按重要性排序，组内第一条（最重要的）作为主报道。
    用标准化标题的字符二元组 Jaccard 相似度 + 序列相似度双重判断，
    比纯标题相似度能合并更多"同事件不同措辞"的报道。
    """
    if df is None or df.empty:
        return []

    clusters: list[dict[str, Any]] = []
    for position in range(len(df)):
        title_key = _normalize_title_for_dedupe(df.iloc[position].get("标题", ""))
        bigrams = _title_bigrams(title_key)

        matched = None
        if title_key:
            for cluster in clusters:
                if not cluster["title_key"]:
                    continue
                union = bigrams | cluster["bigrams"]
                overlap = len(bigrams & cluster["bigrams"]) / len(union) if union else 0.0
                if overlap >= EVENT_CLUSTER_JACCARD:
                    matched = cluster
                    break
                if overlap >= 0.3 and _is_similar_title(title_key, [cluster["title_key"]]):
                    matched = cluster
                    break

        if matched is not None:
            matched["members"].append(position)
        else:
            clusters.append(
                {"title_key": title_key, "bigrams": bigrams, "members": [position]}
            )

    return [(cluster["members"][0], cluster["members"][1:]) for cluster in clusters]


def sentiment_counts(analyses: list[dict[str, Any]] | pd.Series) -> dict[str, int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for analysis in analyses:
        if isinstance(analysis, dict):
            counts[analysis.get("sentiment", "neutral")] = (
                counts.get(analysis.get("sentiment", "neutral"), 0) + 1
            )
    return counts
