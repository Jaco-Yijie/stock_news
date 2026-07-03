from __future__ import annotations

import pandas as pd

from analysis import (
    analyze_display_frame,
    analyze_news_item,
    select_hot_news,
    sentiment_counts,
)
from time_utils import now_utc8_naive


def test_oil_up_divergent_sectors() -> None:
    news = {"标题": "国际油价大涨创阶段新高", "新闻内容": "OPEC 减产推动原油价格上涨"}
    result = analyze_news_item(news, ["石油石化", "化工", "半导体芯片"])
    assert result["sector_assessments"]["石油石化"]["sentiment"] == "positive"
    assert result["sector_assessments"]["化工"]["sentiment"] == "negative"
    assert result["divergent"] is True
    assert result["sentiment"] == "neutral"


def test_sanctions_negative_for_chips_positive_for_xinchuang() -> None:
    news = {"标题": "美国宣布新一轮出口管制，多家企业列入实体清单", "新闻内容": ""}
    result = analyze_news_item(news, ["半导体芯片", "信创软件"])
    assert result["sector_assessments"]["半导体芯片"]["sentiment"] == "negative"
    assert result["sector_assessments"]["信创软件"]["sentiment"] == "positive"
    assert result["impact_level"] == "high"


def test_generic_positive_signals() -> None:
    news = {"标题": "公司签订大额订单，全年业绩预增", "新闻内容": ""}
    result = analyze_news_item(news, ["机器人"])
    assert result["sentiment"] == "positive"
    assert result["confidence"] >= 45
    assert result["sector_assessments"]["机器人"]["sentiment"] == "positive"


def test_ai_compute_demand_positive() -> None:
    news = {"标题": "AI服务器需求增长，HBM 供不应求", "新闻内容": ""}
    result = analyze_news_item(news, ["半导体芯片", "人工智能"])
    assert result["sector_assessments"]["半导体芯片"]["sentiment"] == "positive"
    assert result["sector_assessments"]["人工智能"]["sentiment"] == "positive"
    assert result["sentiment"] == "positive"


def test_low_confidence_defaults_to_neutral() -> None:
    news = {"标题": "公司召开例行股东大会", "新闻内容": ""}
    result = analyze_news_item(news, ["银行"])
    assert result["sentiment"] == "neutral"
    assert result["reason"] == "判断依据不足"


def _display_row(
    title: str,
    publish_time: str,
    sector: str = "半导体芯片",
    news_type: str = "sector_news",
    related_sectors: str = "",
    link: str = "",
) -> dict:
    return {
        "news_type": news_type,
        "sector": sector,
        "标题": title,
        "来源媒体": "测试媒体",
        "发布时间": publish_time,
        "原文链接": link or f"http://e.com/{title}",
        "匹配关键词": "测试",
        "新闻内容": "",
        "event_category": "",
        "related_sectors": related_sectors,
        "reason": "",
    }


def test_hot_news_selection_prefers_today_high_impact() -> None:
    now = now_utc8_naive()
    today = now.strftime("%Y-%m-%d %H:%M:%S")
    old = (now - pd.Timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(
        [
            _display_row("美国宣布新一轮出口管制涉及芯片设备", today),
            _display_row("公司召开例行股东大会", today),
            _display_row("十天前的旧闻：某公司签订大额订单", old),
        ]
    )
    df["analysis"] = analyze_display_frame(df)

    hot_df, used_fallback = select_hot_news(df, ["半导体芯片"])
    assert not hot_df.empty
    assert hot_df.iloc[0]["标题"] == "美国宣布新一轮出口管制涉及芯片设备"
    titles = set(hot_df["标题"])
    assert "十天前的旧闻：某公司签订大额订单" not in titles


def test_hot_news_dedupes_similar_titles() -> None:
    now = now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [
            _display_row("美国宣布新一轮出口管制涉及芯片设备", now, link="http://a.com/1"),
            _display_row("美国宣布新一轮出口管制涉及芯片设备！", now, link="http://b.com/2"),
        ]
    )
    df["analysis"] = analyze_display_frame(df)
    hot_df, _ = select_hot_news(df, ["半导体芯片"])
    assert len(hot_df) == 1


def test_external_event_only_hot_when_related_to_selection() -> None:
    now = now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [
            _display_row(
                "美联储宣布降息，流动性宽松",
                now,
                sector="",
                news_type="external_event",
                related_sectors="黄金、证券",
            ),
        ]
    )
    df["analysis"] = analyze_display_frame(df)

    hot_df, _ = select_hot_news(df, ["半导体芯片"])
    assert hot_df.empty

    hot_df, _ = select_hot_news(df, ["黄金"])
    assert len(hot_df) == 1


def test_sentiment_counts() -> None:
    analyses = [
        {"sentiment": "positive"},
        {"sentiment": "positive"},
        {"sentiment": "negative"},
        {"sentiment": "neutral"},
    ]
    counts = sentiment_counts(analyses)
    assert counts == {"positive": 2, "neutral": 1, "negative": 1}


if __name__ == "__main__":
    test_oil_up_divergent_sectors()
    test_sanctions_negative_for_chips_positive_for_xinchuang()
    test_generic_positive_signals()
    test_ai_compute_demand_positive()
    test_low_confidence_defaults_to_neutral()
    test_hot_news_selection_prefers_today_high_impact()
    test_hot_news_dedupes_similar_titles()
    test_external_event_only_hot_when_related_to_selection()
    test_sentiment_counts()
    print("test_analysis.py: ok")
