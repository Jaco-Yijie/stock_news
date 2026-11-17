from __future__ import annotations

import pandas as pd
import pytest

from analysis import (
    analyze_display_frame,
    analyze_news_item,
    group_similar_news,
    select_hot_news,
    sentiment_counts,
    sort_news_by_importance,
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


@pytest.mark.parametrize(("title", "sector", "expected"), [
    ("算力需求同比下降，云厂商削减资本开支", "算力数据中心", "negative"),
    ("云厂商上调资本开支，算力订单增长", "半导体芯片", "positive"),
    ("iPhone 订单下滑，Apple 被迫砍单", "消费电子", "negative"),
    ("欧盟宣布取消对家电加征关税", "家电", "positive"),
    ("美国宣布加征家电进口关税", "家电", "negative"),
    ("欧盟讨论家电关税安排", "家电", "neutral"),
    ("美国宣布放宽芯片出口限制", "半导体芯片", "positive"),
    ("传闻称美国将收紧芯片出口管制", "半导体芯片", "neutral"),
    ("美联储维持利率不变，否认即将加息", "黄金", "neutral"),
    ("美联储降息？", "黄金", "neutral"),
    ("中国央行宣布降准", "证券", "positive"),
    ("中国央行宣布降准", "银行", "neutral"),
    ("欧洲央行宣布降息", "房地产", "neutral"),
    ("公司净利润同比下降30%", "机器人", "negative"),
    ("公司经营现金流转正", "机器人", "positive"),
    ("行业龙头债券违约", "银行", "negative"),
    ("机构公布设备项目中标候选人名单", "机器人", "neutral"),
    ("公司中标轨道交通设备项目", "机器人", "positive"),
    ("创新药三期临床达到主要终点", "创新药", "positive"),
    ("创新药三期临床未能达到主要终点", "创新药", "negative"),
    ("公司否认已获批并购传闻", "证券", "neutral"),
    ("行业价格下跌", "有色金属", "neutral"),
    ("纯盘面：芯片板块拉升，多股涨停", "半导体芯片", "neutral"),
    ("公司中标后公告取消订单", "机器人", "neutral"),
    ("公司净利润增长，但经营现金流转负", "机器人", "neutral"),
    ("国际油价大涨后又下跌", "石油石化", "neutral"),
])
def test_direction_uses_event_context(title: str, sector: str, expected: str) -> None:
    result = analyze_news_item({"标题": title}, [sector])
    assert result["sentiment"] == result["sector_assessments"][sector]["sentiment"] == expected


def test_commodity_prices_split_upstream_and_downstream() -> None:
    result = analyze_news_item({"标题": "碳酸锂价格大涨"}, ["有色金属", "新能源汽车", "储能"])
    assert result["sentiment"] == "neutral" and result["divergent"]
    assert [value["sentiment"] for value in result["sector_assessments"].values()] == [
        "positive", "negative", "negative",
    ]


def test_title_direction_takes_priority_over_background_and_rumours() -> None:
    body = "公司净利润同比增长30%。"
    assert analyze_news_item({"标题": "公司净利润下滑", "新闻内容": body}, [])["sentiment"] == "negative"
    assert analyze_news_item({"标题": "公司公告业绩", "新闻内容": body}, [])["sentiment"] == "positive"
    assert analyze_news_item({"标题": "公司否认净利润增长传闻", "新闻内容": body}, [])["sentiment"] == "neutral"


def test_macro_subject_and_direction_must_share_a_clause() -> None:
    result = analyze_news_item({"标题": "油价持稳，海外零售额上涨"}, ["石油石化"])
    assert result["sentiment"] == "neutral" and not result["macro_rules"]


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


def test_importance_ranking_policy_over_stock_move() -> None:
    policy = analyze_news_item({"标题": "证监会发布新规征求意见", "新闻内容": ""}, ["证券"])
    industry = analyze_news_item({"标题": "龙头厂商宣布扩产，订单饱满", "新闻内容": ""}, ["光伏"])
    stock_move = analyze_news_item({"标题": "半导体板块盘中拉升，多股涨停", "新闻内容": ""}, ["半导体芯片"])
    assert policy["category"] == "policy"
    assert industry["category"] == "industry"
    assert stock_move["category"] == "stock_move"
    assert policy["importance"] > industry["importance"] > stock_move["importance"]


def test_sort_news_by_importance() -> None:
    now = now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [
            _display_row("半导体板块盘中拉升，多股涨停", now),
            _display_row("工信部发布半导体产业支持政策", now),
        ]
    )
    df["analysis"] = analyze_display_frame(df)
    sorted_df = sort_news_by_importance(df)
    assert sorted_df.iloc[0]["标题"] == "工信部发布半导体产业支持政策"


def test_group_similar_news_clusters_same_event() -> None:
    now = now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [
            _display_row("英伟达发布新一代 GPU 芯片，算力大幅提升", now, link="http://a.com/1"),
            _display_row("英伟达正式发布新一代 GPU 芯片 算力提升明显", now, link="http://b.com/2"),
            _display_row("光伏组件价格企稳回升", now, link="http://c.com/3"),
        ]
    )
    df["analysis"] = analyze_display_frame(df)
    clusters = group_similar_news(df)
    assert len(clusters) == 2
    primary, related = clusters[0]
    assert len(related) == 1


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
    test_importance_ranking_policy_over_stock_move()
    test_sort_news_by_importance()
    test_group_similar_news_clusters_same_event()
    test_sentiment_counts()
    print("test_analysis.py: ok")
