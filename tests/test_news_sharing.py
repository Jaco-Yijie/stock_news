import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config_store
import news_store
from main import news_share_id, news_share_url
from paths import BASE_DIR


@pytest.mark.parametrize(("app_url", "public_url"), [
    ("https://example.streamlit.app/~/+/?embed=true", "https://example.streamlit.app/"),
    ("https://example.streamlit.app/", "https://example.streamlit.app/"),
    ("http://localhost:8501/news-board/?x=1#top", "http://localhost:8501/news-board/"),
    ("javascript:alert(1)", ""),
    (None, ""),
])
def test_share_url_targets_the_public_page(app_url, public_url):
    assert news_share_url("abc", app_url) == public_url + "?news=abc"


def test_news_id_preserves_article_and_analysis_context():
    row = pd.Series({"原文链接": "https://example.com/feed#story1", "sector": "半导体芯片"})
    original = news_share_id(row)
    row["analysis"] = {"confidence": 90}
    row["匹配关键词"] = "晶圆"
    assert news_share_id(row) == original
    for changes in ({"sector": "机器人"}, {"原文链接": "https://example.com/feed#story2"}):
        assert news_share_id(pd.Series({**row, **changes})) != original
    without_link = pd.Series({"标题": "第一条快讯", "发布时间": "2026-01-01"})
    assert news_share_id(without_link) == news_share_id(without_link.copy())
    without_link["发布时间"] = "2026-01-02"
    assert news_share_id(without_link) != news_share_id(pd.Series({"标题": "第一条快讯", "发布时间": "2026-01-01"}))


def test_shared_card_opens_in_a_new_session_without_dashboard_filters(monkeypatch):
    monkeypatch.setattr(config_store, "_get_config_store", lambda: None)
    cache = pd.DataFrame([{
        "title": "晶圆厂技术突破并量产", "content": "<script>已有的报道内容</script>",
        "sector": "半导体芯片", "source": "测试来源", "link": "https://example.com/story",
        "publish_time": "2025-01-01 10:00:00",
    }])
    monkeypatch.setattr(news_store, "read_cache", lambda: news_store.CacheReadResult(cache))
    monkeypatch.setattr(news_store, "cache_fingerprint", lambda: "share-detail-test")
    path = BASE_DIR / "main.py"
    origin = "https://example.streamlit.app"

    dashboard = AppTest.from_file(path)
    dashboard.session_state["cache_version"] = 910011
    dashboard.context["url"] = origin + "/~/+/"
    dashboard.run()
    assert not dashboard.exception
    cards = "".join(element.value for element in dashboard.html)
    assert "<h2>近期导读</h2>" in cards
    news_id = re.search(r'href="https://example\.streamlit\.app/\?news=([0-9a-f]{24})"', cards)[1]

    detail = AppTest.from_file(path)
    detail.session_state["cache_version"] = 910011
    detail.session_state["keyword_search"] = "无关筛选"
    detail.context["url"] = origin + "/~/+/"
    detail.query_params["news"] = news_id
    detail.run()
    assert not detail.exception
    assert not detail.tabs
    article = next(element.value for element in detail.html if "<article" in element.value)
    assert "晶圆厂技术突破并量产" in article
    assert 'href="https://example.com/story"' in article
    assert "详情 / 分享" not in article
    assert detail.text[0].value == "<script>已有的报道内容</script>"
    assert detail.code[0].value == origin + "/?news=" + news_id
    detail.button[0].click().run()
    assert not detail.exception
    assert "news" not in detail.query_params
    assert any(tab.label == "市场概览" for tab in detail.tabs)


@pytest.mark.parametrize(("news_id", "cache_error", "expected"), [
    ("invalid", None, "新闻链接无效"),
    ("c" * 24, None, "报道可能已过期"),
    ("c" * 24, "暂时不可用", "新闻缓存暂时无法读取"),
])
def test_shared_news_unavailable_states(monkeypatch, news_id, cache_error, expected):
    monkeypatch.setattr(config_store, "_get_config_store", lambda: None)
    empty = news_store.CacheReadResult(news_store.empty_cache_frame(), cache_error)
    monkeypatch.setattr(news_store, "read_cache", lambda: empty)
    monkeypatch.setattr(news_store, "cache_fingerprint", lambda: f"share-empty-{cache_error}")
    app = AppTest.from_file(BASE_DIR / "main.py")
    app.session_state["cache_version"] = 910012 if cache_error else 910013
    app.query_params["news"] = news_id
    app.run()
    assert not app.exception
    assert any(expected in element.value for element in (*app.info, *app.warning))
    assert [button.label for button in app.button] == ["返回新闻看板"]
