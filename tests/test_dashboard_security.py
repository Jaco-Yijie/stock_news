from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config_store
import llm_provider
import news_store
from news_links import safe_news_url
from paths import BASE_DIR
from time_utils import now_utc8_naive


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "data:text/html,x", "file:///etc/hosts",
    "//example.com", "https://user:pass@example.com", "https://example.com/\nx",
    'https://example.com/"x', "https://", "https://example.com:invalid",
])
def test_rejects_unsafe_news_urls(url):
    assert safe_news_url(url) == ""


def test_accepts_https_links_with_normal_query_values():
    url = "https://example.com/news?a=1&b=%E6%96%B0%E9%97%BB"
    assert safe_news_url(url) == url


def test_dashboard_tabs_follow_news_scope_and_keep_empty_state(monkeypatch):
    monkeypatch.setattr(config_store, "_get_config_store", lambda: None)
    monkeypatch.setattr(news_store, "get_supabase_store", lambda: None)
    monkeypatch.setattr(llm_provider, "load_llm_verifier_from_env", lambda: (None, None))
    app = AppTest.from_file(BASE_DIR / "main.py").run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["市场概览", "板块新闻", "外部事件"]
    scope = next(radio for radio in app.sidebar.radio if radio.label == "新闻类型")
    scope.set_value("外部事件").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["市场概览", "外部事件"]

    scope = next(radio for radio in app.sidebar.radio if radio.label == "新闻类型")
    scope.set_value("板块新闻").run()
    app.button(key="clear_all_sectors").click().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["市场概览", "板块新闻"]
    assert any("选择至少一个板块" in message.value for message in app.tabs[1].info)


def test_dashboard_writes_require_admin_and_logout_revokes_access(monkeypatch):
    monkeypatch.setattr(config_store, "_get_config_store", lambda: None)
    monkeypatch.setattr(news_store, "get_supabase_store", lambda: None)
    monkeypatch.setattr(llm_provider, "load_llm_verifier_from_env", lambda: (object(), None))
    monkeypatch.delenv("DASHBOARD_ADMIN_PASSWORD", raising=False)
    app_path = BASE_DIR / "main.py"
    disabled = AppTest.from_file(app_path).run()
    assert not disabled.exception
    assert not any(button.label == "清空缓存" for button in disabled.button)

    app = AppTest.from_file(app_path)
    app.secrets["DASHBOARD_ADMIN_PASSWORD"] = "synthetic-test-only"
    app.run()

    def by_label(elements, label):
        return next(element for element in elements if element.label == label)

    def labels():
        return {button.label for button in app.button}

    assert "重新加载缓存" in labels()
    assert not ({"清空缓存", "全量刷新", "生成 / 更新今日日报"} & labels())
    by_label(app.text_input, "管理员口令").set_value("wrong")
    by_label(app.button, "登录").click().run()
    assert "清空缓存" not in labels()
    assert any("口令不正确" in item.value for item in app.error)

    by_label(app.text_input, "管理员口令").set_value("synthetic-test-only")
    by_label(app.button, "登录").click().run()
    assert not app.exception
    assert {"退出管理模式", "清空缓存", "生成 / 更新今日日报"} <= labels()

    by_label(app.button, "退出管理模式").click().run()
    assert "清空缓存" not in labels()


def test_dashboard_pauses_admin_writes_when_remote_config_is_unreadable(monkeypatch):
    class DownConfigStore:
        def get_value(self, key):
            raise RuntimeError("Supabase 503")

        def set_value(self, *args, **kwargs):
            raise AssertionError("配置读取失败时不应写远端配置")

    monkeypatch.setattr(config_store, "_get_config_store", lambda: DownConfigStore())
    monkeypatch.setattr(news_store, "get_supabase_store", lambda: None)
    monkeypatch.setattr(llm_provider, "load_llm_verifier_from_env", lambda: (object(), None))
    app = AppTest.from_file(BASE_DIR / "main.py")
    app.secrets["DASHBOARD_ADMIN_PASSWORD"] = "synthetic-test-only"
    app.run()
    next(item for item in app.text_input if item.label == "管理员口令").set_value("synthetic-test-only")
    next(button for button in app.button if button.label == "登录").click().run()

    assert not app.exception
    labels = {button.label for button in app.button}
    assert "退出管理模式" in labels
    assert not ({"清空缓存", "全量刷新", "增量刷新", "生成 / 更新今日日报"} & labels)
    assert any("管理操作已暂停" in item.value for item in app.warning)


def test_overview_filters_keep_followed_sectors_and_handle_no_matches(monkeypatch):
    monkeypatch.setattr(config_store, "_get_config_store", lambda: None)
    monkeypatch.setattr(llm_provider, "load_llm_verifier_from_env", lambda: (None, None))
    titles = ["晶圆厂技术突破并量产", "封测企业订单下降并停产", "机器人产业举办技术交流会"]
    cache = pd.DataFrame(
        {
            "sector": ["半导体芯片", "半导体芯片", "机器人"],
            "title": titles,
            "source": "测试媒体",
            "publish_time": now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S"),
            "link": ["https://example.com/positive", "https://example.com/negative", "https://example.com/robot"],
        }
    )
    monkeypatch.setattr(news_store, "read_cache", lambda: news_store.CacheReadResult(cache))
    monkeypatch.setattr(news_store, "cache_fingerprint", lambda: "overview-filters-test")
    app = AppTest.from_file(BASE_DIR / "main.py")
    app.session_state["cache_version"] = 987654
    app.run()

    def overview_cards():
        return "".join(
            item.value for item in app.tabs[0].markdown if 'class="news-card ' in item.value
        )

    assert not app.exception
    assert titles[0] in overview_cards() and titles[1] in overview_cards()
    app.button(key="overview_negative").click().run()
    assert titles[0] not in overview_cards()
    assert titles[1] in overview_cards()

    app.button(key="overview_negative").click().run()
    app.button(key="overview_heat::机器人").click().run()
    assert titles[0] not in overview_cards()
    assert titles[2] in overview_cards()
    assert app.multiselect(key="followed_sectors").value == ["半导体芯片"]

    app.text_input(key="keyword_search").set_value("没有匹配的测试内容").run()
    assert not app.exception
    assert not overview_cards()
    assert any("当前筛选条件下暂无相关新闻" in item.value for item in app.tabs[0].info)


def test_shared_news_card_escapes_text_and_only_links_safe_urls():
    from main import render_news_item

    row = pd.Series(
        {
            "标题": "<script>headline</script>",
            "来源媒体": "<source>",
            "原文链接": "javascript:alert(1)",
            "news_type": "external_event",
            "related_sectors": "<sector>",
            "analysis": {
                "sentiment": "negative",
                "reason": "<reason>",
                "confidence": 53,
                "sector_assessments": {"<sector>": {"sentiment": "negative"}},
            },
        }
    )
    html = render_news_item(row)
    for value in ("script", "source", "sector", "reason"):
        assert f"&lt;{value}&gt;" in html
    assert '<article class="news-card mood-negative">' in html
    assert 'aria-label="置信度 53 / 100"' in html
    assert "href=" not in html

    row["原文链接"] = "https://example.com/story?a=1&b=2"
    html = render_news_item(row)
    assert 'href="https://example.com/story?a=1&amp;b=2"' in html
    assert 'rel="noopener noreferrer"' in html
