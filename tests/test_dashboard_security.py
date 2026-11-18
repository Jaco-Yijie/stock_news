from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import config_store
import llm_provider
import news_store
from news_links import safe_news_url
from paths import BASE_DIR


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
    monkeypatch.setattr(llm_provider, "load_llm_verifier_from_env", lambda: (None, None))
    app = AppTest.from_file(BASE_DIR / "main.py")
    app.secrets["DASHBOARD_ADMIN_PASSWORD"] = "synthetic-test-only"
    app.run()
    next(item for item in app.text_input if item.label == "管理员口令").set_value("synthetic-test-only")
    next(button for button in app.button if button.label == "登录").click().run()

    assert not app.exception
    labels = {button.label for button in app.button}
    assert "退出管理模式" in labels
    assert not ({"清空缓存", "全量刷新", "增量刷新"} & labels)
    assert any("管理操作已暂停" in item.value for item in app.warning)
