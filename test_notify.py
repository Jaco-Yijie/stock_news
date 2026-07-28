from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import notify
from analysis import analyze_display_frame
from notify import (
    TELEGRAM_MESSAGE_LIMIT,
    PushPlusNotifier,
    TelegramNotifier,
    filter_unpushed,
    format_push_html,
    format_push_markdown,
    hashes_for,
    load_push_history,
    save_push_history,
    select_push_worthy,
    send_to_all,
)
from time_utils import now_utc8_naive


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _display_row(title: str, publish_time: str, link: str, sector: str = "半导体芯片") -> dict:
    return {
        "news_type": "sector_news",
        "sector": sector,
        "标题": title,
        "来源媒体": "测试媒体",
        "发布时间": publish_time,
        "原文链接": link,
        "匹配关键词": "测试",
        "新闻内容": "",
        "event_category": "",
        "related_sectors": "",
        "reason": "",
    }


def _one_row_df(title: str = "证监会发布重磅新规") -> pd.DataFrame:
    df = pd.DataFrame([_display_row(title, "2026-07-05 09:00:00", "http://e.com/x")])
    df["analysis"] = analyze_display_frame(df)
    return df


def test_serverchan_channel_is_gone() -> None:
    assert not hasattr(notify, "ServerChanNotifier")

    import os

    os.environ["SERVERCHAN_SENDKEY"] = "SCT123abc"
    try:
        notifiers = notify.load_notifiers_from_env()
    finally:
        os.environ.pop("SERVERCHAN_SENDKEY", None)
    assert notifiers == [], "SERVERCHAN_SENDKEY 不应再产生任何通道"


def test_notifiers_send_expected_requests() -> None:
    calls = []

    def fake_post(url, data=None, json=None, timeout=None):
        calls.append((url, data, json))
        if "pushplus" in url:
            return FakeResponse({"code": 200})
        return FakeResponse({"ok": True})

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all(
            [TelegramNotifier("bot789", "chat001"), PushPlusNotifier("token456")],
            "测试标题",
            _one_row_df(),
        )
    finally:
        notify.requests.post = original_post

    assert errors == []
    telegram_payload = calls[0][2]
    assert "api.telegram.org/botbot789/sendMessage" in calls[0][0]
    assert telegram_payload["chat_id"] == "chat001"
    assert telegram_payload["parse_mode"] == "HTML"
    assert telegram_payload["disable_web_page_preview"] is True
    assert "<b>测试标题</b>" in telegram_payload["text"]
    # Telegram 用 HTML，不能出现 markdown 的原样语法
    assert "**" not in telegram_payload["text"]
    assert "[原文]" not in telegram_payload["text"]
    assert '<a href="http://e.com/x">原文</a>' in telegram_payload["text"]
    # PushPlus 仍然收 markdown
    assert calls[1][2]["template"] == "markdown"
    assert "**【" in calls[1][2]["content"]


def test_telegram_escapes_html_special_chars() -> None:
    calls = []

    def fake_post(url, data=None, json=None, timeout=None):
        calls.append(json)
        return FakeResponse({"ok": True})

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all(
            [TelegramNotifier("bot", "chat")],
            "标题",
            _one_row_df("<b>营收</b> 增长 & 毛利 > 30%"),
        )
    finally:
        notify.requests.post = original_post

    assert errors == []
    text = calls[0]["text"]
    assert "&lt;b&gt;营收&lt;/b&gt;" in text
    assert "增长 &amp; 毛利 &gt; 30%" in text


def test_telegram_splits_long_message() -> None:
    calls = []

    def fake_post(url, data=None, json=None, timeout=None):
        calls.append(json)
        return FakeResponse({"ok": True})

    rows = [
        _display_row(f"证监会发布重磅新规第{index}号", "2026-07-05 09:00:00", f"http://e.com/{index}")
        for index in range(120)
    ]
    df = pd.DataFrame(rows)
    df["analysis"] = analyze_display_frame(df)

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all([TelegramNotifier("bot", "chat")], "标题", df)
    finally:
        notify.requests.post = original_post

    assert errors == []
    assert len(calls) > 1, "超过 4096 字符必须分段"
    for call in calls:
        assert len(call["text"]) <= TELEGRAM_MESSAGE_LIMIT + 64
    # 分段不能切开条目，否则 HTML 标签会残缺
    for call in calls:
        assert call["text"].count("<a href=") == call["text"].count("</a>")
        assert call["text"].count("<b>") == call["text"].count("</b>")


def test_send_to_all_collects_errors() -> None:
    def fake_post(url, data=None, json=None, timeout=None):
        return FakeResponse({"ok": False, "description": "chat not found"})

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all([TelegramNotifier("bot", "bad")], "标题", _one_row_df())
    finally:
        notify.requests.post = original_post
    assert len(errors) == 1 and "Telegram" in errors[0]
    assert "chat not found" in errors[0]


def test_select_push_worthy_rules() -> None:
    now = now_utc8_naive()
    recent = now.strftime("%Y-%m-%d %H:%M:%S")
    old = (now - pd.Timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [
            _display_row("证监会发布重磅新规", recent, "http://e.com/policy"),
            _display_row("某股盘中拉升涨停", recent, "http://e.com/move"),
            _display_row("三天前的旧政策：国务院部署产业规划", old, "http://e.com/old"),
        ]
    )
    df["analysis"] = analyze_display_frame(df)
    selected = select_push_worthy(df)
    links = set(selected["原文链接"])
    assert "http://e.com/policy" in links
    assert "http://e.com/move" not in links
    assert "http://e.com/old" not in links


def test_push_history_roundtrip_and_dedup(tmp_path: Path = None) -> None:
    path = (tmp_path or Path("/tmp")) / "push_history_test.json"
    if path.exists():
        path.unlink()

    df = pd.DataFrame(
        [
            _display_row("新闻A", "2026-07-05 09:00:00", "http://e.com/a"),
            _display_row("新闻B", "2026-07-05 09:10:00", "http://e.com/b"),
        ]
    )
    hashes = hashes_for(df)
    save_push_history(hashes[:1], path=path)
    history = load_push_history(path=path)
    assert history == hashes[:1]

    fresh = filter_unpushed(df, history)
    assert list(fresh["原文链接"]) == ["http://e.com/b"]


def test_load_notifiers_cleans_pasted_secrets() -> None:
    import os

    os.environ["TELEGRAM_BOT_TOKEN"] = " 12345:AAH\nabc "
    os.environ["TELEGRAM_CHAT_ID"] = " 987 654 "
    try:
        notifiers = notify.load_notifiers_from_env()
    finally:
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
    telegram = [n for n in notifiers if n.name == "Telegram"]
    assert telegram, "应识别出 Telegram 通道"
    assert telegram[0]._url == "https://api.telegram.org/bot12345:AAHabc/sendMessage"
    assert telegram[0]._chat_id == "987654"


def test_telegram_requires_both_token_and_chat_id() -> None:
    import os

    os.environ["TELEGRAM_BOT_TOKEN"] = "12345:AAH"
    os.environ.pop("TELEGRAM_CHAT_ID", None)
    try:
        assert notify.load_notifiers_from_env() == []
    finally:
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)


def test_http_error_includes_response_body() -> None:
    def fake_post(url, data=None, json=None, timeout=None):
        return FakeResponse(
            {"ok": False, "description": "bot can't initiate conversation"},
            status_code=403,
        )

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all([TelegramNotifier("bot", "chat")], "标题", _one_row_df())
    finally:
        notify.requests.post = original_post
    assert len(errors) == 1
    assert "403" in errors[0] and "initiate conversation" in errors[0]


def test_format_push_markdown_contains_labels() -> None:
    df = pd.DataFrame(
        [_display_row("证监会发布利好新规", "2026-07-05 09:00:00", "http://e.com/x")]
    )
    df["analysis"] = analyze_display_frame(df)
    text = format_push_markdown(df)
    assert "证监会发布利好新规" in text
    assert "板块：半导体芯片" in text
    assert "http://e.com/x" in text


def test_overview_is_prepended_in_both_formats() -> None:
    df = _one_row_df()
    markdown = format_push_markdown(df, "今日整体偏利好。")
    html = format_push_html(df, "今日整体偏利好。")
    assert markdown.startswith("今日整体偏利好。")
    assert html.startswith("今日整体偏利好。")
    assert "证监会发布重磅新规" in markdown and "证监会发布重磅新规" in html


if __name__ == "__main__":
    test_serverchan_channel_is_gone()
    test_notifiers_send_expected_requests()
    test_telegram_escapes_html_special_chars()
    test_telegram_splits_long_message()
    test_send_to_all_collects_errors()
    test_select_push_worthy_rules()
    test_push_history_roundtrip_and_dedup()
    test_load_notifiers_cleans_pasted_secrets()
    test_telegram_requires_both_token_and_chat_id()
    test_http_error_includes_response_body()
    test_format_push_markdown_contains_labels()
    test_overview_is_prepended_in_both_formats()
    print("test_notify.py: ok")
