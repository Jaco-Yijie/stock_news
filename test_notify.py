from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import notify
from analysis import analyze_display_frame
from notify import (
    PushPlusNotifier,
    ServerChanNotifier,
    TelegramNotifier,
    filter_unpushed,
    format_push_markdown,
    hashes_for,
    load_push_history,
    save_push_history,
    select_push_worthy,
    send_to_all,
)
from time_utils import now_utc8_naive


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

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


def test_notifiers_send_expected_requests() -> None:
    calls = []

    def fake_post(url, data=None, json=None, timeout=None):
        calls.append((url, data, json))
        if "sctapi" in url:
            return FakeResponse({"code": 0})
        if "pushplus" in url:
            return FakeResponse({"code": 200})
        return FakeResponse({"ok": True})

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all(
            [
                ServerChanNotifier("sendkey123"),
                PushPlusNotifier("token456"),
                TelegramNotifier("bot789", "chat001"),
            ],
            "测试标题",
            "测试内容",
        )
    finally:
        notify.requests.post = original_post

    assert errors == []
    assert "sctapi.ftqq.com/sendkey123.send" in calls[0][0]
    assert calls[1][2]["template"] == "markdown"
    assert calls[2][2]["chat_id"] == "chat001"


def test_send_to_all_collects_errors() -> None:
    def fake_post(url, data=None, json=None, timeout=None):
        return FakeResponse({"code": 999, "message": "bad key"})

    original_post = notify.requests.post
    notify.requests.post = fake_post
    try:
        errors = send_to_all([ServerChanNotifier("bad")], "标题", "内容")
    finally:
        notify.requests.post = original_post
    assert len(errors) == 1 and "Server酱" in errors[0]


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


def test_format_push_markdown_contains_labels() -> None:
    df = pd.DataFrame(
        [_display_row("证监会发布利好新规", "2026-07-05 09:00:00", "http://e.com/x")]
    )
    df["analysis"] = analyze_display_frame(df)
    text = format_push_markdown(df)
    assert "证监会发布利好新规" in text
    assert "板块：半导体芯片" in text
    assert "http://e.com/x" in text


if __name__ == "__main__":
    test_notifiers_send_expected_requests()
    test_send_to_all_collects_errors()
    test_select_push_worthy_rules()
    test_push_history_roundtrip_and_dedup()
    test_format_push_markdown_contains_labels()
    print("test_notify.py: ok")
