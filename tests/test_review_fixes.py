"""代码审查发现的 5 个问题的回归测试，均不触网。"""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import config_store
import news_store
import notify
import refresh_job
from agent import compare_report
from fetcher import _normalize_link_for_dedupe, _normalize_stream_frame, deduplicate_news
from supabase_store import SupabaseError
from time_utils import now_utc8_naive

CLS_BASE = "https://www.cls.cn/telegraph"


def _telegraph_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "标题": ["央行宣布降准0.5个百分点", "工信部印发集成电路产业行动方案", "宁德时代签下百亿储能订单"],
            "内容": [
                "央行今日宣布全面降准释放长期资金一万亿",
                "工信部今日印发集成电路产业高质量发展行动方案",
                "宁德时代与海外客户签署储能系统供货协议金额超百亿",
            ],
            "发布时间": ["2026-11-18 09:00:00", "2026-11-18 09:05:00", "2026-11-18 09:10:00"],
        }
    )
    return _normalize_stream_frame(raw, "财联社电报", CLS_BASE)


# ---- 1. 快讯链接去重 ----

def test_stream_items_from_one_source_are_not_collapsed_by_link_dedupe():
    frame = _telegraph_frame()
    assert len(deduplicate_news(frame)) == 3
    # 同一条快讯重复出现时仍然只保留一条
    assert len(deduplicate_news(pd.concat([frame, frame], ignore_index=True))) == 3


def test_incremental_filter_keeps_new_stream_items_when_cache_already_has_one():
    frame = _telegraph_frame()
    existing = news_store.display_to_cache("半导体芯片", frame.iloc[[0]], fetched_at="2026-11-18T01:00:00Z")
    fetched = news_store.display_to_cache("银行", frame, fetched_at="2026-11-18T01:30:00Z")

    incremental = news_store.filter_incremental_news(existing, fetched)

    assert sorted(incremental["title"]) == sorted(frame["标题"].iloc[[1, 2]])


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        # 普通文章的锚点、查询参数不是新闻身份，仍然去掉
        ("https://Example.com/a/1.html?from=x#comments", "https://example.com/a/1.html"),
        # 快讯列表页上不是内容指纹的锚点也去掉
        (f"{CLS_BASE}#top", CLS_BASE),
        (f"{CLS_BASE}#0123456789ab", f"{CLS_BASE}#0123456789ab"),
        ("https://finance.sina.com.cn/7x24/#0123456789ab", "https://finance.sina.com.cn/7x24#0123456789ab"),
    ],
)
def test_link_dedupe_key_keeps_only_stream_fingerprints(link, expected):
    assert _normalize_link_for_dedupe(link) == expected


# ---- 2. 推送历史 ----

class _FakeConfigStore:
    """按需失败的 app_config 替身；saved 记录写入内容。"""

    saved: dict[str, str] = {}

    def __init__(self, error: Exception | None = None, value: str | None = None) -> None:
        self.error = error
        self.value = value

    def get_value(self, key: str) -> str | None:
        if self.error is not None:
            raise self.error
        return self.value

    def set_value(self, key: str, value: str, **_: object) -> None:
        type(self).saved[key] = value


def _use_push_store(monkeypatch, store: _FakeConfigStore) -> None:
    _FakeConfigStore.saved = {}
    monkeypatch.setattr(notify, "load_supabase_credentials", lambda: ("https://example.supabase.co", "k"))
    monkeypatch.setattr(notify, "SupabaseConfigStore", lambda *args, **kwargs: store)


def test_push_history_read_failure_is_not_reported_as_empty_history(monkeypatch, tmp_path):
    _use_push_store(monkeypatch, _FakeConfigStore(SupabaseError("HTTP 503", status_code=503)))

    with pytest.raises(notify.PushHistoryUnavailable):
        notify.load_push_history(tmp_path / "push_history.json")


def test_push_history_falls_back_to_local_file_when_table_is_missing(monkeypatch, tmp_path):
    _use_push_store(monkeypatch, _FakeConfigStore(SupabaseError("HTTP 404", status_code=404)))
    path = tmp_path / "push_history.json"
    path.write_text(json.dumps(["abc"]), encoding="utf-8")

    assert notify.load_push_history(path) == ["abc"]


def _policy_cache(sectors: tuple[str, ...]) -> pd.DataFrame:
    row = {
        "标题": "国务院印发意见 支持半导体与人工智能产业发展",
        "来源媒体": "新华社",
        "发布时间": now_utc8_naive().strftime("%Y-%m-%d %H:%M:%S"),
        "原文链接": "https://example.com/a/1.html",
        "匹配关键词": "半导体",
        "新闻内容": "国务院印发意见",
    }
    return pd.concat(
        [
            news_store.display_to_cache(sector, pd.DataFrame([row]), fetched_at="2026-11-18T01:00:00Z")
            for sector in sectors
        ],
        ignore_index=True,
    )


class _RecordingNotifier:
    name = "记录"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def render(self, df: pd.DataFrame, overview: str = "") -> str:
        return "\n".join(df["标题"])

    def send(self, title: str, content: str) -> None:
        self.sent.append(content)


def test_important_push_is_skipped_and_history_untouched_when_history_unreadable(monkeypatch, capsys):
    _use_push_store(monkeypatch, _FakeConfigStore(SupabaseError("HTTP 503", status_code=503)))
    notifier = _RecordingNotifier()
    monkeypatch.setattr(notify, "load_notifiers_from_env", lambda: [notifier])

    notify.push_important_news(_policy_cache(("半导体芯片",)))

    assert notifier.sent == []
    assert _FakeConfigStore.saved == {}
    assert "本次跳过" in capsys.readouterr().out


# ---- 4. 同一条新闻命中多个板块 ----

def test_news_matching_several_sectors_is_pushed_once(monkeypatch, tmp_path):
    _use_push_store(monkeypatch, _FakeConfigStore(value=None))
    # 本地历史文件路径是函数默认参数，定义时已绑定，只能替换默认值来隔离
    local_history = (tmp_path / "push_history.json",)
    monkeypatch.setattr(notify.load_push_history, "__defaults__", local_history)
    monkeypatch.setattr(notify.save_push_history, "__defaults__", local_history)
    notifier = _RecordingNotifier()
    monkeypatch.setattr(notify, "load_notifiers_from_env", lambda: [notifier])

    notify.push_important_news(_policy_cache(("半导体芯片", "人工智能")))

    assert len(notifier.sent) == 1
    assert notifier.sent[0].count("国务院印发意见") == 1
    assert len(json.loads(_FakeConfigStore.saved[notify.PUSH_HISTORY_KEY])) == 1


# ---- 3. 远端配置读取失败 ----

def _use_config_store(monkeypatch, tmp_path, store: _FakeConfigStore) -> None:
    monkeypatch.setattr(config_store, "_get_config_store", lambda: store)
    monkeypatch.setattr(config_store, "SECTORS_CONFIG_PATH", tmp_path / "sectors_config.json")
    monkeypatch.setattr(config_store, "EVENTS_CONFIG_PATH", tmp_path / "events_config.json")


def test_remote_config_read_failure_is_not_treated_as_missing_config(monkeypatch, tmp_path):
    _use_config_store(monkeypatch, tmp_path, _FakeConfigStore(SupabaseError("HTTP 503", status_code=503)))

    with pytest.raises(config_store.ConfigUnavailableError):
        config_store.load_events_config()
    with pytest.raises(config_store.ConfigUnavailableError):
        config_store.try_load_sectors_config(strict=True)
    # 非严格模式：只读展示可以临时用默认配置，但必须给出提示
    events_config, notice = config_store.try_load_events_config()
    assert events_config == config_store.default_events_config()
    assert notice and "暂时读取失败" in notice


def test_missing_config_table_still_falls_back_to_local_config(monkeypatch, tmp_path):
    _use_config_store(monkeypatch, tmp_path, _FakeConfigStore(SupabaseError("HTTP 404", status_code=404)))

    events_config, notice = config_store.try_load_events_config(strict=True)

    assert notice is None
    assert set(events_config["external_events"]) == set(config_store.default_events_config()["external_events"])


def test_refresh_job_aborts_instead_of_refreshing_with_default_config(monkeypatch, tmp_path, capsys):
    _use_config_store(monkeypatch, tmp_path, _FakeConfigStore(SupabaseError("HTTP 503", status_code=503)))

    def fail(*args, **kwargs):
        raise AssertionError("配置读取失败时不应抓取或写缓存")

    monkeypatch.setattr(refresh_job, "run_incremental_refresh", fail)
    monkeypatch.setattr(refresh_job, "save_cache", fail)

    assert refresh_job.main(["--no-notify"]) == 1
    assert "本次不刷新" in capsys.readouterr().out


# ---- 5. 对比报告的未知板块分支 ----

def test_compare_report_rejects_unknown_sector_with_exit_code(monkeypatch, tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["compare_report", "--trace", str(trace), "--sectors", "不存在的板块",
         "--out-md", str(tmp_path / "o.md"), "--out-json", str(tmp_path / "o.json"),
         "--bad-cases", str(tmp_path / "bad.jsonl")],
    )

    assert compare_report.main() == 2
    assert "未知板块" in capsys.readouterr().err
    assert not (tmp_path / "o.md").exists()
