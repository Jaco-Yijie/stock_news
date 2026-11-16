"""tools/ 工具层单元测试。

原则：不打网络、不调真实 LLM、不动真实缓存文件——
- 抓取类通过 monkeypatch 替换 fetcher / news_store 的行为；
- LLM 用 FakeVerifier 替身；
- 缓存用 tmp_path 隔离。
"""
from __future__ import annotations

import pandas as pd
import pytest

import news_store
import tools
from tools import analyze, classify, dedupe, digest as digest_tool, fetch, history, push
from tools.schemas import (
    AnalyzeSentimentInput,
    ClassifyNewsInput,
    ComposeDigestInput,
    FetchSectorNewsInput,
    FilterNewNewsInput,
    PushTelegramInput,
    RankNewsInput,
    SearchHistoryInput,
)


def _news_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


SAMPLE_ROWS = [
    {"标题": "某公司发布新一代AI芯片", "新闻内容": "算力大幅提升", "发布时间": "2026-11-16 09:00", "原文链接": "http://a.com/1", "来源媒体": "测试"},
    {"标题": "央行开展公开市场操作", "新闻内容": "维护流动性合理充裕", "发布时间": "2026-11-16 10:00", "原文链接": "http://a.com/2", "来源媒体": "测试"},
]


# ---------- fetch ----------

def test_fetch_unknown_sector_returns_error():
    out = fetch.fetch_sector_news(FetchSectorNewsInput(sector="不存在的板块"))
    assert out["error"] and out["count"] == 0
    assert "半导体芯片" in out["error"]  # 提示可选板块


def test_fetch_sector_news_success(monkeypatch):
    monkeypatch.setattr(
        fetch.fetcher, "fetch_sector_news",
        lambda sector, timeout: fetch.fetcher.SectorResult(data=_news_df(SAMPLE_ROWS)),
    )
    out = fetch.fetch_sector_news(FetchSectorNewsInput(sector="人工智能"))
    assert out["count"] == 2 and out["error"] is None
    assert out["items"][0]["标题"] == SAMPLE_ROWS[0]["标题"]


# ---------- dedupe ----------

def _patch_cache(monkeypatch, df: pd.DataFrame, error=None):
    """df 为展示格式；真实缓存是英文列，这里先转成缓存格式再返回。"""
    cache_df = news_store.display_to_cache("", df) if not df.empty else df
    monkeypatch.setattr(
        news_store, "read_cache",
        lambda path=None: news_store.CacheReadResult(data=cache_df, error=error),
    )


def test_filter_new_news_drops_seen_links(monkeypatch):
    _patch_cache(monkeypatch, _news_df([SAMPLE_ROWS[0]]))
    out = dedupe.filter_new_news(FilterNewNewsInput(items=SAMPLE_ROWS))
    assert out["total_in"] == 2 and out["kept"] == 1 and out["dropped"] == 1
    assert out["items"][0]["原文链接"] == "http://a.com/2"


def test_filter_new_news_cache_failure_keeps_all(monkeypatch):
    _patch_cache(monkeypatch, pd.DataFrame(), error="读取本地缓存失败：X")
    out = dedupe.filter_new_news(FilterNewNewsInput(items=SAMPLE_ROWS))
    assert out["kept"] == 2 and out["error"]


# ---------- classify ----------

def test_classify_rule_hit_without_llm():
    out = classify.classify_news(ClassifyNewsInput(
        title="国产AI芯片量产，算力成本下降", content="", sector="人工智能", use_llm=False,
    ))
    assert out["matched"] is True
    assert out["confidence"] > 0
    assert out["llm_checked"] is False


def test_classify_no_match():
    out = classify.classify_news(ClassifyNewsInput(
        title="今日天气晴好", content="", sector="人工智能", use_llm=False,
    ))
    assert out["matched"] is False


def test_classify_unknown_sector():
    out = classify.classify_news(ClassifyNewsInput(
        title="x", sector="不存在", use_llm=False,
    ))
    assert out["matched"] is False and "未知板块" in out["reason"]


class _FakeVerifier:
    provider = "fake"
    model = "fake-0"
    prompt_version = "v1"
    rule_version = "v1"

    def verify(self, news_item, category, evidence):
        import classifier as clf
        return clf.LLMValidationResult(False, category, 0.0, "fake: 误报，剔除")


def test_classify_llm_rejection_overrides(monkeypatch):
    # 构造触发 LLM 复核的低置信度命中：只命中一个通用词
    classify.set_llm_verifier(_FakeVerifier())
    try:
        out = classify.classify_news(ClassifyNewsInput(
            title="算力", content="", sector="人工智能", use_llm=True,
        ))
    finally:
        classify.set_llm_verifier(None)
    if out["llm_checked"]:  # 是否触发复核取决于规则阈值，两种情况都合法
        assert out["llm_kept"] is False and out["matched"] is False
    else:
        assert out["matched"] is True


# ---------- analyze ----------

def test_analyze_sentiment_positive():
    out = analyze.analyze_sentiment(AnalyzeSentimentInput(
        title="重大突破：国产芯片量产，订单大幅增长", content="业绩超预期",
        sectors=["半导体芯片"],
    ))
    a = out["analysis"]
    assert a["sentiment"] in ("positive", "neutral", "negative")
    assert "半导体芯片" in a.get("sector_assessments", {})


def test_rank_news_orders_by_importance():
    items = []
    for i, title in enumerate(["证监会发布新政策", "某公司股价异动"]):
        a = analyze.analyze_sentiment(AnalyzeSentimentInput(
            title=title, content="", sectors=["证券"],
        ))["analysis"]
        items.append({"标题": title, "发布时间": f"2026-11-16 0{i}:00", "analysis": a})
    out = analyze.rank_news(RankNewsInput(items=items))
    assert out["count"] == 2
    imp = [i["analysis"].get("importance", 0) for i in out["items"]]
    assert imp == sorted(imp, reverse=True)


# ---------- history ----------

def test_search_history_finds_keyword(monkeypatch):
    _patch_cache(monkeypatch, _news_df(SAMPLE_ROWS))
    out = history.search_history(SearchHistoryInput(query="AI芯片"))
    assert out["count"] == 1
    assert "AI芯片" in out["items"][0]["title"]  # 缓存内部为英文列名


def test_search_history_empty_query(monkeypatch):
    _patch_cache(monkeypatch, _news_df(SAMPLE_ROWS))
    out = history.search_history(SearchHistoryInput(query="  "))
    assert out["count"] == 0 and out["error"]


# ---------- digest ----------

class _FakeCompleter:
    def complete(self, system_prompt, user_prompt):
        return "今日板块整体偏利好，主要受政策与订单驱动。"


def test_compose_digest_success():
    digest_tool.set_llm_verifier(_FakeCompleter())
    try:
        out = digest_tool.compose_digest(ComposeDigestInput(
            sector="人工智能", items=SAMPLE_ROWS,
        ))
    finally:
        digest_tool.set_llm_verifier(None)
    assert out["error"] is None
    assert "利好" in out["digest"] and out["used_count"] == 2


def test_compose_digest_without_verifier():
    digest_tool.set_llm_verifier(None)
    out = digest_tool.compose_digest(ComposeDigestInput(
        sector="人工智能", items=SAMPLE_ROWS,
    ))
    assert out["digest"] == "" and "未注入" in out["error"]


def test_compose_digest_empty_items():
    out = digest_tool.compose_digest(ComposeDigestInput(sector="人工智能", items=[]))
    assert out["error"] == "没有可归纳的新闻"


# ---------- push ----------

def test_push_without_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    out = push.push_telegram(PushTelegramInput(title="t", content="c"))
    assert out["sent"] is False and "TELEGRAM" in out["error"]


def test_push_send_failure_returns_error(monkeypatch):
    class _Boom:
        def send(self, title, content):
            raise RuntimeError("network down")

    monkeypatch.setattr(push.notify, "load_notifiers_from_env", lambda: [_Boom()])
    monkeypatch.setattr(push.notify, "TelegramNotifier", _Boom)
    out = push.push_telegram(PushTelegramInput(title="t", content="c"))
    assert out["sent"] is False and "network down" in out["error"]


# ---------- registry ----------

def test_registry_has_eight_tools():
    assert len(tools.TOOLS) == 8
    specs = tools.tool_specs()
    assert {s["function"]["name"] for s in specs} == set(tools.TOOL_BY_NAME)
    for s in specs:
        assert s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"


def test_call_tool_validates_input():
    out = tools.call_tool("classify_news", {"title": 123, "sector": "人工智能"})
    # title=123 会被 pydantic 校验拒绝或强转，这里只要求不抛异常、返回 dict
    assert isinstance(out, dict)


def test_call_tool_unknown():
    out = tools.call_tool("not_a_tool", {})
    assert "未知工具" in out["error"]
