"""Agent 决策层测试：全部使用 fake LLM 与 monkeypatch 工具层，不触网、不调真模型。"""
from __future__ import annotations

import json

import pytest

import agent.graph as graph
from agent.graph import AgentDecisionError, decide_sector_news, run_agent


class FakeLLM:
    """按队列依次返回预设输出；队列耗尽后返回 final 空结果。"""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.outputs:
            return self.outputs.pop(0)
        return json.dumps({"action": "final", "kept_ids": [], "reason": "队列耗尽"})


CANDIDATES = [
    {"id": "n0", "title": "某公司发布新一代AI芯片", "summary": "算力大幅提升"},
    {"id": "n1", "title": "今日天气晴朗", "summary": "与板块无关"},
]


# ---------- _parse_agent_action ----------

@pytest.mark.parametrize("text", [
    '{"action": "final", "kept_ids": ["n0"], "reason": "x"}',
    '```json\n{"action": "final", "kept_ids": ["n0"], "reason": "x"}\n```',
    '好的，结论如下：{"action": "final", "kept_ids": ["n0"], "reason": "x"} 以上。',
])
def test_parse_action_accepts_fenced_and_noisy(text):
    action = graph._parse_agent_action(text)
    assert action is not None and action["action"] == "final"


@pytest.mark.parametrize("text", ["", "没有JSON", "[]", '{"no_action": 1}'])
def test_parse_action_rejects_garbage(text):
    assert graph._parse_agent_action(text) is None


# ---------- decide_sector_news ----------

def test_decide_direct_final():
    llm = FakeLLM(['{"action": "final", "kept_ids": ["n0", "n999"], "reason": "保留芯片"}'])
    kept, reason, records = decide_sector_news("半导体芯片", CANDIDATES, llm)
    assert kept == ["n0"]  # n999 不在候选中，被过滤
    assert "芯片" in reason and records == []


def test_decide_calls_whitelisted_tool_then_final(monkeypatch):
    calls = []
    monkeypatch.setattr(
        graph, "call_tool",
        lambda name, args: calls.append((name, args)) or {"matched": True, "reason": "ok"},
    )
    llm = FakeLLM([
        '{"action": "tool", "tool": "classify_news", "args": {"title": "今日天气晴朗", "sector": "半导体芯片"}}',
        '{"action": "final", "kept_ids": ["n0"], "reason": "n1复核不相关"}',
    ])
    kept, _, records = decide_sector_news("半导体芯片", CANDIDATES, llm)
    assert kept == ["n0"]
    assert calls and calls[0][0] == "classify_news"
    assert len(records) == 1 and records[0].ok and records[0].latency_ms >= 0


def test_decide_rejects_non_whitelisted_tool(monkeypatch):
    monkeypatch.setattr(graph, "call_tool", lambda n, a: {})
    llm = FakeLLM([
        '{"action": "tool", "tool": "push_telegram", "args": {}}',
        '{"action": "final", "kept_ids": [], "reason": "放弃"}',
    ])
    kept, _, records = decide_sector_news("半导体芯片", CANDIDATES, llm)
    assert kept == [] and records == []  # 白名单外工具不执行


def test_decide_budget_exhausted_raises(monkeypatch):
    monkeypatch.setattr(graph, "call_tool", lambda n, a: {"ok": True})
    # 每次都要求调工具，预算 2 次用完后仍不给 final -> 报错
    looping = ['{"action": "tool", "tool": "search_history", "args": {"query": "x"}}'] * 10
    llm = FakeLLM(looping)
    with pytest.raises(AgentDecisionError):
        decide_sector_news("半导体芯片", CANDIDATES, llm, max_tool_calls=2)


def test_decide_unparseable_retries_then_final():
    llm = FakeLLM([
        "让我想想……",
        '{"action": "final", "kept_ids": ["n1"], "reason": "都行"}',
    ])
    kept, _, _ = decide_sector_news("半导体芯片", CANDIDATES, llm)
    assert kept == ["n1"]


def test_decide_empty_candidates_short_circuits():
    llm = FakeLLM([])
    kept, reason, records = decide_sector_news("半导体芯片", [], llm)
    assert kept == [] and records == [] and llm.calls == []


# ---------- run_agent 全流程（工具层全部 mock）----------

@pytest.fixture
def mock_tools(monkeypatch):
    state = {"cache_seen": False}

    def fake_call(name, args):
        if name == "fetch_sector_news":
            return {"sector": args["sector"], "count": 2, "warnings": [], "error": None,
                    "items": [
                        {"标题": "某公司发布新一代AI芯片", "新闻内容": "算力大幅提升",
                         "发布时间": "2026-11-16 09:00", "原文链接": "http://a/1"},
                        {"标题": "今日天气晴朗", "新闻内容": "无关",
                         "发布时间": "2026-11-16 09:10", "原文链接": "http://a/2"},
                    ]}
        if name == "filter_new_news":
            return {"total_in": 2, "kept": 2, "dropped": 0, "items": args["items"], "error": None}
        if name == "analyze_sentiment":
            return {"analysis": {"overall_sentiment": "positive", "importance": 3}, "error": None}
        if name == "rank_news":
            return {"count": len(args["items"]), "items": args["items"]}
        if name == "compose_digest":
            return {"sector": args["sector"], "digest": "日报内容", "used_count": len(args["items"]), "error": None}
        if name == "push_telegram":
            return {"ok": True, "error": None}
        if name == "classify_news":
            return {"matched": "天气" not in args.get("title", ""), "reason": "规则", "error": None}
        raise AssertionError(f"unexpected tool {name}")

    monkeypatch.setattr(graph, "call_tool", fake_call)
    return state


def test_run_agent_happy_path(mock_tools):
    llm = FakeLLM(['{"action": "final", "kept_ids": ["n0"], "reason": "保留芯片新闻"}'])
    report = run_agent(["半导体芯片"], "2026-11-16", llm, push=True)
    s = report.sectors[0]
    assert s.fetched == 2 and s.after_dedupe == 2
    assert len(s.kept) == 1 and s.kept[0]["title"] == "某公司发布新一代AI芯片"
    assert not s.agent_fallback
    assert report.digests["半导体芯片"] == "日报内容"
    assert report.pushed
    assert report.trace[0]["sector"] == "半导体芯片"


def test_run_agent_fallback_when_llm_none(mock_tools):
    report = run_agent(["半导体芯片"], "2026-11-16", llm=None)
    s = report.sectors[0]
    assert s.agent_fallback
    assert len(s.kept) == 1  # 规则兜底剔除了「天气」
    assert report.digests  # 兜底路径也照常出日报


def test_run_agent_fallback_when_decision_fails(mock_tools):
    llm = FakeLLM(["乱输出"] * 30)
    report = run_agent(["半导体芯片"], "2026-11-16", llm)
    s = report.sectors[0]
    assert s.agent_fallback and "规则兜底" in s.agent_reason


def test_run_agent_fetch_error_continues(monkeypatch):
    def fake_call(name, args):
        if name == "fetch_sector_news":
            return {"sector": args["sector"], "count": 0, "items": [], "warnings": [],
                    "error": "网络超时"}
        return {"items": [], "error": None}
    monkeypatch.setattr(graph, "call_tool", fake_call)
    report = run_agent(["半导体芯片"], "2026-11-16", FakeLLM([]))
    assert "网络超时" in (report.sectors[0].error or "")
    assert not report.digests
