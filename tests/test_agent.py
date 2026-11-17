"""Agent 决策层测试：全部使用 fake LLM 与 monkeypatch 工具层，不触网、不调真模型。"""
from __future__ import annotations

import json

import pytest

import agent.graph as graph
from agent.graph import AgentDecisionError, decide_sector_news, run_agent


class FakeLLM:
    """按队列依次返回预设输出；队列耗尽后返回 final 空结果。"""

    def __init__(self, outputs: list[str | Exception]):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.outputs:
            output = self.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output
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
            state["push"] = args
            return {"ok": True, "error": None}
        if name == "classify_news":
            return {"matched": "天气" not in args.get("title", ""), "reason": "规则", "error": None}
        raise AssertionError(f"unexpected tool {name}")

    monkeypatch.setattr(graph, "call_tool", fake_call)
    return state


def test_run_agent_happy_path(mock_tools):
    llm = FakeLLM([
        '{"focus": "新一代芯片", "tool_strategy": "按需查历史"}',
        '{"action": "final", "kept_ids": ["n0"], "reason": "保留芯片新闻"}',
        '{"remove_ids": [], "reason": "与板块相关"}',
    ])
    report = run_agent(["半导体芯片"], "2026-11-16", llm, push=True)
    s = report.sectors[0]
    assert s.fetched == 2 and s.after_dedupe == 2
    assert len(s.kept) == 1 and s.kept[0]["title"] == "某公司发布新一代AI芯片"
    assert not s.agent_fallback
    assert report.digests["半导体芯片"] == "日报内容"
    assert report.pushed
    assert mock_tools["push"]["feedback_items"] == [{
        "date": "2026-11-16", "sector": "半导体芯片",
        "title": "某公司发布新一代AI芯片", "link": "http://a/1",
    }]
    assert report.trace[0]["sector"] == "半导体芯片"
    assert report.trace[0]["plan"]["focus"] == "新一代芯片"
    assert "按需查历史" in llm.calls[1][1]
    assert report.summary()["llm_calls"] == 3 and report.summary()["tool_calls"] == 0


def test_critic_uses_confirmed_same_sector_cases_and_removes_only_selected_ids(mock_tools, tmp_path):
    cases = tmp_path / "cases.jsonl"
    samples = [
        {"sector": "半导体芯片", "title": "历史广告", "type": "false_positive"},
        {"sector": "半导体芯片", "title": "未审核样例", "type": "pending"},
        {"sector": "人工智能", "title": "其他板块误判", "type": "false_positive"},
    ]
    cases.write_text("\n".join(
        json.dumps({"date": "2026-11-15", **row}, ensure_ascii=False) for row in samples
    ), encoding="utf-8")
    llm = FakeLLM([
        '{"focus": "芯片产业", "tool_strategy": "按需复核"}',
        '{"action": "final", "kept_ids": ["n0", "n1"]}',
        '{"remove_ids": ["n1", "n999", "n1"], "reason": "天气无关"}',
    ])
    report = run_agent(["半导体芯片"], "2026-11-16", llm, bad_cases_path=cases, compare=True)
    trace = report.trace[0]
    assert json.loads(llm.calls[2][1])["past_false_positives"] == ["历史广告"]
    assert trace["critic_removed"] == ["n1"]
    assert trace["critic_reason"] == "天气无关"
    assert report.sectors[0].dropped_ids == ["n1"]
    assert [item["id"] for item in trace["candidate_items"]] == ["n0", "n1"]
    assert trace["agent_selected"] == trace["rules_selected"] == trace["kept_items"]


@pytest.mark.parametrize("failure", ["乱输出", RuntimeError("网络故障"), "{}"])
def test_planner_and_critic_fail_open(mock_tools, failure, caplog):
    llm = FakeLLM([failure, '{"action": "final", "kept_ids": ["n0"]}', failure])
    report = run_agent(["半导体芯片"], "2026-11-16", llm)
    assert not report.sectors[0].agent_fallback
    assert report.trace[0]["plan"] is None
    assert report.trace[0]["critic_removed"] == []
    assert report.trace[0]["kept"] == 1
    assert "Planner 已跳过" in caplog.text and "Critic 已跳过" in caplog.text


def test_executor_request_error_falls_back(mock_tools):
    llm = FakeLLM(['{"focus": "芯片", "tool_strategy": "按需复核"}', TimeoutError()])
    report = run_agent(["半导体芯片"], "2026-11-16", llm, compare=True)
    assert report.trace[0]["agent_fallback"]
    assert report.trace[0]["llm_calls"] == 2
    assert len(report.sectors[0].kept) == 1


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
