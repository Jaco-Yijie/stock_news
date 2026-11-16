"""agent.compare_report 的单元测试：diff/load/render 均不触网。"""
from __future__ import annotations

import json

from agent.compare_report import (
    SectorCompare,
    diff_kept,
    load_agent_trace,
    render_markdown,
)


def test_diff_kept_overlap_by_link():
    agent = [
        {"title": "甲", "link": "https://a/1"},
        {"title": "乙", "link": "https://a/2"},
    ]
    rules = [
        {"title": "乙（转载）", "link": "https://a/2"},
        {"title": "丙", "link": "https://a/3"},
    ]
    both, agent_only, rules_only = diff_kept(agent, rules)
    assert both == ["https://a/2"]
    assert agent_only == ["https://a/1"]
    assert rules_only == ["https://a/3"]


def test_diff_kept_falls_back_to_title_when_no_link():
    agent = [{"title": "固态电池量产", "link": ""}]
    rules = [{"title": "固态电池量产", "link": ""}]
    both, agent_only, rules_only = diff_kept(agent, rules)
    assert both == ["固态电池量产"]
    assert agent_only == [] and rules_only == []


def test_diff_kept_dedupes_and_ignores_empty_keys():
    agent = [{"title": "", "link": ""}, {"title": "甲", "link": "x"}, {"title": "甲", "link": "x"}]
    both, _, _ = diff_kept(agent, [{"title": "甲", "link": "x"}])
    assert both == ["x"]


def test_load_agent_trace_filters_by_date(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"date": "2026-11-16", "sector": "半导体芯片", "kept": 2,
         "kept_items": [{"title": "甲", "link": "l1"}]},
        {"date": "2026-11-15", "sector": "半导体芯片", "kept": 9},
        {"date": "2026-11-16", "sector": "人工智能", "kept": 1},
    ]
    trace.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    records = load_agent_trace(str(trace), "2026-11-16")
    assert set(records) == {"半导体芯片", "人工智能"}
    assert records["半导体芯片"]["kept_items"][0]["link"] == "l1"


def test_load_agent_trace_tolerates_blank_lines(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('\n{"date": "2026-11-16", "sector": "人工智能"}\n\n', encoding="utf-8")
    records = load_agent_trace(str(trace), "2026-11-16")
    assert list(records) == ["人工智能"]


def test_render_markdown_contains_table_and_totals():
    row = SectorCompare(
        sector="半导体芯片",
        agent_kept=[{"title": "甲", "link": "l1"}, {"title": "乙", "link": "l2"}],
        rules_kept=[{"title": "乙", "link": "l2"}, {"title": "丙", "link": "l3"}],
        both=["l2"], agent_only=["l1"], rules_only=["l3"],
        tool_calls=5,
    )
    md = render_markdown("2026-11-16", [row])
    assert "| 半导体芯片 | 2 | 2 | 1 | 1 | 1 | 否 | 5 |" in md
    assert "LLM 调用合计：5 次" in md
    assert "仅 Agent 保留" in md and "l1" in md
    assert "仅规则保留" in md and "l3" in md


def test_render_markdown_skips_detail_when_no_diff():
    row = SectorCompare(sector="人工智能", both=["l1"])
    md = render_markdown("2026-11-16", [row])
    assert "差异明细" not in md
