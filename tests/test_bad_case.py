"""W3 差异收集、人工标签和离线评测的回归测试。"""
from __future__ import annotations

import json

import pytest

from agent.bad_case import evaluate, false_positive_examples, from_compare_report, load_cases, stats

DATE = "2026-11-16"
KEPT = {"title": "广告", "link": "https://example.com/ad"}
REMOVED = {"title": "行业政策", "link": "https://example.com/policy"}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def sector_report(sector="人工智能", **overrides):
    return {
        "sector": sector, "comparable": True, "agent_fallback": False, "llm_calls": 3,
        "agent_only_items": [KEPT], "rules_only_items": [REMOVED], **overrides,
    }


def test_collect_keeps_directions_pending_and_preserves_labels_on_retry(tmp_path):
    report = tmp_path / "report.json"
    cases = tmp_path / "bad_cases.jsonl"
    report.write_text(json.dumps({"date": DATE, "sectors": [
        sector_report(), sector_report(), sector_report("半导体芯片"),
    ]}), encoding="utf-8")

    assert from_compare_report(report, cases) == 4
    rows = load_cases(cases)
    assert [row["direction"] for row in rows[:2]] == ["agent_only", "rules_only"]
    assert {row["type"] for row in rows} == {"pending"}
    rows[0].update(type="false_positive", note="人工确认广告")
    write_jsonl(cases, rows)

    assert from_compare_report(report, cases) == 0
    assert load_cases(cases)[0]["note"] == "人工确认广告"
    assert stats(cases)["by_sector"]["人工智能"] == {"false_positive": 1, "pending": 1}


@pytest.mark.parametrize("unusable", [
    {"comparable": False}, {"agent_fallback": True}, {"llm_calls": 0},
])
def test_collect_skips_unusable_results(tmp_path, unusable):
    report = tmp_path / "report.json"
    cases = tmp_path / "bad_cases.jsonl"
    report.write_text(json.dumps({"date": DATE, "sectors": [sector_report(**unusable)]}),
                      encoding="utf-8")
    assert from_compare_report(report, cases) == 0
    assert cases.read_text(encoding="utf-8") == ""
    assert load_cases(cases) == []


def test_examples_use_only_prior_confirmed_false_positives_and_cap_per_sector(tmp_path):
    cases = tmp_path / "bad_cases.jsonl"
    history = [
        {"date": DATE, "sector": "人工智能", "title": f"误判{i}", "type": "false_positive"}
        for i in range(15)
    ]
    write_jsonl(cases, [
        *history,
        {**history[0], "title": "同日样例", "date": "2026-11-17"},
        {**history[0], "title": "待审核", "type": "pending"},
    ])
    assert false_positive_examples(cases, before="2026-11-17") == {
        "人工智能": [f"误判{i}" for i in range(10)],
    }


def test_evaluate_checks_confirmed_labels_against_same_day_candidate_snapshot(tmp_path):
    cases = tmp_path / "bad_cases.jsonl"
    first_trace, second_trace = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    base = {"date": DATE, "sector": "人工智能"}
    write_jsonl(cases, [
        {**base, **KEPT, "type": "false_positive"},
        {**base, **REMOVED, "type": "miss"},
        {**base, **KEPT, "sector": "半导体芯片", "type": "false_positive"},
        {**base, "title": "未出现的样例", "type": "miss"},
        {**base, "title": "未审核的样例", "type": "pending"},
    ])
    write_jsonl(first_trace, [{
        **base, "candidate_items": [KEPT, REMOVED], "agent_selected": [REMOVED], "llm_calls": 3,
    }])
    write_jsonl(second_trace, [{
        **base, "sector": "半导体芯片", "candidate_items": [KEPT], "agent_selected": [],
        "agent_fallback": True, "llm_calls": 2,
    }])
    assert evaluate([first_trace, second_trace], cases)["counts"] == {
        "passed": 2, "failed": 0, "not_covered": 2,
    }
    write_jsonl(first_trace, [{
        **base, "candidate_items": [KEPT, REMOVED], "agent_selected": [KEPT], "llm_calls": 3,
    }])
    assert evaluate([first_trace, second_trace], cases)["counts"]["failed"] == 2


def test_missing_case_file_is_empty_but_invalid_line_is_reported(tmp_path):
    cases = tmp_path / "bad_cases.jsonl"
    assert stats(cases) == {"total": {}, "by_sector": {}}
    cases.write_text('\n{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_cases(cases)
