"""新旧管线对比报告：python -m agent.compare_report --trace agent_trace.jsonl [--date 2026-11-16]

读取 --compare 生成的同批候选 trace，比较两侧排序截断前的分类结果。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.bad_case import BAD_CASES_PATH, from_compare_report, item_key
from sectors import SECTORS
from time_utils import now_utc8_naive


@dataclass
class SectorCompare:
    sector: str
    agent_kept: list[dict[str, str]] = field(default_factory=list)
    rules_kept: list[dict[str, str]] = field(default_factory=list)
    both: list[str] = field(default_factory=list)
    agent_only: list[str] = field(default_factory=list)
    rules_only: list[str] = field(default_factory=list)
    agent_fallback: bool = False
    tool_calls: int = 0
    llm_calls: int | None = None
    unavailable: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "agent_kept": len(self.agent_kept),
            "rules_kept": len(self.rules_kept),
            "overlap": len(self.both),
            "agent_only": len(self.agent_only),
            "rules_only": len(self.rules_only),
            "agent_fallback": self.agent_fallback,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "comparable": not self.unavailable,
            "unavailable": self.unavailable,
            "agent_only_items": _items_for_keys(self.agent_kept, self.agent_only),
            "rules_only_items": _items_for_keys(self.rules_kept, self.rules_only),
        }


def _items_for_keys(items: list[dict[str, str]], keys: list[str]) -> list[dict[str, str]]:
    by_key = {item_key(item): item for item in items}
    return [by_key[key] for key in keys]


def diff_kept(
    agent_items: list[dict[str, str]],
    rules_items: list[dict[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """返回 (交集, 仅 Agent 保留, 仅规则保留)，按对比键去重。纯函数，便于单测。"""
    agent_keys = set(map(item_key, agent_items)) - {""}
    rules_keys = set(map(item_key, rules_items)) - {""}
    both = sorted(agent_keys & rules_keys)
    agent_only = sorted(agent_keys - rules_keys)
    rules_only = sorted(rules_keys - agent_keys)
    return both, agent_only, rules_only


def load_agent_trace(trace_path: str, date: str) -> dict[str, dict[str, Any]]:
    """读取指定日期的 trace 记录，按板块索引；无 kept_items 的旧记录按空列表处理。"""
    records: dict[str, dict[str, Any]] = {}
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("date") != date:
                continue
            records[record["sector"]] = record
    return records


def compare(date: str, trace_path: str, sectors: list[str]) -> list[SectorCompare]:
    trace = load_agent_trace(trace_path, date)
    results: list[SectorCompare] = []
    for sector in sectors:
        row = SectorCompare(sector=sector)
        record = trace.get(sector, {})
        row.agent_kept = record.get("agent_selected", record.get("kept_items", []))
        row.agent_fallback = bool(record.get("agent_fallback", False))
        row.tool_calls = len(record.get("tool_calls", []))
        row.llm_calls = record.get("llm_calls")
        if "candidate_items" not in record:
            row.unavailable = record.get("snapshot_error", "缺少同批候选快照")
        elif record.get("rules_selected") is None:
            row.unavailable = "规则侧分类失败"
        else:
            row.rules_kept = record["rules_selected"]
            row.both, row.agent_only, row.rules_only = diff_kept(row.agent_kept, row.rules_kept)
        results.append(row)
    return results


def render_markdown(date: str, rows: list[SectorCompare]) -> str:
    lines = [
        f"# 新旧管线对比（{date}）",
        "",
        "两侧使用同一批候选，比较分类与复核后的去留，统计点在排序取前 10 条之前。",
        "",
        "| 板块 | Agent 保留 | 规则保留 | 交集 | 仅 Agent | 仅规则 | 兜底 | 工具调用 | 决策模型调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :-: | ---: | ---: |",
    ]
    for row in rows:
        s = row.summary()
        if row.unavailable:
            lines.append(f"| {row.sector}（{row.unavailable}） | — | — | — | — | — | — | — | — |")
            continue
        llm_calls = row.llm_calls if row.llm_calls is not None else "—"
        lines.append(
            f"| {s['sector']} | {s['agent_kept']} | {s['rules_kept']} | {s['overlap']} "
            f"| {s['agent_only']} | {s['rules_only']} "
            f"| {'是' if s['agent_fallback'] else '否'} | {s['tool_calls']} | {llm_calls} |"
        )
    measured = [row.llm_calls for row in rows if row.llm_calls is not None]
    total = f"{sum(measured)} 次" if measured else "无记录"
    lines += ["", f"决策模型调用合计：{total}（含 Planner、ReAct、Critic 的调用尝试）。", ""]
    for row in rows:
        if not row.agent_only and not row.rules_only:
            continue
        lines.append(f"## {row.sector} 差异明细")
        if row.agent_only:
            lines.append("仅 Agent 保留：")
            lines += [f"- {t}" for t in row.agent_only]
        if row.rules_only:
            lines.append("仅规则保留：")
            lines += [f"- {t}" for t in row.rules_only]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 管线 vs 规则管线对比报告")
    parser.add_argument("--trace", default="agent_trace.jsonl", help="run_agent 落盘的 trace 文件")
    parser.add_argument("--date", default=now_utc8_naive().date().isoformat())
    parser.add_argument("--sectors", nargs="+", default=None, help="默认取 trace 中当日出现的板块")
    parser.add_argument("--out-md", default="compare_report.md")
    parser.add_argument("--out-json", default="compare_report.json")
    parser.add_argument("--bad-cases", type=Path, default=BAD_CASES_PATH,
                        help="待审核差异与人工标签的 JSONL 文件")
    args = parser.parse_args()

    trace = load_agent_trace(args.trace, args.date)
    sectors = args.sectors or sorted(trace) or sorted(SECTORS)
    invalid = [s for s in sectors if s not in SECTORS]
    if invalid:
        print(f"未知板块：{invalid}", file=sys.stderr)
        return 2

    rows = compare(args.date, args.trace, sectors)
    md = render_markdown(args.date, rows)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"date": args.date, "sectors": [r.summary() for r in rows]},
            fh, ensure_ascii=False, indent=2,
        )
    print(md)
    added = from_compare_report(args.out_json, args.bad_cases)
    print(f"新增待审核案例：{added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
