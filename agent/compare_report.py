"""新旧管线对比报告：python -m agent.compare_report --trace agent_trace.jsonl [--date 2026-11-16]

对同一天、同一批板块：
- Agent 侧：读取 run_agent 落盘的 trace（kept_items 来自决策结果）；
- 规则侧：重新抓取同板块新闻，走纯规则分类（use_llm=False），得到规则保留集；
- 输出交集/差集统计与 LLM 调用量，写 Markdown + JSON，供并行期每天对比。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from sectors import SECTORS
from tools.classify import classify_news
from tools.dedupe import filter_new_news
from tools.fetch import fetch_sector_news
from tools.schemas import (
    ClassifyNewsInput,
    FetchSectorNewsInput,
    FilterNewNewsInput,
)


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
        }


def _key(item: dict[str, str]) -> str:
    """对比键：优先链接，无链接退回标题。"""
    return str(item.get("link") or item.get("title") or "")


def diff_kept(
    agent_items: list[dict[str, str]],
    rules_items: list[dict[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """返回 (交集, 仅 Agent 保留, 仅规则保留)，按对比键去重。纯函数，便于单测。"""
    agent_map = {_key(i): i for i in agent_items if _key(i)}
    rules_map = {_key(i): i for i in rules_items if _key(i)}
    both = sorted(set(agent_map) & set(rules_map))
    agent_only = sorted(set(agent_map) - set(rules_map))
    rules_only = sorted(set(rules_map) - set(agent_map))
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


def rules_side_kept(sector: str) -> tuple[list[dict[str, str]], str | None]:
    """规则管线基线：抓取 → 增量过滤 → 纯规则分类，返回 (kept 列表, 错误信息)。"""
    fetched = fetch_sector_news(FetchSectorNewsInput(sector=sector))
    if fetched.get("error"):
        return [], fetched["error"]
    fresh = filter_new_news(FilterNewNewsInput(items=fetched.get("items", [])))
    kept: list[dict[str, str]] = []
    for item in fresh.get("items", []):
        result = classify_news(ClassifyNewsInput(
            sector=sector,
            title=str(item.get("标题", "")),
            content=str(item.get("新闻内容", "")),
            use_llm=False,
        ))
        if result.get("matched"):
            kept.append({
                "title": str(item.get("标题", "")),
                "link": str(item.get("原文链接", "")),
            })
    return kept, None


def compare(date: str, trace_path: str, sectors: list[str]) -> list[SectorCompare]:
    trace = load_agent_trace(trace_path, date)
    results: list[SectorCompare] = []
    for sector in sectors:
        row = SectorCompare(sector=sector)
        record = trace.get(sector, {})
        row.agent_kept = list(record.get("kept_items", []))
        row.agent_fallback = bool(record.get("agent_fallback", False))
        row.tool_calls = len(record.get("tool_calls", []))
        rules_kept, error = rules_side_kept(sector)
        if error:
            print(f"[warn] 规则侧「{sector}」抓取失败：{error}", file=sys.stderr)
        row.rules_kept = rules_kept
        row.both, row.agent_only, row.rules_only = diff_kept(row.agent_kept, row.rules_kept)
        results.append(row)
    return results


def render_markdown(date: str, rows: list[SectorCompare]) -> str:
    lines = [
        f"# 新旧管线对比（{date}）",
        "",
        "| 板块 | Agent 保留 | 规则保留 | 交集 | 仅 Agent | 仅规则 | 兜底 | LLM 调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :-: | ---: |",
    ]
    total_calls = 0
    for row in rows:
        s = row.summary()
        total_calls += s["tool_calls"]
        lines.append(
            f"| {s['sector']} | {s['agent_kept']} | {s['rules_kept']} | {s['overlap']} "
            f"| {s['agent_only']} | {s['rules_only']} "
            f"| {'是' if s['agent_fallback'] else '否'} | {s['tool_calls']} |"
        )
    lines += ["", f"LLM 调用合计：{total_calls} 次", ""]
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
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--sectors", nargs="+", default=None, help="默认取 trace 中当日出现的板块")
    parser.add_argument("--out-md", default="compare_report.md")
    parser.add_argument("--out-json", default="compare_report.json")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
