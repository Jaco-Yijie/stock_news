"""收集待审核差异，读取人工标签，按同一批候选快照离线评测。"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from feedback_store import FEEDBACK_CASES
from paths import DATA_DIR

BAD_CASES_PATH = DATA_DIR / "bad_cases.jsonl"
CONFIRMED_TYPES = {"miss", "false_positive", "kept"}
CASE_TYPES = CONFIRMED_TYPES | {"pending", "dismissed"}


def item_key(item: dict[str, Any]) -> str:
    return str(item.get("link") or item.get("title") or "")


def _case_key(item: dict[str, Any]) -> tuple[str, str]:
    return item["sector"], item_key(item)


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = BAD_CASES_PATH if path is None else Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            rows = []
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or not all(isinstance(row.get(key), str) for key in ("date", "sector", "type"))
                        or not row["date"] or not row["sector"] or not item_key(row)
                        or row["type"] not in CASE_TYPES
                    ):
                        raise ValueError("字段无效")
                except ValueError as exc:
                    raise ValueError(f"{path} 第 {number} 行的 bad case 格式无效") from exc
                rows.append(row)
    except FileNotFoundError:
        return []
    return rows


def load_case_library(
    path: str | Path | None = None, feedback_path: str | Path | None = None
) -> list[dict[str, Any]]:
    local = load_cases(path)
    if feedback_path is None:
        return local
    cases = {(row["date"], *_case_key(row)): row for row in load_cases(feedback_path)}
    for row in local:
        key = (row["date"], *_case_key(row))
        if row["type"] != "pending" or key not in cases:
            cases[key] = row
    return list(cases.values())


def false_positive_examples(
    path: str | Path | None = None, *, before: str, feedback_path: str | Path | None = None
) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for row in sorted(load_case_library(path, feedback_path), key=lambda case: case["date"], reverse=True):
        if row["date"] < before and row["type"] == "false_positive" and row.get("title"):
            titles = examples.setdefault(row["sector"], [])
            if len(titles) < 10 and row["title"] not in titles:
                titles.append(row["title"])
    return examples


def from_compare_report(
    report_json_path: str | Path, case_path: str | Path = BAD_CASES_PATH
) -> int:
    """合并可比且实际经过 Agent 决策的差异，保留既有人工标签。"""
    report = json.loads(Path(report_json_path).read_text(encoding="utf-8"))
    case_path = Path(case_path)
    cases = load_cases(case_path)
    seen = {_case_key(row) for row in cases}
    added = []
    for sector in report["sectors"]:
        if not sector.get("comparable") or sector.get("agent_fallback") or not sector.get("llm_calls"):
            continue
        for direction, field in (("agent_only", "agent_only_items"), ("rules_only", "rules_only_items")):
            for item in sector[field]:
                if not item_key(item):
                    continue
                row = {
                    "date": report["date"], "sector": sector["sector"],
                    "title": item.get("title", ""), "link": item.get("link", ""),
                    "type": "pending", "direction": direction, "note": "",
                }
                key = _case_key(row)
                if key not in seen:
                    seen.add(key)
                    added.append(row)
    if not added and case_path.exists():
        return 0

    case_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=case_path.parent, delete=False
    ) as handle:
        temp = Path(handle.name)
        try:
            for row in [*cases, *added]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.close()
            temp.replace(case_path)
        finally:
            temp.unlink(missing_ok=True)
    return len(added)


def stats(path: str | Path | None = None, *, feedback_path: str | Path | None = None) -> dict[str, Any]:
    rows = load_case_library(path, feedback_path)
    by_sector: dict[str, Counter] = {}
    for row in rows:
        by_sector.setdefault(row["sector"], Counter())[row["type"]] += 1
    return {"total": dict(Counter(row["type"] for row in rows)), "by_sector": by_sector}


def evaluate(
    trace_paths: list[str | Path], case_path: str | Path | None = None, *,
    feedback_path: str | Path | None = None,
) -> dict[str, Any]:
    """对 trace 中出现的人工标注案例核对去留；缺席或规则兜底的案例标为未覆盖。"""
    trace = {}
    for trace_path in trace_paths:
        with Path(trace_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    trace[(record["date"], record["sector"])] = record

    results = []
    counts = {"passed": 0, "failed": 0, "not_covered": 0}
    for case in load_case_library(case_path, feedback_path):
        if case["type"] not in CONFIRMED_TYPES:
            continue
        record = trace.get((case["date"], case["sector"]), {})
        candidates = {item_key(item) for item in record.get("candidate_items", [])}
        selected = {item_key(item) for item in record.get("agent_selected", [])}
        key = item_key(case)
        outcome = "not_covered"
        if key in candidates and record.get("llm_calls") and not record.get("agent_fallback"):
            should_keep = case["type"] in {"miss", "kept"}
            outcome = "passed" if (key in selected) == should_keep else "failed"
        counts[outcome] += 1
        results.append({**case, "outcome": outcome})
    return {"counts": counts, "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="收集、统计和回放评测板块新闻 bad case")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--report", default="compare_report.json")
    summary = commands.add_parser("stats")
    check = commands.add_parser("evaluate")
    check.add_argument("--trace", nargs="+", default=["agent_trace.jsonl"])
    for command in (collect, summary, check):
        command.add_argument("--cases", default=BAD_CASES_PATH, type=Path)
    for command in (summary, check):
        command.add_argument("--feedback-cases", default=FEEDBACK_CASES, type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        result = {"added": from_compare_report(args.report, args.cases)}
    elif args.command == "stats":
        result = stats(args.cases, feedback_path=args.feedback_cases)
    else:
        result = evaluate(args.trace, args.cases, feedback_path=args.feedback_cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
