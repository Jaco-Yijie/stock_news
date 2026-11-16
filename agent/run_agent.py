"""Agent 日报入口：python -m agent.run_agent --sectors 半导体芯片 人工智能 [--push] [--dry-run]

与现有 daily-report workflow 并存：W2 期间新旧并行，便于对比效果。
环境变量沿用 llm_provider 的约定（LLM_VERIFY_PROVIDER / 各供应商 KEY / ENDPOINT / MODEL），
推送沿用 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from agent.graph import run_agent
from llm_provider import load_llm_verifier_from_env
from sectors import SECTORS
from tools.digest import set_llm_verifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 驱动的板块新闻日报")
    parser.add_argument("--sectors", nargs="+", default=None,
                        help="板块名列表，默认取 SECTORS 前 3 个（试运行）")
    parser.add_argument("--all", action="store_true", help="跑全部板块（并行对比用）")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--push", action="store_true", help="推送到 Telegram")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印决策与日报，不推送（默认即如此，显式标记便于 CI 阅读）")
    parser.add_argument("--trace-out", default="agent_trace.jsonl",
                        help="决策 trace 输出文件（JSONL，每行一个板块）")
    args = parser.parse_args()

    sectors = sorted(SECTORS) if args.all else (args.sectors or sorted(SECTORS)[:3])
    invalid = [s for s in sectors if s not in SECTORS]
    if invalid:
        print(f"未知板块：{invalid}，可选见 sectors.SECTORS", file=sys.stderr)
        return 2

    llm, reason = load_llm_verifier_from_env()
    if llm is None:
        print(f"[warn] {reason} 将走规则兜底。", file=sys.stderr)
    else:
        set_llm_verifier(llm)

    report = run_agent(sectors, date=args.date, llm=llm, push=args.push and not args.dry_run)

    with open(args.trace_out, "a", encoding="utf-8") as fh:
        for record in report.trace:
            fh.write(json.dumps({"date": args.date, **record}, ensure_ascii=False) + "\n")

    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    for sector, text in report.digests.items():
        print(f"\n===== {sector} =====\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
