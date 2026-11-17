"""编排板块新闻的 Planner、ReAct 决策、Critic 与日报工具。"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Protocol

from agent import prompts
from tools import call_tool, tool_specs

logger = logging.getLogger(__name__)


class ChatLLM(Protocol):
    """决策层对 LLM 的最小依赖：纯文本进、纯文本出。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


# 决策循环里允许 Agent 调用的只读工具白名单（写操作不在其中）
EXECUTOR_ALLOWED_TOOLS = ("classify_news", "search_history")

DEFAULT_MAX_TOOL_CALLS_PER_SECTOR = 12


@dataclass
class DecisionLLM:
    llm: ChatLLM
    calls: int = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.llm.complete(system_prompt, user_prompt)


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    ok: bool
    latency_ms: int


@dataclass
class SectorResult:
    sector: str
    fetched: int = 0
    after_dedupe: int = 0
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    agent_reason: str = ""
    agent_fallback: bool = False  # Agent 决策失败时是否走了规则兜底
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    error: str | None = None


@dataclass
class AgentReport:
    date: str
    sectors: list[SectorResult] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)
    pushed: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "sectors": len(self.sectors),
            "fetched": sum(s.fetched for s in self.sectors),
            "kept": sum(len(s.kept) for s in self.sectors),
            "fallback_sectors": sum(1 for s in self.sectors if s.agent_fallback),
            "tool_calls": sum(len(s.tool_calls) for s in self.sectors),
            "llm_calls": sum(s.llm_calls for s in self.sectors),
            "digests": len(self.digests),
            "pushed": self.pushed,
        }


def _parse_model_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象，容忍代码围栏和首尾噪声。"""
    if not text:
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_agent_action(text: str) -> dict[str, Any] | None:
    data = _parse_model_object(text)
    return data if data is not None and "action" in data else None


def _plan_sector(
    sector: str, candidates: list[dict[str, Any]], llm: ChatLLM
) -> dict[str, str] | None:
    payload = {"sector": sector, "titles": [item["title"] for item in candidates[:5]]}
    try:
        plan = _parse_model_object(
            llm.complete(prompts.PLANNER_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        )
        if plan is None or not all(
            isinstance(plan.get(key), str) and plan[key].strip()
            for key in ("focus", "tool_strategy")
        ):
            raise ValueError("无效的计划")
    except Exception as exc:
        logger.warning("板块「%s」Planner 已跳过（%s）", sector, type(exc).__name__)
        return None
    return {key: plan[key].strip()[:500] for key in ("focus", "tool_strategy")}


def _review_selection(
    sector: str, kept: list[dict[str, Any]], examples: list[str], llm: ChatLLM
) -> tuple[list[str], str]:
    payload = {
        "sector": sector,
        "selected": [{"id": item["id"], "title": item["title"]} for item in kept],
        "past_false_positives": examples,
    }
    try:
        review = _parse_model_object(
            llm.complete(prompts.CRITIC_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        )
        if review is None or not isinstance(review.get("reason"), str):
            raise ValueError("无效的复核结果")
        remove_ids = review.get("remove_ids")
        if not isinstance(remove_ids, list) or not all(isinstance(item, str) for item in remove_ids):
            raise ValueError("无效的复核编号")
    except Exception as exc:
        logger.warning("板块「%s」Critic 已跳过（%s）", sector, type(exc).__name__)
        return [], ""
    removed = set(remove_ids)
    return [item["id"] for item in kept if item["id"] in removed], review["reason"][:500]


def _tool_docs() -> str:
    lines = []
    for spec in tool_specs():
        fn = spec["function"]
        if fn["name"] in EXECUTOR_ALLOWED_TOOLS:
            lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


def decide_sector_news(
    sector: str,
    candidates: list[dict[str, Any]],
    llm: ChatLLM,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_SECTOR,
    *,
    plan: dict[str, str] | None = None,
) -> tuple[list[str], str, list[ToolCallRecord]]:
    """Agent 决策循环：返回 (保留的id列表, 决策理由, 工具调用记录)。

    协议：模型每轮输出一个 JSON 动作（tool 或 final）。
    兜底：超出预算或输出无法解析时抛 AgentDecisionError，由调用方走规则兜底。
    """
    if not candidates:
        return [], "无候选新闻", []

    tool_calls: list[ToolCallRecord] = []
    observations: list[str] = []
    system = prompts.executor_system_prompt(
        tool_docs=_tool_docs(), budget=max_tool_calls
    )

    base_user = prompts.executor_user_prompt(sector, candidates)
    if plan is not None:
        base_user += "\n\nPlanner 的参考建议（仍须遵守分类规则和工具预算）：\n"
        base_user += json.dumps(plan, ensure_ascii=False)
    for _round in range(max_tool_calls + 4):  # 预留几轮给 final 输出
        user = base_user
        if observations:
            user += "\n\n已获得的工具结果：\n" + "\n".join(observations)

        try:
            action = _parse_agent_action(llm.complete(system, user))
        except Exception as exc:
            raise AgentDecisionError(f"模型调用失败（{type(exc).__name__}）") from exc
        if action is None:
            continue  # 输出无法解析，原样重试（temperature=0 时最多空转，受轮数上限保护）

        if action["action"] == "final":
            kept = action.get("kept_ids")
            if not isinstance(kept, list) or not all(isinstance(i, str) for i in kept):
                continue
            valid = {c["id"] for c in candidates}
            return [i for i in kept if i in valid], str(action.get("reason", "")), tool_calls

        if action["action"] == "tool":
            if len(tool_calls) >= max_tool_calls:
                observations.append("（工具预算已用尽，请立即输出 final 动作）")
                continue
            name = str(action.get("tool", ""))
            args = action.get("args") or {}
            if name not in EXECUTOR_ALLOWED_TOOLS or not isinstance(args, dict):
                observations.append(f"（工具 {name!r} 不在白名单或参数非法，请重新选择）")
                continue
            started = time.monotonic()
            result = call_tool(name, args)
            latency_ms = int((time.monotonic() - started) * 1000)
            tool_calls.append(
                ToolCallRecord(tool=name, args=args, ok=not result.get("error"), latency_ms=latency_ms)
            )
            observations.append(f"[{name}] {json.dumps(result, ensure_ascii=False)[:800]}")

    raise AgentDecisionError(f"板块「{sector}」Agent 未能在预算内给出 final 动作")


class AgentDecisionError(RuntimeError):
    pass


def run_agent(
    sectors: list[str],
    date: str,
    llm: ChatLLM | None,
    push: bool = False,
    max_tool_calls_per_sector: int = DEFAULT_MAX_TOOL_CALLS_PER_SECTOR,
    *,
    bad_cases_path: str | Path | None = None,
    feedback_cases_path: str | Path | None = None,
    compare: bool = False,
) -> AgentReport:
    """先复核分类决策，再排序生成日报；仅对比运行记录用于评测的候选快照。"""
    report = AgentReport(date=date)
    examples: dict[str, list[str]] = {}
    if llm is not None:
        from agent.bad_case import false_positive_examples

        try:
            examples = false_positive_examples(bad_cases_path, before=date, feedback_path=feedback_cases_path)
        except (OSError, ValueError) as exc:
            logger.warning("无法读取 bad case，Critic 将不使用历史样例（%s）", type(exc).__name__)

    for sector in sectors:
        result = SectorResult(sector=sector)

        fetched = call_tool("fetch_sector_news", {"sector": sector})
        if fetched.get("error"):
            result.error = f"fetch: {fetched['error']}"
            report.sectors.append(result)
            if compare:
                report.trace.append({"sector": sector, "snapshot_error": "抓取失败"})
            continue
        items = fetched.get("items", [])
        result.fetched = len(items)

        fresh = call_tool("filter_new_news", {"items": items})
        if fresh.get("error"):
            result.error = f"filter_new_news: {fresh['error']}"
            logger.warning("板块「%s」增量过滤失败，继续使用返回的候选新闻", sector)
        candidates = fresh.get("items", [])
        result.after_dedupe = len(candidates)
        candidates = [
            {**item, "id": f"n{idx}", "title": item.get("标题") or "",
             "summary": item.get("新闻内容") or ""}
            for idx, item in enumerate(candidates)
        ]

        plan = None
        critic_removed: list[str] = []
        critic_reason = ""
        if candidates and llm is None:
            _rule_fallback(result, candidates, reason="未配置 LLM")
        elif candidates and llm is not None:
            decision_llm = DecisionLLM(llm)
            plan = _plan_sector(sector, candidates, decision_llm)
            try:
                kept_ids, reason, records = decide_sector_news(
                    sector, candidates, decision_llm, max_tool_calls=max_tool_calls_per_sector,
                    plan=plan,
                )
                result.tool_calls = records
                result.agent_reason = reason
                kept_set = set(kept_ids)
                result.kept = [c for c in candidates if c["id"] in kept_set]
                if result.kept:
                    critic_removed, critic_reason = _review_selection(
                        sector, result.kept, examples.get(sector, []), decision_llm
                    )
                    kept_set.difference_update(critic_removed)
                    result.kept = [c for c in result.kept if c["id"] in kept_set]
                result.dropped_ids = [c["id"] for c in candidates if c["id"] not in kept_set]
            except AgentDecisionError as exc:
                _rule_fallback(result, candidates, reason=str(exc))
            result.llm_calls = decision_llm.calls

        snapshot = {}
        if compare and fresh.get("error"):
            snapshot = {"snapshot_error": "增量过滤失败"}
        elif compare:
            snapshot = {
                "candidate_items": [_trace_item(item) for item in candidates],
                "agent_selected": [_trace_item(item) for item in result.kept],
                "rules_selected": _rules_snapshot(sector, candidates),
            }

        if result.kept:
            analyzed = []
            for item in result.kept:
                sentiment = call_tool(
                    "analyze_sentiment",
                    {"title": item.get("title", ""), "content": item.get("summary", ""),
                     "sectors": [sector]},
                )
                item["analysis"] = sentiment.get("analysis", {})
                analyzed.append(item)
            ranked = call_tool("rank_news", {"items": analyzed})
            result.kept = ranked.get("items", analyzed)[:10]
            digest = call_tool(
                "compose_digest",
                {"sector": sector, "items": result.kept},
            )
            if not digest.get("error"):
                report.digests[sector] = digest.get("digest", "")

        report.sectors.append(result)
        report.trace.append(
            {
                "sector": sector,
                "fetched": result.fetched,
                "after_dedupe": result.after_dedupe,
                "kept": len(result.kept),
                "kept_items": [_trace_item(item) for item in result.kept],
                "agent_fallback": result.agent_fallback,
                "agent_reason": result.agent_reason,
                "plan": plan,
                "critic_removed": critic_removed,
                "critic_reason": critic_reason,
                "llm_calls": result.llm_calls,
                "tool_calls": [
                    {"tool": r.tool, "ok": r.ok, "latency_ms": r.latency_ms}
                    for r in result.tool_calls
                ],
                **snapshot,
            }
        )

    if push and report.digests:
        content = "\n\n".join(
            f"<b>{escape(sector)}</b>\n{escape(text)}" for sector, text in report.digests.items()
        )
        pushed = call_tool(
            "push_telegram",
            {
                "title": f"板块日报 {date}", "content": content,
                "feedback_items": [
                    {"date": date, "sector": result.sector, "title": item["title"],
                     "link": item.get("原文链接", "")}
                    for result in report.sectors if report.digests.get(result.sector)
                    for item in result.kept
                ],
            },
        )
        report.pushed = not pushed.get("error")

    return report


def _trace_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "link": item.get("原文链接", ""),
    }


def _rules_snapshot(
    sector: str, candidates: list[dict[str, Any]]
) -> list[dict[str, str]] | None:
    selected = []
    for item in candidates:
        classified = call_tool("classify_news", {
            "sector": sector, "title": item["title"], "content": item["summary"],
            "use_llm": False,
        })
        if classified.get("error"):
            logger.warning("板块「%s」规则侧分类失败，本次跳过比较", sector)
            return None
        if classified.get("matched"):
            selected.append(_trace_item(item))
    return selected


def _rule_fallback(result: SectorResult, candidates: list[dict[str, Any]], reason: str) -> None:
    """Agent 不可用时的规则兜底：与现有 classifier 行为一致。"""
    kept = []
    for item in candidates:
        classified = call_tool(
            "classify_news",
            {"title": item.get("title", ""), "sector": result.sector, "use_llm": False},
        )
        if classified.get("matched"):
            kept.append(item)
    result.kept = kept
    result.agent_fallback = True
    result.agent_reason = f"规则兜底（{reason}）"
