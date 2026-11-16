"""Agent 编排层：W2 最小闭环。

定位：抓取、去重、排序、日报生成仍是确定性工具调用（复用 tools/ 包）；
「每条候选新闻去留」这个模糊判断交给 Agent 决策循环。

为什么不引入 LangGraph：现阶段节点是线性流水线 + 一个决策循环，
llm_provider.complete() 是纯文本接口，用 JSON 协议自制 ReAct 循环
（本文件 ~120 行）即可跑通所有模型供应商，零新增重依赖，全部可单测。
W3 加入 Planner / Critic / 重试分支后，图结构变复杂再迁 LangGraph，
届时 nodes 函数签名（state -> state）无需改动。

LLM 可注入：生产用 llm_provider.load_llm_verifier_from_env()，
测试传 fake callable，所有测试不触网。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent import prompts
from tools import call_tool, tool_specs


class ChatLLM(Protocol):
    """决策层对 LLM 的最小依赖：纯文本进、纯文本出。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


# 决策循环里允许 Agent 调用的只读工具白名单（写操作不在其中）
EXECUTOR_ALLOWED_TOOLS = ("classify_news", "search_history")

DEFAULT_MAX_TOOL_CALLS_PER_SECTOR = 12


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
            "digests": len(self.digests),
            "pushed": self.pushed,
        }


def _parse_agent_action(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 动作。容忍 ```json 围栏和首尾噪声。"""
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
    return data if isinstance(data, dict) and "action" in data else None


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

    for _round in range(max_tool_calls + 4):  # 预留几轮给 final 输出
        user = prompts.executor_user_prompt(sector, candidates)
        if observations:
            user += "\n\n已获得的工具结果：\n" + "\n".join(observations)

        action = _parse_agent_action(llm.complete(system, user))
        if action is None:
            continue  # 输出无法解析，原样重试（temperature=0 时最多空转，受轮数上限保护）

        if action["action"] == "final":
            kept = [str(i) for i in action.get("kept_ids", [])]
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
) -> AgentReport:
    """W2 主流程：固定流水线，分类决策交给 Agent。

    每板块：fetch → filter_new_news → Agent 决策去留 → rank → compose_digest。
    Agent 失败（未配置 LLM / 决策异常）时回退到规则分类，保证日报不缺席。
    """
    report = AgentReport(date=date)

    for sector in sectors:
        result = SectorResult(sector=sector)

        fetched = call_tool("fetch_sector_news", {"sector": sector})
        if fetched.get("error"):
            result.error = f"fetch: {fetched['error']}"
            report.sectors.append(result)
            continue
        items = fetched.get("items", [])
        result.fetched = len(items)

        fresh = call_tool("filter_new_news", {"items": items})
        candidates = fresh.get("items", [])
        result.after_dedupe = len(candidates)
        for idx, item in enumerate(candidates):
            # 展示层字段为中文（标题/新闻内容/发布时间/原文链接）
            item["id"] = f"n{idx}"
            item["title"] = item.get("标题", "")
            item["summary"] = item.get("新闻内容", "")

        if not candidates:
            report.sectors.append(result)
            continue

        if llm is None:
            _rule_fallback(result, candidates, reason="未配置 LLM")
        else:
            try:
                kept_ids, reason, records = decide_sector_news(
                    sector, candidates, llm, max_tool_calls=max_tool_calls_per_sector
                )
                result.tool_calls = records
                result.agent_reason = reason
                kept_set = set(kept_ids)
                result.kept = [c for c in candidates if c["id"] in kept_set]
                result.dropped_ids = [c["id"] for c in candidates if c["id"] not in kept_set]
            except AgentDecisionError as exc:
                _rule_fallback(result, candidates, reason=str(exc))

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
                "kept_items": [
                    {
                        "id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "link": item.get("原文链接", ""),
                    }
                    for item in result.kept
                ],
                "agent_fallback": result.agent_fallback,
                "agent_reason": result.agent_reason,
                "tool_calls": [
                    {"tool": r.tool, "ok": r.ok, "latency_ms": r.latency_ms}
                    for r in result.tool_calls
                ],
            }
        )

    if push and report.digests:
        content = "\n\n".join(
            f"<b>{sector}</b>\n{text}" for sector, text in report.digests.items()
        )
        pushed = call_tool(
            "push_telegram",
            {"title": f"板块日报 {date}", "content": content},
        )
        report.pushed = not pushed.get("error")

    return report


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
