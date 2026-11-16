"""Agent 工具层：把现有确定性模块包装成 Agent 可调用的工具。

设计原则：
- 只做包装，不改 fetcher / classifier / analysis / digest / notify 的任何逻辑；
- 每个工具：Pydantic 输入模型 + 返回 JSON 可序列化 dict + docstring 即说明书；
- 工具边界内捕获异常并转为 error 字段，不向上抛（推送/缓存失败不应中断 Agent）；
- 需要 LLM 的工具通过 set_llm_verifier() 在运行时注入，工具本身不感知供应商。

TOOLS 注册表供 W2 的 LangGraph 节点消费：name/docstring/input_schema 可直接
转换为 Function Calling 的 tool spec。
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from . import analyze, classify, dedupe, digest, fetch, history, push
from .schemas import (
    AnalyzeSentimentInput,
    ClassifyNewsInput,
    ComposeDigestInput,
    FetchSectorNewsInput,
    FilterNewNewsInput,
    PushTelegramInput,
    RankNewsInput,
    SearchHistoryInput,
)

# (名称, 处理函数, 输入模型) —— Agent 的"手"
TOOLS: list[dict[str, Any]] = [
    {"name": "fetch_sector_news", "fn": fetch.fetch_sector_news, "input_model": FetchSectorNewsInput},
    {"name": "filter_new_news", "fn": dedupe.filter_new_news, "input_model": FilterNewNewsInput},
    {"name": "classify_news", "fn": classify.classify_news, "input_model": ClassifyNewsInput},
    {"name": "analyze_sentiment", "fn": analyze.analyze_sentiment, "input_model": AnalyzeSentimentInput},
    {"name": "rank_news", "fn": analyze.rank_news, "input_model": RankNewsInput},
    {"name": "search_history", "fn": history.search_history, "input_model": SearchHistoryInput},
    {"name": "compose_digest", "fn": digest.compose_digest, "input_model": ComposeDigestInput},
    {"name": "push_telegram", "fn": push.push_telegram, "input_model": PushTelegramInput},
]

TOOL_BY_NAME: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOLS}


def tool_specs() -> list[dict[str, Any]]:
    """导出 OpenAI Function Calling 风格的工具描述，供 Agent 决策层使用。"""
    specs = []
    for t in TOOLS:
        specs.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t["fn"].__doc__ or "").strip(),
                "parameters": t["input_model"].model_json_schema(),
            },
        })
    return specs


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """按名称调用工具：校验输入 → 执行 → 返回 dict。未知工具返回 error。"""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return {"error": f"未知工具「{name}」，可选：{'、'.join(sorted(TOOL_BY_NAME))}"}
    try:
        input_obj = tool["input_model"].model_validate(arguments)
        return tool["fn"](input_obj)
    except Exception as exc:  # noqa: BLE001 - 工具边界：异常转为返回值
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "TOOLS", "TOOL_BY_NAME", "tool_specs", "call_tool",
    "analyze", "classify", "dedupe", "digest", "fetch", "history", "push",
    "BaseModel", "Callable",
]
