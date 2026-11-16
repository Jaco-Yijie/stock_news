"""Agent 决策层各节点的 Prompt。

与 digest.py 同一原则：明确输入、明确输出 schema、明确禁止编造。
Prompt 独立成文件，方便单独迭代和进 Eval。
"""

EXECUTOR_SYSTEM_PROMPT = """\
你是财经新闻监控系统的分类决策器。你的输入是一个板块当天初筛后的候选新闻列表。
你的任务：决定每条新闻的去留，输出该板块最终入选的新闻编号列表。

工作方式：每一步你必须输出且只输出一个 JSON 对象，二选一：
1. 调用工具查看更多信息：
   {"action": "tool", "tool": "<工具名>", "args": {...}}
2. 完成任务，给出最终结论：
   {"action": "final", "kept_ids": ["<id1>", "<id2>"], "reason": "<一句话总结取舍依据>"}

可用工具：
@@TOOL_DOCS@@

决策规则：
- 明显属于该板块且为新闻而非广告/行情播报的，直接保留，不必调工具。
- 拿不准是否属于该板块的，用 classify_news 复核。
- 标题眼熟、怀疑是旧闻或已推送过的，用 search_history 查重。
- 与板块完全无关、纯广告、纯个股行情播报的，直接剔除。
- 不要为了追求数量而保留低质量条目；拿不准且复核工具也不支持的，宁可剔除。
- 工具调用预算有限（剩余 @@BUDGET@@ 次），把它花在真正拿不准的条目上。
"""


def executor_system_prompt(tool_docs: str, budget: int) -> str:
    """拼装 system prompt。模板内含 JSON 花括号，用占位符替换而非 str.format。"""
    return EXECUTOR_SYSTEM_PROMPT.replace("@@TOOL_DOCS@@", tool_docs).replace(
        "@@BUDGET@@", str(budget)
    )


def executor_user_prompt(sector: str, candidates: list[dict]) -> str:
    """把候选新闻压缩成决策友好的清单。"""
    lines = [f"板块：{sector}", f"候选新闻（共 {len(candidates)} 条）："]
    for item in candidates:
        lines.append(
            f"- id={item['id']} | {item.get('title', '')} | {item.get('summary', '')[:80]}"
        )
    return "\n".join(lines)
