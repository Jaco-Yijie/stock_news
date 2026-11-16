"""工具：板块日报生成。包装 digest.generate_sector_digest。"""
from __future__ import annotations

from typing import Any

import digest as digest_module

from .schemas import ComposeDigestInput

# Agent 运行时装配点：graph 层在调用前注入 verifier。
_llm_verifier: Any = None


def set_llm_verifier(verifier: Any) -> None:
    """注入具备 complete(system_prompt, user_prompt) 方法的 LLM 客户端。"""
    global _llm_verifier
    _llm_verifier = verifier


def compose_digest(input: ComposeDigestInput) -> dict[str, Any]:
    """用 LLM 把板块当日新闻归纳为 3-5 句中文日报，并给出利好/利空倾向。

    返回 {"sector", "digest", "used_count", "error"}。
    items 为空或 verifier 未注入时返回 error，不抛异常。
    """
    records = digest_module.build_digest_records(_to_frame(input.items))
    if not records:
        return {"sector": input.sector, "digest": "", "used_count": 0,
                "error": "没有可归纳的新闻"}
    if _llm_verifier is None:
        return {"sector": input.sector, "digest": "", "used_count": len(records),
                "error": "LLM verifier 未注入，请先调用 set_llm_verifier()"}

    text = digest_module.generate_sector_digest(_llm_verifier, input.sector, records)
    return {"sector": input.sector, "digest": text, "used_count": len(records),
            "error": None}


def _to_frame(items: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(items)
