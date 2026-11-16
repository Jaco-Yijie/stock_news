"""Agent 工具层的输入/输出模型（Pydantic v2）。

每个工具函数对应一个 *Input 模型：字段即 Agent 可见的参数说明，
description 会直接进入工具的 JSON Schema，是 Agent 决定"何时、如何
调用这个工具"的主要依据，务必写得具体、可执行。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FetchSectorNewsInput(BaseModel):
    """抓取指定板块的最新新闻。"""

    sector: str = Field(
        description="板块名称，必须是 sectors.SECTORS 中的 key，如「半导体芯片」「人工智能」"
    )
    timeout: int = Field(
        default=20,
        ge=5,
        le=120,
        description="单源抓取超时秒数，网络差时调大，常规保持默认",
    )


class FilterNewNewsInput(BaseModel):
    """与历史缓存比对，只保留没出现过的新闻。"""

    items: list[dict[str, Any]] = Field(
        description="新闻字典列表，每条至少包含 标题/新闻内容/发布时间/链接 字段"
    )


class ClassifyNewsInput(BaseModel):
    """判断一条新闻是否属于某板块（规则 + 可选 LLM 复核）。"""

    title: str = Field(description="新闻标题")
    content: str = Field(default="", description="新闻正文或摘要，可为空")
    sector: str = Field(description="目标板块名称，同 sectors.SECTORS 的 key")
    use_llm: bool = Field(
        default=True,
        description="命中低置信度/高风险规则时是否调用 LLM 复核；批量粗筛时可关闭以省成本",
    )


class AnalyzeSentimentInput(BaseModel):
    """按规则评估一条新闻对目标板块的利好/利空方向与重要性。"""

    title: str = Field(description="新闻标题")
    content: str = Field(default="", description="新闻正文或摘要，可为空")
    sectors: list[str] = Field(description="需要评估影响的目标板块列表")


class SearchHistoryInput(BaseModel):
    """在历史缓存中检索是否已出现过相关新闻（去重、查旧闻）。"""

    query: str = Field(description="检索关键词，匹配标题与正文，如「固态电池」「降息」")
    limit: int = Field(default=10, ge=1, le=50, description="最多返回条数，按时间倒序")


class RankNewsInput(BaseModel):
    """按重要性（政策>产业事件>一般资讯>个股异动）对新闻排序。"""

    items: list[dict[str, Any]] = Field(
        description="带 analysis 字段的新闻列表（先经 analyze_sentiment 处理）"
    )


class ComposeDigestInput(BaseModel):
    """用 LLM 把一个板块的当日新闻归纳为 3-5 句中文日报。"""

    sector: str = Field(description="板块名称")
    items: list[dict[str, Any]] = Field(
        description="该板块当日新闻列表，每条含 标题/发布时间/新闻内容",
        max_length=20,
    )


class PushTelegramInput(BaseModel):
    """把内容推送到 Telegram（需要环境变量 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）。"""

    title: str = Field(description="推送标题，会加粗显示")
    content: str = Field(description="推送正文，支持 Telegram HTML 子集；超长自动分段")
