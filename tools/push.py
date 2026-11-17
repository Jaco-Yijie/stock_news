"""工具：Telegram 推送。包装 notify.TelegramNotifier。"""
from __future__ import annotations

from typing import Any

import notify
from feedback_store import FeedbackItem

from .schemas import PushTelegramInput


def push_telegram(input: PushTelegramInput) -> dict[str, Any]:
    """推送消息到 Telegram。

    返回 {"sent", "error"}。环境变量缺失或发送失败时 sent=False，
    error 中说明原因，不抛异常（推送失败不应中断 Agent 流程）。
    """
    notifiers = [n for n in notify.load_notifiers_from_env()
                 if isinstance(n, notify.TelegramNotifier)]
    if not notifiers:
        return {"sent": False,
                "error": "未配置 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"}

    try:
        items = []
        for row in input.feedback_items:
            item = FeedbackItem.from_news(row.date, row.sector, row.title, row.link)
            if item is not None:
                items.append(item)
        if items:
            notifiers[0].send_digest(input.title, input.content, items)
        else:
            notifiers[0].send(input.title, input.content)
    except Exception as exc:  # noqa: BLE001 - 工具边界：异常转为返回值
        return {"sent": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"sent": True, "error": None}
