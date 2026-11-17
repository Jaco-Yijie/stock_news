"""轮询 Telegram 回调，持久化审核结果，或导出离线案例。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from feedback_store import FEEDBACK_CASES, FeedbackConfigurationError, FeedbackStore, get_feedback_store
from notify import TelegramAPI, TelegramAPIError, _clean_secret
from supabase_store import SupabaseError

CALLBACK_DATA = re.compile(r"(keep|drop|why):([0-9a-f]{12}-[0-9a-f]{8})")
PAGE_LIMIT = 100
MAX_PAGES = 10
POLL_TIMEOUT = 25


class FeedbackPoller:
    def __init__(self, api: TelegramAPI, store: FeedbackStore, bot_id: str, chat_id: str) -> None:
        self.api = api
        self.store = store
        self.bot_id = bot_id
        self.chat_id = chat_id

    def _best_effort(self, method: str, body: dict[str, Any]) -> None:
        try:
            self.api.call(method, body)
        except RuntimeError:
            print(f"[warning] Telegram {method} 失败，已保存的反馈保留", file=sys.stderr)

    def _process(self, update: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        callback = update.get("callback_query")
        if not isinstance(callback, dict) or not isinstance(callback.get("id"), str):
            return False, None
        message = callback.get("message")
        chat = message.get("chat") if isinstance(message, dict) else None
        if not isinstance(chat, dict) or str(chat.get("id")) != self.chat_id:
            return False, None

        data = callback.get("data")
        parsed = CALLBACK_DATA.fullmatch(data) if isinstance(data, str) else None
        item = self.store.get_item(parsed[2]) if parsed else None
        recorded = False
        edited_message = None
        if parsed is None or item is None:
            answer = "这条反馈已不可用"
        elif parsed[1] == "why":
            answer = "请按这条新闻是否应该进入当前推送来选择。"
        else:
            update_id = update["update_id"]
            vote = self.store.get_vote(self.chat_id, item.id)
            if vote is None or not vote.already_processed(update_id):
                self.store.save_vote(self.chat_id, item, parsed[1], update_id)
                recorded = True
            edited_message = message
            answer = "已记录"
        self._best_effort("answerCallbackQuery", {
            "callback_query_id": callback["id"], "text": answer,
        })
        return recorded, edited_message

    def _mark_keyboard(self, message: dict[str, Any]) -> None:
        markup = message.get("reply_markup")
        if not isinstance(markup, dict) or not isinstance(markup.get("inline_keyboard"), list):
            return
        votes = {}
        keyboard = []
        for row in markup["inline_keyboard"]:
            if not isinstance(row, list):
                return
            buttons = []
            for button in row:
                if not isinstance(button, dict):
                    return
                match = CALLBACK_DATA.fullmatch(str(button.get("callback_data", "")))
                if not match or not isinstance(button.get("text"), str):
                    buttons.append(button)
                    continue
                action, item_id = match.groups()
                if item_id not in votes:
                    vote = self.store.get_vote(self.chat_id, item_id)
                    votes[item_id] = vote.case["type"] if vote else ""
                marked = {"keep": "kept", "drop": "false_positive"}.get(action) == votes[item_id]
                text = button["text"].removeprefix("✓ ")
                buttons.append({**button, "text": f"✓ {text}" if marked else text})
            keyboard.append(buttons)
        if keyboard != markup["inline_keyboard"]:
            self._best_effort("editMessageReplyMarkup", {
                "chat_id": self.chat_id, "message_id": message["message_id"],
                "reply_markup": {"inline_keyboard": keyboard},
            })

    def poll_page(self, *, wait: int = 0) -> tuple[int, int]:
        params = {"timeout": wait, "allowed_updates": ["callback_query"], "limit": PAGE_LIMIT}
        offset = self.store.cursor(self.bot_id)
        if offset is not None:
            params["offset"] = offset
        updates = self.api.call("getUpdates", params)
        if not isinstance(updates, list) or any(
            not isinstance(row, dict) or type(row.get("update_id")) is not int for row in updates
        ):
            raise RuntimeError("Telegram 更新格式无效")

        count = 0
        messages: dict[int, dict[str, Any]] = {}
        for update in sorted(updates, key=lambda row: row["update_id"]):
            if offset is not None and update["update_id"] < offset:
                continue
            recorded, message = self._process(update)
            count += recorded
            if message and type(message.get("message_id")) is int:
                messages[message["message_id"]] = message
            offset = update["update_id"] + 1
        for message in messages.values():
            self._mark_keyboard(message)
        if offset is not None and offset != params.get("offset"):
            self.store.save_cursor(self.bot_id, offset)
        return len(updates), count


def export_cases(store: FeedbackStore, path: Path) -> int:
    rows = store.cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            for row in sorted(rows, key=lambda row: (row["date"], row["sector"], row["link"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="处理最多 1000 条待处理更新后退出")
    mode.add_argument("--export", action="store_true", help="导出案例 JSONL，不调用 Telegram")
    parser.add_argument("--output", type=Path, default=FEEDBACK_CASES, help="案例导出路径")
    parser.add_argument("--require-supabase", action="store_true", help="缺少 Supabase 配置时退出")
    args = parser.parse_args()
    try:
        store = get_feedback_store(require_supabase=args.require_supabase)
        if args.export:
            print(f"[info] 导出 {export_cases(store, args.output)} 条 Telegram 反馈")
            return 0
        token = _clean_secret(os.getenv("TELEGRAM_BOT_TOKEN", ""))
        chat_id = _clean_secret(os.getenv("TELEGRAM_CHAT_ID", ""))
        bot_id, _, _ = token.partition(":")
        if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) or not re.fullmatch(r"-?[0-9]+", chat_id):
            print("[error] 需要有效的 TELEGRAM_BOT_TOKEN 和数字 TELEGRAM_CHAT_ID", file=sys.stderr)
            return 1
        poller = FeedbackPoller(TelegramAPI(token, timeout=POLL_TIMEOUT + 10), store, bot_id, chat_id)
        if args.once:
            total = 0
            for _ in range(MAX_PAGES):
                size, count = poller.poll_page()
                total += count
                if size < PAGE_LIMIT:
                    break
            else:
                print("[warning] 已达轮询批次上限，后续运行将继续处理")
            print(f"[info] 已保存 {total} 条 Telegram 反馈")
        else:
            while True:
                _, count = poller.poll_page(wait=POLL_TIMEOUT)
                if count:
                    print(f"[info] 已保存 {count} 条 Telegram 反馈", flush=True)
    except KeyboardInterrupt:
        return 130
    except (FeedbackConfigurationError, SupabaseError, TelegramAPIError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[error] Telegram 反馈任务失败（{type(exc).__name__}）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
