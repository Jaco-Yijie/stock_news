"""推送通道：Telegram / PushPlus（微信）。

- 通道由环境变量决定，配了哪个用哪个，都未配置时静默跳过；
- 各通道自己渲染消息：Telegram 用 HTML，PushPlus 用 markdown；
- 推送历史（链接指纹）保存在 Supabase app_config / 本地文件，避免重复推送；
- 只推"政策类或高影响"的新闻，且限制单次条数，避免刷屏。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from html import escape
from typing import Any

import pandas as pd
import requests

from analysis import (
    SENTIMENT_LABELS,
    analyze_display_frame,
    group_similar_news,
    parse_publish_times,
    sort_news_by_importance,
    target_sectors_for_row,
)
from feedback_store import FeedbackItem, get_feedback_store
from paths import DATA_DIR
from news_links import safe_news_url
from supabase_store import SupabaseConfigStore, load_supabase_credentials
from time_utils import now_utc8_naive


DEFAULT_TIMEOUT = 15
MAX_PUSH_ITEMS = 8
DAILY_REPORT_ITEMS = 12
PUSH_WINDOW_HOURS = 24
PUSH_HISTORY_KEY = "push_history"
PUSH_HISTORY_PATH = DATA_DIR / "push_history.json"
PUSH_HISTORY_LIMIT = 800
# Telegram 单条消息上限 4096 字符，留出标题和分段标记的余量
TELEGRAM_MESSAGE_LIMIT = 3800
# 标题过长会让单个条目块超过分段上限，硬切可能切坏 HTML 标签，先在源头截断
MAX_TITLE_CHARS = 200


def _clean_secret(value: str) -> str:
    # 粘贴密钥时容易混入空格/换行，出现在 URL 或请求头里会导致 400
    return "".join(str(value or "").split())


def _redact_secret(value: str, secret: str) -> str:
    return value.replace(secret, "[已隐藏]") if secret else value


def _ensure_http_ok(response: requests.Response, channel: str, secret: str) -> None:
    if response.status_code >= 400:
        raise RuntimeError(
            f"{channel} HTTP {response.status_code}：{_redact_secret(response.text, secret)[:200]}"
        )


class PushPlusNotifier:
    name = "PushPlus"

    def __init__(self, token: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._token = token
        self._timeout = timeout

    def render(self, df: pd.DataFrame, overview: str = "") -> str:
        return format_push_markdown(df, overview)

    def send(self, title: str, content: str) -> None:
        try:
            response = requests.post(
                "https://www.pushplus.plus/send",
                json={
                    "token": self._token,
                    "title": title,
                    "content": content,
                    "template": "markdown",
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"PushPlus 网络请求失败（{type(exc).__name__}）") from None
        _ensure_http_ok(response, self.name, self._token)
        payload = response.json()
        if payload.get("code") != 200:
            detail = _redact_secret(str(payload), self._token)[:200]
            raise RuntimeError(f"PushPlus 返回错误：{detail}")


class TelegramAPIError(RuntimeError):
    pass


class TelegramAPI:
    def __init__(self, token: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._token = token
        self._timeout = timeout

    def call(self, method: str, body: dict[str, Any]) -> Any:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self._token}/{method}",
                json=body, timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise TelegramAPIError(f"Telegram 网络请求失败（{type(exc).__name__}）") from None
        if response.status_code >= 400:
            detail = _redact_secret(response.text, self._token)[:200]
            raise TelegramAPIError(f"Telegram HTTP {response.status_code}：{detail}")
        try:
            payload = response.json()
        except ValueError:
            raise TelegramAPIError("Telegram 返回的响应格式无效") from None
        if not isinstance(payload, dict) or not payload.get("ok"):
            detail = _redact_secret(str(payload), self._token)[:200]
            raise TelegramAPIError(f"Telegram 返回错误：{detail}")
        return payload.get("result")


class TelegramNotifier:
    name = "Telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._api = TelegramAPI(bot_token, timeout)
        self._chat_id = chat_id

    def render(self, df: pd.DataFrame, overview: str = "") -> str:
        return format_push_html(df, overview)

    def send(self, title: str, content: str) -> None:
        """按 Telegram 的 4096 字符上限分段发送。

        用 HTML 而不是 MarkdownV2：中文标题里的 ()、-、.、! 等字符在
        MarkdownV2 里都必须转义，漏一个就是 400；HTML 只需处理 & < >。
        """
        body = f"<b>{_escape_html(title)}</b>\n\n{content}".strip()
        chunks = _split_for_telegram(body)
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            text = chunk if total == 1 else f"{chunk}\n\n（{index}/{total}）"
            self._send_message(text)

    def _send_message(self, text: str, buttons: list[list[dict[str, str]]] | None = None) -> None:
        body: dict[str, Any] = {
            "chat_id": self._chat_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": True,
        }
        if buttons:
            body["reply_markup"] = {"inline_keyboard": buttons}
        self._api.call("sendMessage", body)

    def send_news(self, title: str, df: pd.DataFrame, overview: str = "") -> None:
        date = now_utc8_naive().date().isoformat()
        blocks = []
        for _, row in df.iterrows():
            _, _, headline, _, _, link = _row_fields(row)
            sectors = target_sectors_for_row(row)
            sector = sectors[0] if len(sectors) == 1 else "外部事件"
            blocks.append((_format_line_html(row), FeedbackItem.from_news(date, sector, headline, link)))
        intro = f"{_escape_html(overview.strip())}\n\n———" if overview.strip() else ""
        self._send_feedback_blocks(title, intro, blocks)

    def send_digest(self, title: str, content: str, items: list[FeedbackItem]) -> None:
        if not items or not re.fullmatch(r"-?[0-9]+", self._chat_id):
            self.send(title, content)
            return
        blocks = [
            (
                f'<b>{escape(item.title)}</b>\n{escape(item.sector)} ｜ '
                f'<a href="{escape(item.link, quote=True)}">原文</a>',
                item,
            )
            for item in items
        ]
        self.send(title, content)
        self._send_feedback_blocks(f"{title} · 新闻反馈", "", blocks, skip_if_unavailable=True)

    def _send_feedback_blocks(
        self, title: str, intro: str, blocks: list[tuple[str, FeedbackItem | None]], *,
        skip_if_unavailable: bool = False,
    ) -> None:
        items = [item for _, item in blocks if item is not None]
        if items and re.fullmatch(r"-?[0-9]+", self._chat_id):
            try:
                get_feedback_store().register(items)
            except Exception as exc:
                print(f"[warning] Telegram 反馈存储不可用（{type(exc).__name__}）")
            else:
                self._send_registered_blocks(title, intro, blocks)
                return
        if not skip_if_unavailable:
            self.send(title, "\n\n".join(part for part in [intro, *(block for block, _ in blocks)] if part))

    def _send_registered_blocks(
        self, title: str, intro: str, blocks: list[tuple[str, FeedbackItem | None]]
    ) -> None:
        heading = f"<b>{_escape_html(title)}</b>"
        current = f"{heading}\n\n{intro}" if intro else heading
        buttons = []
        for index, (block, item) in enumerate(blocks, 1):
            if item is not None:
                block = f"{index}. {block}"
            if len(current) + len(block) + 2 > TELEGRAM_MESSAGE_LIMIT or len(buttons) == DAILY_REPORT_ITEMS:
                if len(current) > TELEGRAM_MESSAGE_LIMIT:
                    for chunk in _split_for_telegram(current):
                        self._send_message(chunk)
                elif current != heading:
                    self._send_message(current, buttons)
                current, buttons = heading, []
            if len(heading) + len(block) + 2 > TELEGRAM_MESSAGE_LIMIT:
                self.send(title, block)
                continue
            current += f"\n\n{block}"
            if item is not None:
                buttons.append([
                    {"text": f"👍{index} 该留", "callback_data": f"keep:{item.id}"},
                    {"text": f"👎{index} 不该留", "callback_data": f"drop:{item.id}"},
                ])
        if current != heading:
            self._send_message(current, buttons)


def load_notifiers_from_env() -> list[Any]:
    notifiers: list[Any] = []
    telegram_token = _clean_secret(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat = _clean_secret(os.getenv("TELEGRAM_CHAT_ID", ""))
    if telegram_token and telegram_chat:
        notifiers.append(TelegramNotifier(telegram_token, telegram_chat))
    pushplus_token = _clean_secret(os.getenv("PUSHPLUS_TOKEN", ""))
    if pushplus_token:
        notifiers.append(PushPlusNotifier(pushplus_token))
    return notifiers


def send_to_all(
    notifiers: list[Any],
    title: str,
    df: pd.DataFrame,
    overview: str = "",
) -> list[str]:
    """让每个通道按自己的格式渲染并发送，返回失败信息列表（全部成功时为空）。"""
    errors: list[str] = []
    for notifier in notifiers:
        try:
            if isinstance(notifier, TelegramNotifier):
                notifier.send_news(title, df, overview)
            else:
                notifier.send(title, notifier.render(df, overview))
        except Exception as exc:
            errors.append(f"{notifier.name}: {exc}")
    return errors


def _link_hash(link: str) -> str:
    return hashlib.sha1(str(link or "").encode("utf-8")).hexdigest()[:16]


def load_push_history(path=PUSH_HISTORY_PATH) -> list[str]:
    credentials = load_supabase_credentials()
    if credentials is not None:
        try:
            value = SupabaseConfigStore(credentials[0], credentials[1]).get_value(
                PUSH_HISTORY_KEY
            )
            if value:
                data = json.loads(value)
                if isinstance(data, list):
                    return [str(item) for item in data]
        except Exception:
            pass
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item) for item in data]
    except Exception:
        pass
    return []


def save_push_history(history: list[str], path=PUSH_HISTORY_PATH) -> None:
    trimmed = history[-PUSH_HISTORY_LIMIT:]
    payload = json.dumps(trimmed, ensure_ascii=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    except Exception:
        pass
    credentials = load_supabase_credentials()
    if credentials is not None:
        try:
            SupabaseConfigStore(credentials[0], credentials[1]).set_value(
                PUSH_HISTORY_KEY, payload
            )
        except Exception:
            pass


def select_push_worthy(display_df: pd.DataFrame) -> pd.DataFrame:
    """筛选值得推送的新闻：政策类或高影响，且发布在最近 24 小时内。"""
    if display_df is None or display_df.empty or "analysis" not in display_df.columns:
        return display_df.iloc[0:0] if display_df is not None else pd.DataFrame()

    publish_times = parse_publish_times(display_df)
    cutoff = pd.Timestamp(now_utc8_naive()) - pd.Timedelta(hours=PUSH_WINDOW_HOURS)
    recent_mask = publish_times.ge(cutoff).fillna(False)

    def important(analysis: Any) -> bool:
        if not isinstance(analysis, dict):
            return False
        return analysis.get("category") == "policy" or analysis.get("impact_level") == "high"

    important_mask = display_df["analysis"].map(important)
    selected = display_df[recent_mask & important_mask]
    return sort_news_by_importance(selected)


def filter_unpushed(df: pd.DataFrame, history: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    pushed = set(history)
    keep_mask = [
        _link_hash(link) not in pushed for link in df["原文链接"].astype(str)
    ]
    return df[keep_mask].reset_index(drop=True)


def hashes_for(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    return [_link_hash(link) for link in df["原文链接"].astype(str)]


def _escape_html(value: Any) -> str:
    """Telegram HTML 模式只把 & < > 当特殊字符，转义这三个即可。"""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _row_fields(row: pd.Series) -> tuple[str, str, str, str, str, str]:
    analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
    sentiment = SENTIMENT_LABELS.get(analysis.get("sentiment", "neutral"), "中性")
    sector = str(row.get("sector", "") or "").strip()
    sectors = sector or "、".join(list(analysis.get("sector_assessments") or {})[:3]) or "-"
    title = str(row.get("标题", "")).strip()[:MAX_TITLE_CHARS]
    source = str(row.get("来源媒体", "")).strip()
    publish_time = str(row.get("发布时间", "")).strip()
    link = safe_news_url(row.get("原文链接"))
    return sentiment, sectors, title, source, publish_time, link


def _escape_markdown(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()!<>])", r"\\\1", " ".join(value.split()))


def _format_line(row: pd.Series) -> str:
    sentiment, sectors, title, source, publish_time, link = _row_fields(row)
    destination = link.replace("(", "%28").replace(")", "%29")
    return (
        f"**【{sentiment}】{_escape_markdown(title)}**\n"
        f"板块：{_escape_markdown(sectors)} ｜ {_escape_markdown(source)} {publish_time}"
        + (f"\n[原文]({destination})" if destination else "")
    )


def _format_line_html(row: pd.Series) -> str:
    sentiment, sectors, title, source, publish_time, link = _row_fields(row)
    return (
        f"<b>【{sentiment}】{_escape_html(title)}</b>\n"
        f"板块：{_escape_html(sectors)} ｜ {_escape_html(source)} "
        f"{_escape_html(publish_time)}"
        + (f'\n<a href="{escape(link, quote=True)}">原文</a>' if link else "")
    )


def _join_blocks(overview: str, blocks: list[str], separator: str) -> str:
    parts = [block for block in blocks if block]
    if overview:
        parts = [overview, separator, *parts]
    return "\n\n".join(parts)


def format_push_markdown(df: pd.DataFrame, overview: str = "") -> str:
    blocks = [_format_line(row) for _, row in df.iterrows()]
    return _join_blocks(overview.strip(), blocks, "---")


def format_push_html(df: pd.DataFrame, overview: str = "") -> str:
    blocks = [_format_line_html(row) for _, row in df.iterrows()]
    return _join_blocks(_escape_html(overview.strip()) if overview else "", blocks, "———")


def _split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """按空行边界切分，保证不切开单个条目（也就不会切坏 HTML 标签）。"""
    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # 单块自身超限只可能是 LLM 总览那种纯文本，硬切不会破坏标签
        while len(block) > limit:
            chunks.append(block[:limit])
            block = block[limit:]
        current = block
    if current:
        chunks.append(current)
    return chunks or [text]


def _prepare_display(cache_df: pd.DataFrame) -> pd.DataFrame:
    from news_store import cache_to_display

    display_df = cache_to_display(cache_df)
    if display_df.empty:
        return display_df
    display_df = display_df.copy()
    display_df["analysis"] = analyze_display_frame(display_df)
    return display_df


def push_important_news(cache_df: pd.DataFrame) -> None:
    """抓取任务完成后调用：推送尚未推送过的重要新闻。"""
    notifiers = load_notifiers_from_env()
    if not notifiers:
        print("[info] 未配置推送通道，跳过重要新闻推送。")
        return

    display_df = _prepare_display(cache_df)
    candidates = select_push_worthy(display_df)
    history = load_push_history()
    fresh = filter_unpushed(candidates, history).head(MAX_PUSH_ITEMS)
    if fresh.empty:
        print("[info] 没有新的重要新闻需要推送。")
        return

    now = now_utc8_naive()
    title = f"板块要闻 {len(fresh)} 条 · {now.strftime('%m-%d %H:%M')}"
    errors = send_to_all(notifiers, title, fresh)
    for error in errors:
        print(f"[warning] 推送失败 {error}")
    if len(errors) < len(notifiers):
        save_push_history(history + hashes_for(fresh))
        print(f"[info] 已推送重要新闻 {len(fresh)} 条。")


def push_daily_report(cache_df: pd.DataFrame, llm_verifier: Any = None) -> None:
    """每日早报：最近 24 小时按重要性排序的要闻 Top 榜（可选 LLM 总览）。"""
    notifiers = load_notifiers_from_env()
    if not notifiers:
        print("[info] 未配置推送通道，跳过每日早报。")
        return

    display_df = _prepare_display(cache_df)
    if display_df.empty:
        print("[info] 缓存为空，跳过每日早报。")
        return

    publish_times = parse_publish_times(display_df)
    cutoff = pd.Timestamp(now_utc8_naive()) - pd.Timedelta(hours=PUSH_WINDOW_HOURS)
    recent_df = display_df[publish_times.ge(cutoff).fillna(False)]
    if recent_df.empty:
        print("[info] 最近 24 小时没有新闻，跳过每日早报。")
        return

    sorted_df = sort_news_by_importance(recent_df.reset_index(drop=True))
    clusters = group_similar_news(sorted_df.head(DAILY_REPORT_ITEMS * 3))
    top_df = sorted_df.iloc[
        [primary for primary, _ in clusters[:DAILY_REPORT_ITEMS]]
    ].reset_index(drop=True)

    overview = ""
    if llm_verifier is not None:
        titles = [str(row.get("标题", "")) for _, row in top_df.iterrows()]
        try:
            overview = llm_verifier.complete(
                "你是A股新闻编辑。只根据给定标题归纳，不得编造。",
                json.dumps(
                    {
                        "task": "用 3 到 5 句中文概括这些新闻反映的市场重点与方向，直接输出段落。",
                        "titles": titles,
                    },
                    ensure_ascii=False,
                ),
            )
            overview = " ".join(str(overview or "").split())
        except Exception as exc:
            print(f"[warning] 早报 LLM 总览生成失败（{type(exc).__name__}）")
            overview = ""

    now = now_utc8_naive()
    title = f"板块早报 {now.strftime('%m-%d')}"

    errors = send_to_all(notifiers, title, top_df, overview)
    for error in errors:
        print(f"[warning] 早报推送失败 {error}")
    if len(errors) < len(notifiers):
        history = load_push_history()
        save_push_history(history + hashes_for(top_df))
        print(f"[info] 每日早报已推送（{len(top_df)} 条要闻）。")
