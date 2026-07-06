"""推送通道：Server酱 / PushPlus（微信）、Telegram。

- 通道由环境变量决定，配了哪个用哪个，都未配置时静默跳过；
- 推送历史（链接指纹）保存在 Supabase app_config / 本地文件，避免重复推送；
- 只推"政策类或高影响"的新闻，且限制单次条数，避免刷屏。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd
import requests

from analysis import (
    SENTIMENT_LABELS,
    analyze_display_frame,
    group_similar_news,
    parse_publish_times,
    sort_news_by_importance,
)
from paths import DATA_DIR
from supabase_store import SupabaseConfigStore, load_supabase_credentials
from time_utils import now_utc8_naive


DEFAULT_TIMEOUT = 15
MAX_PUSH_ITEMS = 8
DAILY_REPORT_ITEMS = 12
PUSH_WINDOW_HOURS = 24
PUSH_HISTORY_KEY = "push_history"
PUSH_HISTORY_PATH = DATA_DIR / "push_history.json"
PUSH_HISTORY_LIMIT = 800


def _clean_secret(value: str) -> str:
    # 粘贴密钥时容易混入空格/换行，出现在 URL 或请求头里会导致 400
    return "".join(str(value or "").split())


def _ensure_http_ok(response: requests.Response, channel: str) -> None:
    if response.status_code >= 400:
        raise RuntimeError(
            f"{channel} HTTP {response.status_code}：{response.text[:200]}"
        )


class ServerChanNotifier:
    name = "Server酱"

    def __init__(self, sendkey: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._url = f"https://sctapi.ftqq.com/{sendkey}.send"
        self._timeout = timeout

    def send(self, title: str, content: str) -> None:
        response = requests.post(
            self._url,
            data={"title": title[:32], "desp": content},
            timeout=self._timeout,
        )
        _ensure_http_ok(response, self.name)
        payload = response.json()
        if payload.get("code") not in (0, 200):
            raise RuntimeError(f"Server酱返回错误：{str(payload)[:200]}")


class PushPlusNotifier:
    name = "PushPlus"

    def __init__(self, token: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._token = token
        self._timeout = timeout

    def send(self, title: str, content: str) -> None:
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
        _ensure_http_ok(response, self.name)
        payload = response.json()
        if payload.get("code") != 200:
            raise RuntimeError(f"PushPlus 返回错误：{str(payload)[:200]}")


class TelegramNotifier:
    name = "Telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, title: str, content: str) -> None:
        response = requests.post(
            self._url,
            json={"chat_id": self._chat_id, "text": f"{title}\n\n{content}"},
            timeout=self._timeout,
        )
        _ensure_http_ok(response, self.name)
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram 返回错误：{str(payload)[:200]}")


def load_notifiers_from_env() -> list[Any]:
    notifiers: list[Any] = []
    sendkey = _clean_secret(os.getenv("SERVERCHAN_SENDKEY", ""))
    if sendkey:
        notifiers.append(ServerChanNotifier(sendkey))
    pushplus_token = _clean_secret(os.getenv("PUSHPLUS_TOKEN", ""))
    if pushplus_token:
        notifiers.append(PushPlusNotifier(pushplus_token))
    telegram_token = _clean_secret(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat = _clean_secret(os.getenv("TELEGRAM_CHAT_ID", ""))
    if telegram_token and telegram_chat:
        notifiers.append(TelegramNotifier(telegram_token, telegram_chat))
    return notifiers


def send_to_all(notifiers: list[Any], title: str, content: str) -> list[str]:
    """向所有通道发送，返回失败信息列表（全部成功时为空）。"""
    errors: list[str] = []
    for notifier in notifiers:
        try:
            notifier.send(title, content)
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


def _format_line(row: pd.Series) -> str:
    analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
    sentiment = SENTIMENT_LABELS.get(analysis.get("sentiment", "neutral"), "中性")
    sector = str(row.get("sector", "") or "").strip()
    sectors = sector or "、".join(list(analysis.get("sector_assessments") or {})[:3]) or "-"
    title = str(row.get("标题", "")).strip()
    source = str(row.get("来源媒体", "")).strip()
    publish_time = str(row.get("发布时间", "")).strip()
    link = str(row.get("原文链接", "")).strip()
    return (
        f"**【{sentiment}】{title}**\n"
        f"板块：{sectors} ｜ {source} {publish_time}\n"
        f"[原文]({link})"
    )


def format_push_markdown(df: pd.DataFrame) -> str:
    return "\n\n".join(_format_line(row) for _, row in df.iterrows())


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
    errors = send_to_all(notifiers, title, format_push_markdown(fresh))
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
            print(f"[warning] 早报 LLM 总览生成失败：{exc}")
            overview = ""

    now = now_utc8_naive()
    title = f"A股板块早报 {now.strftime('%m-%d')}"
    content = format_push_markdown(top_df)
    if overview:
        content = f"{overview}\n\n---\n\n{content}"

    errors = send_to_all(notifiers, title, content)
    for error in errors:
        print(f"[warning] 早报推送失败 {error}")
    if len(errors) < len(notifiers):
        history = load_push_history()
        save_push_history(history + hashes_for(top_df))
        print(f"[info] 每日早报已推送（{len(top_df)} 条要闻）。")
