"""保存 Telegram 已推送新闻、最新标签和轮询游标。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from news_links import safe_news_url
from paths import DATA_DIR
from supabase_store import SupabaseConfigStore, load_supabase_credentials

FEEDBACK_DB = DATA_DIR / "telegram_feedback.sqlite3"
FEEDBACK_CASES = DATA_DIR / "telegram_feedback_cases.jsonl"
ITEM_PREFIX = "telegram-feedback-item-"
VOTE_PREFIX = "telegram-feedback-vote-"
# Telegram 在一周无更新后可能重置 update_id。
CURSOR_TTL = 6 * 24 * 60 * 60


class FeedbackBackend(Protocol):
    def get_value(self, config_key: str) -> str | None: ...
    def set_values(self, values: dict[str, str], *, overwrite: bool = True) -> None: ...
    def list_values(self, prefix: str) -> list[str]: ...


class FeedbackConfigurationError(RuntimeError):
    pass


class LocalFeedbackBackend:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS feedback (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            yield connection

    def get_value(self, config_key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM feedback WHERE key = ?", (config_key,)).fetchone()
        return row[0] if row else None

    def set_values(self, values: dict[str, str], *, overwrite: bool = True) -> None:
        action = "REPLACE" if overwrite else "IGNORE"
        with self._connect() as connection:
            connection.executemany(f"INSERT OR {action} INTO feedback VALUES (?, ?)", values.items())

    def list_values(self, prefix: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT value FROM feedback WHERE key LIKE ? ORDER BY key", (prefix + "%",))
            return [row[0] for row in rows]


@dataclass(frozen=True)
class FeedbackItem:
    date: str
    sector: str
    title: str
    link: str

    @classmethod
    def from_news(cls, date: str, sector: str, title: str, link: str) -> FeedbackItem | None:
        link = safe_news_url(link)
        if not link or not sector.strip() or not title.strip():
            return None
        return cls(date, sector.strip(), " ".join(title.split())[:200], link)

    @property
    def id(self) -> str:
        link_hash = hashlib.sha1(self.link.encode()).hexdigest()[:12]
        context_hash = hashlib.sha256(f"{self.date}\n{self.sector}".encode()).hexdigest()[:8]
        return f"{link_hash}-{context_hash}"


@dataclass(frozen=True)
class FeedbackVote:
    update_id: int
    case: dict[str, str]
    saved_at: float

    def already_processed(self, update_id: int) -> bool:
        return self.update_id >= update_id and time.time() - self.saved_at < CURSOR_TTL


class FeedbackStore:
    def __init__(self, backend: FeedbackBackend) -> None:
        self.backend = backend

    def register(self, items: list[FeedbackItem]) -> None:
        self.backend.set_values({
            ITEM_PREFIX + item.id: json.dumps(vars(item), ensure_ascii=False) for item in items
        }, overwrite=False)

    def get_item(self, item_id: str) -> FeedbackItem | None:
        value = self.backend.get_value(ITEM_PREFIX + item_id)
        return FeedbackItem(**json.loads(value)) if value is not None else None

    def get_vote(self, chat_id: str, item_id: str) -> FeedbackVote | None:
        value = self.backend.get_value(f"{VOTE_PREFIX}{chat_id}-{item_id}")
        return FeedbackVote(**json.loads(value)) if value is not None else None

    def save_vote(self, chat_id: str, item: FeedbackItem, action: str, update_id: int) -> None:
        case = {**vars(item), "type": {"keep": "kept", "drop": "false_positive"}[action],
                "note": "Telegram 推送人工反馈"}
        vote = FeedbackVote(update_id, case, time.time())
        self.backend.set_values({
            f"{VOTE_PREFIX}{chat_id}-{item.id}": json.dumps(vars(vote), ensure_ascii=False),
        })

    def cases(self) -> list[dict[str, str]]:
        return [FeedbackVote(**json.loads(value)).case for value in self.backend.list_values(VOTE_PREFIX)]

    def cursor(self, bot_id: str) -> int | None:
        value = self.backend.get_value(f"telegram-feedback-cursor-{bot_id}")
        if value is None:
            return None
        offset, saved_at = json.loads(value)
        return offset if time.time() - saved_at < CURSOR_TTL else None

    def save_cursor(self, bot_id: str, offset: int) -> None:
        self.backend.set_values({
            f"telegram-feedback-cursor-{bot_id}": json.dumps([offset, time.time()]),
        })


def get_feedback_store(*, require_supabase: bool = False, path: Path = FEEDBACK_DB) -> FeedbackStore:
    credentials = load_supabase_credentials()
    if credentials:
        return FeedbackStore(SupabaseConfigStore(*credentials))
    if require_supabase:
        raise FeedbackConfigurationError("缺少 SUPABASE_URL / SUPABASE_KEY")
    return FeedbackStore(LocalFeedbackBackend(path))
