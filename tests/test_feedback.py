from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import feedback_store
import notify
import feedback_bot
from agent.bad_case import evaluate, false_positive_examples, load_cases
from feedback_bot import FeedbackPoller, export_cases
from feedback_store import FeedbackItem, FeedbackStore, LocalFeedbackBackend
from tools.push import push_telegram
from tools.schemas import PushTelegramInput

CHAT = "-100123"
FIRST = FeedbackItem("2026-11-16", "半导体", "订单增长", "https://example.com/first")
SECOND = FeedbackItem("2026-11-16", "人工智能", "模型更新", "https://example.com/second")


class FakeTelegram:
    def __init__(self, pages=(), *, expire_answers=False):
        self.pages = iter(pages)
        self.calls = []
        self.expire_answers = expire_answers

    def call(self, method, body):
        self.calls.append((method, body))
        if method == "getUpdates":
            return next(self.pages)
        if method == "answerCallbackQuery" and self.expire_answers:
            raise RuntimeError("query is too old")
        return True


def callback(update_id, action, item=FIRST, *, chat=CHAT, markup=None):
    message = {"chat": {"id": int(chat)}, "message_id": 15}
    if markup is not None:
        message["reply_markup"] = markup
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"callback-{update_id}", "data": f"{action}:{item.id}", "message": message,
        },
    }


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(LocalFeedbackBackend(tmp_path / "feedback.sqlite3"))


def test_send_click_export_and_evaluate(store, monkeypatch, tmp_path):
    sender = FakeTelegram()
    notifier = notify.TelegramNotifier("fake-token", CHAT)
    notifier._api = sender
    monkeypatch.setattr(notify, "get_feedback_store", lambda: store)
    monkeypatch.setattr(notify, "now_utc8_naive", lambda: pd.Timestamp(FIRST.date))
    rows = pd.DataFrame([
        {"sector": item.sector, "标题": item.title, "原文链接": item.link}
        for item in [FIRST, SECOND]
    ])
    assert notify.send_to_all([notifier], "早报", rows, "总览") == []
    method, body = sender.calls[0]
    assert method == "sendMessage"
    assert "1. <b>【中性】订单增长</b>" in body["text"]
    assert "2. <b>【中性】模型更新</b>" in body["text"]
    markup = body["reply_markup"]
    assert markup["inline_keyboard"][0][0]["callback_data"] == f"keep:{FIRST.id}"
    assert FIRST.id.startswith(hashlib.sha1(FIRST.link.encode()).hexdigest()[:12])
    assert FIRST.id != FeedbackItem(FIRST.date, SECOND.sector, FIRST.title, FIRST.link).id

    receiver = FakeTelegram([[
        callback(12, "keep", SECOND, markup=markup),
        callback(11, "drop", markup=markup),
        callback(10, "keep", markup=markup),
    ]], expire_answers=True)
    poller = FeedbackPoller(receiver, store, "123", CHAT)
    assert poller.poll_page() == (3, 3)
    assert store.cursor("123") == 13
    edits = [request for method, request in receiver.calls if method == "editMessageReplyMarkup"]
    assert len(edits) == 1
    marked = edits[0]["reply_markup"]["inline_keyboard"]
    assert marked[0][1]["text"].startswith("✓ ")
    assert marked[1][0]["text"].startswith("✓ ")

    exported = tmp_path / "feedback.jsonl"
    assert export_cases(store, exported) == 2
    assert {row["type"] for row in load_cases(exported)} == {"kept", "false_positive"}
    assert false_positive_examples(tmp_path / "local.jsonl", before="2026-11-17",
                                   feedback_path=exported) == {FIRST.sector: [FIRST.title]}
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps({
        "date": item.date, "sector": item.sector, "llm_calls": 1,
        "candidate_items": [{"link": item.link}],
        "agent_selected": [{"link": item.link}] if item == SECOND else [],
    }) for item in [FIRST, SECOND]), encoding="utf-8")
    result = evaluate([trace], tmp_path / "local.jsonl", feedback_path=exported)
    assert result["counts"] == {"passed": 2, "failed": 0, "not_covered": 0}


def test_store_failure_replays_without_overwriting_a_saved_vote(store, monkeypatch):
    store.register([FIRST, SECOND])
    updates = [callback(20, "drop"), callback(21, "keep", SECOND)]
    original = store.save_vote

    def fail_second(chat, item, action, update_id):
        if item == SECOND:
            raise RuntimeError("database unavailable")
        return original(chat, item, action, update_id)

    monkeypatch.setattr(store, "save_vote", fail_second)
    with pytest.raises(RuntimeError, match="database unavailable"):
        FeedbackPoller(FakeTelegram([updates]), store, "123", CHAT).poll_page()
    assert store.cursor("123") is None
    first_vote = store.get_vote(CHAT, FIRST.id)
    assert first_vote is not None

    reopened = FeedbackStore(LocalFeedbackBackend(store.backend.path))
    api = FakeTelegram([updates, updates])
    poller = FeedbackPoller(api, reopened, "123", CHAT)
    assert poller.poll_page() == (2, 1)
    assert reopened.get_vote(CHAT, FIRST.id) == first_vote
    assert poller.poll_page() == (2, 0)
    assert api.calls[-1][1]["offset"] == 22


def test_invalid_unknown_and_foreign_callbacks_do_not_create_votes(store):
    store.register([FIRST])
    api = FakeTelegram([[
        callback(11, "delete"), callback(12, "drop", SECOND),
        callback(13, "drop", chat="-999"), callback(14, "why"),
        {"update_id": 15, "message": {}},
    ]])
    assert FeedbackPoller(api, store, "123", CHAT).poll_page() == (5, 0)
    assert store.cases() == []
    assert store.cursor("123") == 16
    answers = [body for method, body in api.calls if method == "answerCallbackQuery"]
    assert [answer["callback_query_id"] for answer in answers] == [
        "callback-11", "callback-12", "callback-14",
    ]


def test_polling_recovers_after_update_ids_reset(store, monkeypatch):
    times = iter([1000.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(feedback_store.time, "time", lambda: next(times, 1000.0))
    store.register([FIRST])
    FeedbackPoller(FakeTelegram([[callback(900, "drop")]]), store, "123", CHAT).poll_page()
    monkeypatch.setattr(feedback_store.time, "time", lambda: 1000.0 + 7 * 24 * 60 * 60)
    api = FakeTelegram([[callback(20, "keep")]])
    assert FeedbackPoller(api, store, "123", CHAT).poll_page() == (1, 1)
    assert "offset" not in api.calls[0][1]
    assert store.cases()[0]["type"] == "kept"


def test_news_still_sends_without_keyboard_when_register_fails(store, monkeypatch):
    sender = FakeTelegram()
    notifier = notify.TelegramNotifier("fake-token", CHAT)
    notifier._api = sender
    monkeypatch.setattr(notify, "get_feedback_store", lambda: store)

    def unavailable(items):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "register", unavailable)
    notifier.send_news("要闻", pd.DataFrame([
        {"sector": FIRST.sector, "标题": FIRST.title, "原文链接": FIRST.link},
    ]))
    assert "订单增长" in sender.calls[0][1]["text"]
    assert "reply_markup" not in sender.calls[0][1]


def test_agent_tool_adds_linked_review_list(store, monkeypatch):
    sender = FakeTelegram()
    notifier = notify.TelegramNotifier("fake-token", CHAT)
    notifier._api = sender
    monkeypatch.setattr(notify, "get_feedback_store", lambda: store)
    monkeypatch.setattr(notify, "load_notifiers_from_env", lambda: [notifier])
    result = push_telegram(PushTelegramInput(title="日报", content="<b>总览</b>", feedback_items=[vars(FIRST)]))
    assert result["sent"]
    overview, review = [body for _, body in sender.calls]
    assert "<b>总览</b>" in overview["text"]
    assert f'href="{FIRST.link}"' in review["text"]
    assert review["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == f"drop:{FIRST.id}"
    assert store.get_item(FIRST.id) == FIRST


def test_split_messages_keep_buttons_with_their_articles(store, monkeypatch):
    sender = FakeTelegram()
    notifier = notify.TelegramNotifier("fake-token", CHAT)
    notifier._api = sender
    monkeypatch.setattr(notify, "get_feedback_store", lambda: store)
    monkeypatch.setattr(notify, "TELEGRAM_MESSAGE_LIMIT", 110)
    notifier.send_digest("日报", "总览", [FIRST, SECOND])
    _, first, second = [body for _, body in sender.calls]
    for index, (body, item) in enumerate(zip((first, second), (FIRST, SECOND)), 1):
        assert f"{index}. <b>{item.title}</b>" in body["text"]
        assert body["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == f"keep:{item.id}"


def test_reviewed_local_label_takes_precedence_over_telegram(store, tmp_path):
    store.save_vote(CHAT, FIRST, "drop", 1)
    exported = tmp_path / "feedback.jsonl"
    export_cases(store, exported)
    local = tmp_path / "local.jsonl"
    local.write_text(json.dumps({**vars(FIRST), "type": "pending"}), encoding="utf-8")
    assert false_positive_examples(local, before="2026-11-17",
                                   feedback_path=exported) == {FIRST.sector: [FIRST.title]}
    local.write_text(json.dumps({**vars(FIRST), "type": "dismissed"}), encoding="utf-8")
    assert false_positive_examples(local, before="2026-11-17", feedback_path=exported) == {}


def test_export_cli_needs_no_telegram_credentials(store, tmp_path, monkeypatch, capsys):
    output = tmp_path / "exported.jsonl"
    store.save_vote(CHAT, FIRST, "drop", 1)
    monkeypatch.setattr(feedback_bot, "get_feedback_store", lambda **kwargs: store)
    monkeypatch.setattr("sys.argv", ["feedback_bot.py", "--export", "--output", str(output)])
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert feedback_bot.main() == 0
    assert "导出 1 条" in capsys.readouterr().out
    assert load_cases(output)[0]["link"] == FIRST.link


def test_require_supabase_fails_with_actionable_error(monkeypatch, capsys):
    monkeypatch.setattr(feedback_store, "load_supabase_credentials", lambda: None)
    monkeypatch.setattr("sys.argv", ["feedback_bot.py", "--once", "--require-supabase"])
    assert feedback_bot.main() == 1
    assert "SUPABASE_URL / SUPABASE_KEY" in capsys.readouterr().err
