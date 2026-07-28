"""Supabase 传输层测试：游标翻页与连接复用。"""
from __future__ import annotations

from typing import Any

import supabase_store
from supabase_store import PAGE_SIZE, SupabaseError, SupabaseNewsStore, get_shared_session


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """按 id 游标返回分页数据，并记录每次请求的参数。"""

    def __init__(self, rows: list[dict[str, Any]], server_cap: int = PAGE_SIZE) -> None:
        self.rows = sorted(rows, key=lambda row: row["id"])
        self.server_cap = server_cap
        self.calls: list[dict[str, str]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs.get("params") or {}
        self.calls.append(dict(params))
        cursor = params.get("id", "")
        candidates = self.rows
        if cursor.startswith("gt."):
            last_id = cursor[len("gt.") :]
            candidates = [row for row in self.rows if row["id"] > last_id]
        limit = min(int(params.get("limit", PAGE_SIZE)), self.server_cap)
        return FakeResponse(candidates[:limit])


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"id": f"{index:06d}", "title": f"新闻{index}"} for index in range(count)]


def _store(session: FakeSession) -> SupabaseNewsStore:
    return SupabaseNewsStore("https://example.supabase.co", "test-key", session=session)


def test_fetch_all_uses_id_cursor_not_offset() -> None:
    session = FakeSession(_rows(2500))
    fetched = _store(session).fetch_all()

    assert len(fetched) == 2500
    assert [row["id"] for row in fetched] == [row["id"] for row in _rows(2500)]
    # 关键：深 offset 会让数据库扫过并丢弃前面所有行，这里必须改用主键游标
    assert all("offset" not in call for call in session.calls)
    assert session.calls[0].get("id") is None
    assert session.calls[1]["id"] == "gt.000999"
    assert session.calls[2]["id"] == "gt.001999"


def test_fetch_all_survives_server_side_row_cap() -> None:
    """服务端把 limit 压到 300 时不能只拿回第一页就当成全部。"""
    session = FakeSession(_rows(1000), server_cap=300)
    fetched = _store(session).fetch_all()

    assert len(fetched) == 1000


def test_fetch_all_stops_on_short_page() -> None:
    session = FakeSession(_rows(1500))
    fetched = _store(session).fetch_all()

    assert len(fetched) == 1500
    # 1000 + 500，第二页不足一页即到底，不再多发请求
    assert len(session.calls) == 2


def test_fetch_all_handles_empty_table() -> None:
    session = FakeSession([])
    assert _store(session).fetch_all() == []
    assert len(session.calls) == 1


def test_fetch_paged_without_id_raises_clear_error() -> None:
    session = FakeSession([{"id": "a", "title": "x"}])

    def request_without_id(method: str, url: str, **kwargs: Any) -> FakeResponse:
        session.calls.append(dict(kwargs.get("params") or {}))
        return FakeResponse([{"title": "x"}])

    session.request = request_without_id  # type: ignore[method-assign]
    try:
        _store(session).fetch_all()
    except SupabaseError as exc:
        assert "id" in str(exc)
    else:
        raise AssertionError("缺少 id 字段时应当报错而不是静默死循环")


def test_stores_share_one_session_for_connection_reuse() -> None:
    supabase_store._shared_session = None
    first = SupabaseNewsStore("https://example.supabase.co", "k")
    second = SupabaseNewsStore("https://example.supabase.co", "k")

    assert first._session is second._session
    assert first._session is get_shared_session()


def test_shared_session_retries_only_get() -> None:
    supabase_store._shared_session = None
    adapter = get_shared_session().get_adapter("https://example.supabase.co")
    allowed = adapter.max_retries.allowed_methods

    assert "GET" in allowed
    # 写入不能自动重试，否则可能重复写库
    assert "POST" not in allowed
    assert "DELETE" not in allowed


if __name__ == "__main__":
    test_fetch_all_uses_id_cursor_not_offset()
    test_fetch_all_survives_server_side_row_cap()
    test_fetch_all_stops_on_short_page()
    test_fetch_all_handles_empty_table()
    test_fetch_paged_without_id_raises_clear_error()
    test_stores_share_one_session_for_connection_reuse()
    test_shared_session_retries_only_get()
    print("test_supabase_store.py: ok")
