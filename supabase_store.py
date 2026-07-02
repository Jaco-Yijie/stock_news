from __future__ import annotations

import hashlib
import os
from typing import Any

import requests


DEFAULT_TIMEOUT = 20
PAGE_SIZE = 1000
UPSERT_CHUNK_SIZE = 500
DELETE_CHUNK_SIZE = 100
NEWS_TABLE = "news_cache"
ID_KEY_COLUMNS = ("news_type", "sector", "event_category", "link", "title", "keyword")


class SupabaseError(RuntimeError):
    pass


def _sanitize_credential(value: str) -> str:
    # 粘贴长 key 时容易混入换行/空格，HTTP 头里不允许这些字符；
    # URL 和 JWT 本身都不含空白字符，直接全部移除是安全的。
    return "".join(str(value or "").split())


def load_supabase_credentials() -> tuple[str, str] | None:
    url = _sanitize_credential(os.getenv("SUPABASE_URL", ""))
    key = _sanitize_credential(os.getenv("SUPABASE_KEY", ""))
    if not url or not key:
        try:
            import streamlit as st

            url = url or _sanitize_credential(st.secrets.get("SUPABASE_URL", ""))
            key = key or _sanitize_credential(st.secrets.get("SUPABASE_KEY", ""))
        except Exception:
            pass
    if url and key:
        return url, key
    return None


def news_row_id(row: dict[str, Any]) -> str:
    payload = "|".join(str(row.get(column, "")) for column in ID_KEY_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SupabaseNewsStore:
    """通过 Supabase PostgREST API 读写 news_cache 表。

    需要使用 service_role key（表已开启 RLS 时 anon key 无法读写），
    key 只在服务端使用，不会暴露给浏览器。
    """

    def __init__(
        self,
        url: str,
        key: str,
        table: str = NEWS_TABLE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        self._timeout = timeout

    def _request(
        self,
        method: str,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        merged_headers = {**self._headers, **(headers or {})}
        try:
            response = requests.request(
                method,
                self._endpoint,
                params=params,
                json=json_body,
                headers=merged_headers,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise SupabaseError(f"Supabase 请求失败：{exc}") from exc

        if response.status_code >= 400:
            raise SupabaseError(
                f"Supabase 返回错误 {response.status_code}：{response.text[:200]}"
            )
        return response

    def _fetch_paged(self, select: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = self._request(
                "GET",
                params={
                    "select": select,
                    "order": "id.asc",
                    "limit": str(PAGE_SIZE),
                    "offset": str(offset),
                },
            )
            page = response.json()
            if not isinstance(page, list):
                raise SupabaseError("Supabase 返回内容不是列表")
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def fetch_all(self) -> list[dict[str, Any]]:
        return self._fetch_paged("*")

    def fetch_ids(self) -> set[str]:
        return {
            str(row.get("id", ""))
            for row in self._fetch_paged("id")
            if row.get("id")
        }

    def replace_all(self, rows: list[dict[str, Any]]) -> None:
        new_ids: set[str] = set()
        deduped_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = news_row_id(row)
            deduped_rows[row_id] = {**row, "id": row_id}
        new_ids = set(deduped_rows)

        payload = list(deduped_rows.values())
        # 先写入新数据再删除过期行，中途失败时不会丢掉整个缓存
        for start in range(0, len(payload), UPSERT_CHUNK_SIZE):
            chunk = payload[start : start + UPSERT_CHUNK_SIZE]
            self._request(
                "POST",
                json_body=chunk,
                headers={
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
            )

        stale_ids = sorted(self.fetch_ids() - new_ids)
        for start in range(0, len(stale_ids), DELETE_CHUNK_SIZE):
            chunk = stale_ids[start : start + DELETE_CHUNK_SIZE]
            self._request("DELETE", params={"id": f"in.({','.join(chunk)})"})

    def delete_all(self) -> None:
        self._request("DELETE", params={"id": "neq."})
