from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_TIMEOUT = 20
PAGE_SIZE = 1000
UPSERT_CHUNK_SIZE = 500
DELETE_CHUNK_SIZE = 100
NEWS_TABLE = "news_cache"
ID_KEY_COLUMNS = ("news_type", "sector", "event_category", "link", "title", "keyword")

# 一次全量读取要翻十几页，每页都新建连接的话光 TCP+TLS 握手就是十几个来回。
# 进程内共享一个 Session，让分页之间（以及 Streamlit 各次 rerun 之间）复用连接。
_session_lock = threading.Lock()
_shared_session: requests.Session | None = None


def get_shared_session() -> requests.Session:
    global _shared_session
    with _session_lock:
        if _shared_session is None:
            session = requests.Session()
            # 只对 GET 重试：读取是幂等的，写入交给调用方处理，避免重复写。
            retry = Retry(
                total=2,
                connect=2,
                read=0,
                status=2,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                backoff_factor=0.3,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _shared_session = session
        return _shared_session


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


class _SupabaseTable:
    """Supabase PostgREST 单表客户端基类。

    需要使用 service_role key（表已开启 RLS 时 anon key 无法读写），
    key 只在服务端使用，不会暴露给浏览器。
    """

    def __init__(
        self,
        url: str,
        key: str,
        table: str,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        self._timeout = timeout
        self._session = session or get_shared_session()

    def _request(
        self,
        method: str,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        merged_headers = {**self._headers, **(headers or {})}
        try:
            response = self._session.request(
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

class SupabaseNewsStore(_SupabaseTable):
    """读写 news_cache 表。"""

    def __init__(
        self,
        url: str,
        key: str,
        table: str = NEWS_TABLE,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(url, key, table, timeout, session)

    def _fetch_paged(self, select: str) -> list[dict[str, Any]]:
        """按 id 游标翻页读取整表。

        用 `id=gt.<上一页末尾>` 而不是 `offset=N`：深 offset 要求数据库
        先扫过并丢弃前面所有行，页数越多越慢；游标翻页每页都走主键索引。
        select 必须包含 id，否则无法定位游标。
        """
        rows: list[dict[str, Any]] = []
        last_id = ""
        # 服务端可能把 limit 压到比 PAGE_SIZE 更小（PostgREST 的 max-rows），
        # 所以用"实际见过的最大页"判断是否到底，避免被截断成半个缓存。
        observed_page_size = 0
        while True:
            params = {
                "select": select,
                "order": "id.asc",
                "limit": str(PAGE_SIZE),
            }
            if last_id:
                params["id"] = f"gt.{last_id}"
            response = self._request("GET", params=params)
            page = response.json()
            if not isinstance(page, list):
                raise SupabaseError("Supabase 返回内容不是列表")
            if not page:
                return rows

            rows.extend(page)
            observed_page_size = max(observed_page_size, len(page))
            last_id = str(page[-1].get("id", "") or "")
            if not last_id:
                raise SupabaseError("Supabase 分页返回缺少 id 字段，无法继续翻页")
            if len(page) < observed_page_size:
                return rows

    def fetch_all(self) -> list[dict[str, Any]]:
        return self._fetch_paged("*")

    def latest_fetched_at(self) -> str:
        response = self._request(
            "GET",
            params={
                "select": "fetched_at",
                "order": "fetched_at.desc",
                "limit": "1",
            },
        )
        rows = response.json()
        if isinstance(rows, list) and rows:
            return str(rows[0].get("fetched_at", ""))
        return ""

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


CONFIG_TABLE = "app_config"


class SupabaseConfigStore(_SupabaseTable):
    """读写 app_config 表（key -> JSON 文本），用于持久化板块/事件配置。"""

    def __init__(
        self,
        url: str,
        key: str,
        table: str = CONFIG_TABLE,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(url, key, table, timeout, session)

    def get_value(self, config_key: str) -> str | None:
        response = self._request(
            "GET",
            params={"select": "value", "key": f"eq.{config_key}", "limit": "1"},
        )
        rows = response.json()
        if isinstance(rows, list) and rows:
            return str(rows[0].get("value", "") or "") or None
        return None

    def set_value(self, config_key: str, value: str) -> None:
        self._request(
            "POST",
            json_body=[{"key": config_key, "value": value}],
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
