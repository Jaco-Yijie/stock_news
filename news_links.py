"""外部新闻链接在 HTML 和消息渲染前共用的校验。"""
from __future__ import annotations

from urllib.parse import urlsplit


def safe_news_url(value: object) -> str:
    url = str(value or "").strip()
    if not url or any(char.isspace() or ord(char) < 32 or char in '\\<>"' for char in url):
        return ""
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
            return ""
        _ = parts.port
    except ValueError:
        return ""
    return url
