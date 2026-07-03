from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


UTC = timezone.utc
UTC8 = timezone(timedelta(hours=8))


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_utc8_naive() -> datetime:
    """当前北京时间（去掉时区信息，便于与新闻发布时间直接比较）。"""
    return datetime.now(UTC8).replace(tzinfo=None)


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text or text == "无":
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for pattern in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, pattern)
                    break
                except ValueError:
                    continue
            else:
                return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC8)


def format_utc8_time(value: Any, suffix: bool = True) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        text = str(value or "").strip()
        return text or "无"

    formatted = parsed.strftime("%Y-%m-%d %H:%M:%S")
    if suffix:
        return f"{formatted} UTC+8"
    return formatted
