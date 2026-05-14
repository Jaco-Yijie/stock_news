from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from paths import DATA_DIR
from sectors import EVENT_TO_SECTORS, EXTERNAL_EVENTS, SECTORS


SECTORS_CONFIG_PATH = DATA_DIR / "sectors_config.json"
EVENTS_CONFIG_PATH = DATA_DIR / "events_config.json"


def _clean_name(value: str) -> str:
    return str(value or "").strip()


def _clean_keywords(keywords: list[str] | tuple[str, ...] | None) -> list[str]:
    cleaned: list[str] = []
    for keyword in keywords or []:
        item = _clean_name(keyword)
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _normalize_mapping(data: Any) -> dict[str, list[str]]:
    if not isinstance(data, dict):
        raise ValueError("配置内容必须是对象")

    normalized: dict[str, list[str]] = {}
    for name, keywords in data.items():
        clean_name = _clean_name(name)
        if not clean_name:
            continue
        if isinstance(keywords, str):
            keyword_list = [keywords]
        elif isinstance(keywords, list | tuple):
            keyword_list = list(keywords)
        else:
            keyword_list = []
        normalized[clean_name] = _clean_keywords(keyword_list)
    return normalized


def _normalize_events_payload(data: Any) -> dict[str, dict[str, list[str]]]:
    if not isinstance(data, dict):
        raise ValueError("事件配置内容必须是对象")

    external_events = data.get("external_events", data.get("EXTERNAL_EVENTS", {}))
    event_to_sectors = data.get("event_to_sectors", data.get("EVENT_TO_SECTORS", {}))
    return {
        "external_events": _normalize_mapping(external_events),
        "event_to_sectors": _normalize_mapping(event_to_sectors),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def default_sectors_config() -> dict[str, list[str]]:
    return deepcopy(SECTORS)


def default_events_config() -> dict[str, dict[str, list[str]]]:
    return {
        "external_events": deepcopy(EXTERNAL_EVENTS),
        "event_to_sectors": deepcopy(EVENT_TO_SECTORS),
    }


def load_sectors_config() -> dict[str, list[str]]:
    if not SECTORS_CONFIG_PATH.exists():
        sectors = default_sectors_config()
        save_sectors_config(sectors)
        return sectors

    data = json.loads(SECTORS_CONFIG_PATH.read_text(encoding="utf-8"))
    return _normalize_mapping(data)


def try_load_sectors_config() -> tuple[dict[str, list[str]], str | None]:
    try:
        return load_sectors_config(), None
    except Exception as exc:
        return default_sectors_config(), f"读取板块关键词配置失败，已使用默认配置：{exc}"


def save_sectors_config(sectors: dict[str, list[str]]) -> None:
    _write_json(SECTORS_CONFIG_PATH, _normalize_mapping(sectors))


def reset_sectors_config() -> dict[str, list[str]]:
    sectors = default_sectors_config()
    save_sectors_config(sectors)
    return sectors


def add_keyword(sector_name: str, keyword: str) -> dict[str, list[str]]:
    sector_name = _clean_name(sector_name)
    keyword = _clean_name(keyword)
    if not sector_name:
        raise ValueError("板块名称不能为空")
    if not keyword:
        raise ValueError("关键词不能为空")

    sectors = load_sectors_config()
    sectors.setdefault(sector_name, [])
    if keyword not in sectors[sector_name]:
        sectors[sector_name].append(keyword)
    save_sectors_config(sectors)
    return sectors


def remove_keyword(sector_name: str, keyword: str) -> dict[str, list[str]]:
    sector_name = _clean_name(sector_name)
    keyword = _clean_name(keyword)
    sectors = load_sectors_config()
    if sector_name in sectors:
        sectors[sector_name] = [item for item in sectors[sector_name] if item != keyword]
    save_sectors_config(sectors)
    return sectors


def add_sector(sector_name: str, keywords: list[str]) -> dict[str, list[str]]:
    sector_name = _clean_name(sector_name)
    if not sector_name:
        raise ValueError("板块名称不能为空")

    sectors = load_sectors_config()
    existing_keywords = sectors.get(sector_name, [])
    sectors[sector_name] = _clean_keywords([*existing_keywords, *keywords])
    save_sectors_config(sectors)
    return sectors


def remove_sector(sector_name: str) -> dict[str, list[str]]:
    sector_name = _clean_name(sector_name)
    sectors = load_sectors_config()
    sectors.pop(sector_name, None)
    save_sectors_config(sectors)
    return sectors


def load_events_config() -> dict[str, dict[str, list[str]]]:
    if not EVENTS_CONFIG_PATH.exists():
        events_config = default_events_config()
        save_events_config(
            events_config["external_events"],
            events_config["event_to_sectors"],
        )
        return events_config

    data = json.loads(EVENTS_CONFIG_PATH.read_text(encoding="utf-8"))
    return _normalize_events_payload(data)


def try_load_events_config() -> tuple[dict[str, dict[str, list[str]]], str | None]:
    try:
        return load_events_config(), None
    except Exception as exc:
        return default_events_config(), f"读取外部事件配置失败，已使用默认配置：{exc}"


def save_events_config(
    external_events: dict[str, list[str]],
    event_to_sectors: dict[str, list[str]],
) -> None:
    payload = {
        "external_events": _normalize_mapping(external_events),
        "event_to_sectors": _normalize_mapping(event_to_sectors),
    }
    _write_json(EVENTS_CONFIG_PATH, payload)


def reset_events_config() -> dict[str, dict[str, list[str]]]:
    events_config = default_events_config()
    save_events_config(
        events_config["external_events"],
        events_config["event_to_sectors"],
    )
    return events_config


def add_event_keyword(event_category: str, keyword: str) -> dict[str, dict[str, list[str]]]:
    event_category = _clean_name(event_category)
    keyword = _clean_name(keyword)
    if not event_category:
        raise ValueError("事件类别不能为空")
    if not keyword:
        raise ValueError("事件关键词不能为空")

    events_config = load_events_config()
    external_events = events_config["external_events"]
    external_events.setdefault(event_category, [])
    if keyword not in external_events[event_category]:
        external_events[event_category].append(keyword)
    save_events_config(external_events, events_config["event_to_sectors"])
    return events_config


def remove_event_keyword(event_category: str, keyword: str) -> dict[str, dict[str, list[str]]]:
    event_category = _clean_name(event_category)
    keyword = _clean_name(keyword)
    events_config = load_events_config()
    external_events = events_config["external_events"]
    if event_category in external_events:
        external_events[event_category] = [
            item for item in external_events[event_category] if item != keyword
        ]
    save_events_config(external_events, events_config["event_to_sectors"])
    return events_config


def add_event_category(
    event_category: str,
    keywords: list[str],
    related_sectors: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    event_category = _clean_name(event_category)
    if not event_category:
        raise ValueError("事件类别不能为空")

    events_config = load_events_config()
    external_events = events_config["external_events"]
    event_to_sectors = events_config["event_to_sectors"]
    external_events[event_category] = _clean_keywords(
        [*external_events.get(event_category, []), *keywords]
    )
    if related_sectors is not None:
        event_to_sectors[event_category] = _clean_keywords(related_sectors)
    save_events_config(external_events, event_to_sectors)
    return events_config


def remove_event_category(event_category: str) -> dict[str, dict[str, list[str]]]:
    event_category = _clean_name(event_category)
    events_config = load_events_config()
    external_events = events_config["external_events"]
    event_to_sectors = events_config["event_to_sectors"]
    external_events.pop(event_category, None)
    event_to_sectors.pop(event_category, None)
    save_events_config(external_events, event_to_sectors)
    return events_config


def update_event_related_sectors(
    event_category: str,
    related_sectors: list[str],
) -> dict[str, dict[str, list[str]]]:
    event_category = _clean_name(event_category)
    if not event_category:
        raise ValueError("事件类别不能为空")

    events_config = load_events_config()
    events_config["event_to_sectors"][event_category] = _clean_keywords(related_sectors)
    save_events_config(
        events_config["external_events"],
        events_config["event_to_sectors"],
    )
    return events_config
