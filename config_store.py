from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from classifier import extract_rule_keywords, normalize_event_rules, normalize_rule
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


def extract_sector_keywords(value: Any) -> list[str]:
    if isinstance(value, dict):
        keywords = value.get("keywords", value.get("KEYWORDS", []))
        return _clean_keywords(keywords if isinstance(keywords, list | tuple) else [keywords])
    if isinstance(value, str):
        return _clean_keywords([value])
    if isinstance(value, list | tuple):
        return _clean_keywords(value)
    return []


def extract_sector_rule_id(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("rule_id", "classification_rule", "risk_filter", "category_key"):
        rule_id = _clean_name(value.get(key, ""))
        if rule_id:
            return rule_id
    return ""


def _normalize_sector_value(value: Any) -> list[str] | dict[str, Any]:
    keywords = extract_sector_keywords(value)
    rule_id = extract_sector_rule_id(value)
    if isinstance(value, dict) or rule_id:
        normalized: dict[str, Any] = {"keywords": keywords}
        if rule_id:
            normalized["rule_id"] = rule_id
        return normalized
    return keywords


def _normalize_sectors_payload(data: Any) -> dict[str, list[str] | dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("板块配置内容必须是对象")

    normalized: dict[str, list[str] | dict[str, Any]] = {}
    for name, value in data.items():
        clean_name = _clean_name(name)
        if clean_name:
            normalized[clean_name] = _normalize_sector_value(value)
    return normalized


def _replace_sector_keywords(value: Any, keywords: list[str]) -> list[str] | dict[str, Any]:
    normalized_keywords = _clean_keywords(keywords)
    rule_id = extract_sector_rule_id(value)
    if isinstance(value, dict) or rule_id:
        updated: dict[str, Any] = {"keywords": normalized_keywords}
        if rule_id:
            updated["rule_id"] = rule_id
        return updated
    return normalized_keywords


def _normalize_events_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("事件配置内容必须是对象")

    external_events = data.get("external_events", data.get("EXTERNAL_EVENTS", {}))
    event_to_sectors = data.get("event_to_sectors", data.get("EVENT_TO_SECTORS", {}))
    return {
        "external_events": normalize_event_rules(external_events),
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


def default_sectors_config() -> dict[str, list[str] | dict[str, Any]]:
    return deepcopy(SECTORS)


def default_events_config() -> dict[str, Any]:
    return {
        "external_events": deepcopy(EXTERNAL_EVENTS),
        "event_to_sectors": deepcopy(EVENT_TO_SECTORS),
    }


def load_sectors_config() -> dict[str, list[str] | dict[str, Any]]:
    if not SECTORS_CONFIG_PATH.exists():
        sectors = default_sectors_config()
        save_sectors_config(sectors)
        return sectors

    data = json.loads(SECTORS_CONFIG_PATH.read_text(encoding="utf-8"))
    return _normalize_sectors_payload(data)


def try_load_sectors_config() -> tuple[dict[str, list[str] | dict[str, Any]], str | None]:
    try:
        return load_sectors_config(), None
    except Exception as exc:
        return default_sectors_config(), f"读取板块关键词配置失败，已使用默认配置：{type(exc).__name__}"


def save_sectors_config(sectors: dict[str, Any]) -> None:
    _write_json(SECTORS_CONFIG_PATH, _normalize_sectors_payload(sectors))


def reset_sectors_config() -> dict[str, list[str] | dict[str, Any]]:
    sectors = default_sectors_config()
    save_sectors_config(sectors)
    return sectors


def add_keyword(sector_name: str, keyword: str) -> dict[str, list[str] | dict[str, Any]]:
    sector_name = _clean_name(sector_name)
    keyword = _clean_name(keyword)
    if not sector_name:
        raise ValueError("板块名称不能为空")
    if not keyword:
        raise ValueError("关键词不能为空")

    sectors = load_sectors_config()
    sectors.setdefault(sector_name, [])
    keywords = extract_sector_keywords(sectors[sector_name])
    if keyword not in keywords:
        keywords.append(keyword)
    sectors[sector_name] = _replace_sector_keywords(sectors[sector_name], keywords)
    save_sectors_config(sectors)
    return sectors


def remove_keyword(sector_name: str, keyword: str) -> dict[str, list[str] | dict[str, Any]]:
    sector_name = _clean_name(sector_name)
    keyword = _clean_name(keyword)
    sectors = load_sectors_config()
    if sector_name in sectors:
        sectors[sector_name] = _replace_sector_keywords(
            sectors[sector_name],
            [item for item in extract_sector_keywords(sectors[sector_name]) if item != keyword],
        )
    save_sectors_config(sectors)
    return sectors


def add_sector(sector_name: str, keywords: list[str]) -> dict[str, list[str] | dict[str, Any]]:
    sector_name = _clean_name(sector_name)
    if not sector_name:
        raise ValueError("板块名称不能为空")

    sectors = load_sectors_config()
    existing_value = sectors.get(sector_name, [])
    sectors[sector_name] = _replace_sector_keywords(
        existing_value,
        [*extract_sector_keywords(existing_value), *keywords],
    )
    save_sectors_config(sectors)
    return sectors


def remove_sector(sector_name: str) -> dict[str, list[str] | dict[str, Any]]:
    sector_name = _clean_name(sector_name)
    sectors = load_sectors_config()
    sectors.pop(sector_name, None)
    save_sectors_config(sectors)
    return sectors


def load_events_config() -> dict[str, Any]:
    if not EVENTS_CONFIG_PATH.exists():
        events_config = default_events_config()
        save_events_config(
            events_config["external_events"],
            events_config["event_to_sectors"],
        )
        return events_config

    data = json.loads(EVENTS_CONFIG_PATH.read_text(encoding="utf-8"))
    return _normalize_events_payload(data)


def try_load_events_config() -> tuple[dict[str, Any], str | None]:
    try:
        return load_events_config(), None
    except Exception as exc:
        return default_events_config(), f"读取外部事件配置失败，已使用默认配置：{type(exc).__name__}"


def save_events_config(
    external_events: dict[str, list[Any]],
    event_to_sectors: dict[str, list[str]],
) -> None:
    payload = {
        "external_events": normalize_event_rules(external_events),
        "event_to_sectors": _normalize_mapping(event_to_sectors),
    }
    _write_json(EVENTS_CONFIG_PATH, payload)


def reset_events_config() -> dict[str, Any]:
    events_config = default_events_config()
    save_events_config(
        events_config["external_events"],
        events_config["event_to_sectors"],
    )
    return events_config


def add_event_keyword(event_category: str, keyword: str) -> dict[str, Any]:
    event_category = _clean_name(event_category)
    keyword = _clean_name(keyword)
    if not event_category:
        raise ValueError("事件类别不能为空")
    if not keyword:
        raise ValueError("事件关键词不能为空")

    events_config = load_events_config()
    external_events = events_config["external_events"]
    external_events.setdefault(event_category, [])
    if keyword not in extract_rule_keywords(external_events[event_category]):
        external_events[event_category].extend(
            normalize_event_rules({event_category: [keyword]}).get(event_category, [])
        )
    save_events_config(external_events, events_config["event_to_sectors"])
    return events_config


def remove_event_keyword(event_category: str, keyword: str) -> dict[str, Any]:
    event_category = _clean_name(event_category)
    keyword = _clean_name(keyword)
    events_config = load_events_config()
    external_events = events_config["external_events"]
    if event_category in external_events:
        updated_rules: list[dict[str, Any]] = []
        for item in external_events[event_category]:
            rule = normalize_rule(item)
            if rule is None:
                continue
            rule["positiveKeywords"] = [
                positive_keyword
                for positive_keyword in rule["positiveKeywords"]
                if positive_keyword != keyword
            ]
            if rule["positiveKeywords"]:
                updated_rules.append(rule)
        external_events[event_category] = updated_rules
    save_events_config(external_events, events_config["event_to_sectors"])
    return events_config


def add_event_category(
    event_category: str,
    keywords: list[str],
    related_sectors: list[str] | None = None,
) -> dict[str, Any]:
    event_category = _clean_name(event_category)
    if not event_category:
        raise ValueError("事件类别不能为空")

    events_config = load_events_config()
    external_events = events_config["external_events"]
    event_to_sectors = events_config["event_to_sectors"]
    external_events[event_category] = normalize_event_rules(
        {event_category: [*external_events.get(event_category, []), *keywords]}
    ).get(event_category, [])
    if related_sectors is not None:
        event_to_sectors[event_category] = _clean_keywords(related_sectors)
    save_events_config(external_events, event_to_sectors)
    return events_config


def remove_event_category(event_category: str) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
