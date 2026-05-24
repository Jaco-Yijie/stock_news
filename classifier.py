from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


Rule = dict[str, Any]
LegacyLLMVerifier = Callable[[dict[str, Any], str], str | dict[str, Any]]

DEFAULT_FIELDS = ["title", "summary", "content"]
LOW_CONFIDENCE_THRESHOLD = 0.75
NEAR_MIN_SCORE_THRESHOLD = 1.25
HIGH_IMPACT_CATEGORIES = {"美联储/利率"}
HIGH_RISK_CATEGORIES = {"苹果产业链", "Vision Pro", "美联储/利率", "美联储利率"}
GENERIC_RISK_KEYWORDS = {"美元", "利率", "Apple", "Pro", "Vision"}
FED_CO_KEYWORDS = ["美联储", "FOMC", "利率", "降息", "加息", "鲍威尔"]
NEVER_MATCH_KEYWORD = "__never_match_keyword__"

FIELD_ALIASES = {
    "title": ("title", "标题"),
    "summary": ("summary", "摘要", "新闻摘要", "新闻内容"),
    "content": ("content", "内容", "新闻内容"),
}


@dataclass(frozen=True)
class ClassificationResult:
    matched: bool
    category: str
    score: float
    confidence: float
    matched_keywords: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class LLMValidationResult:
    should_keep: bool
    category: str
    confidence: float
    reason: str


@runtime_checkable
class LLMVerifier(Protocol):
    provider: str
    model: str
    prompt_version: str
    rule_version: str

    def verify(
        self,
        news_item: dict[str, Any],
        category: str,
        evidence: dict[str, Any],
    ) -> LLMValidationResult | str | dict[str, Any]:
        ...


class LLMValidationCache:
    def __init__(self) -> None:
        self._items: dict[str, LLMValidationResult] = {}

    def get(self, key: str) -> LLMValidationResult | None:
        return self._items.get(key)

    def set(self, key: str, value: LLMValidationResult) -> None:
        self._items[key] = value


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple):
        values = value
    else:
        values = []

    cleaned: list[str] = []
    for item in values:
        keyword = _clean_text(item)
        if keyword and keyword not in cleaned:
            cleaned.append(keyword)
    return cleaned


def _clean_fields(value: Any) -> list[str]:
    fields = _clean_keywords(value)
    return [field for field in fields if field in FIELD_ALIASES] or list(DEFAULT_FIELDS)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _keyword_to_rule(keyword: str) -> Rule:
    keyword = _clean_text(keyword)
    folded = keyword.casefold()

    if folded in {"pro", "vision"}:
        return {
            "positiveKeywords": [keyword],
            "requiredCoKeywords": [NEVER_MATCH_KEYWORD],
            "negativeKeywords": [],
            "weight": 1.0,
            "minScore": 1.0,
            "fields": list(DEFAULT_FIELDS),
        }

    if keyword == "美元":
        return {
            "positiveKeywords": ["美元"],
            "requiredCoKeywords": list(FED_CO_KEYWORDS),
            "negativeKeywords": ["中美元首", "中美关系", "卢比兑美元", "兑美元"],
            "weight": 1.0,
            "minScore": 1.0,
            "fields": list(DEFAULT_FIELDS),
        }

    weight = 2.0 if keyword == "Vision Pro" else 1.0
    return {
        "positiveKeywords": [keyword],
        "requiredCoKeywords": [],
        "negativeKeywords": [],
        "weight": weight,
        "minScore": weight,
        "fields": list(DEFAULT_FIELDS),
    }


def normalize_rule(rule: Any) -> Rule | None:
    if isinstance(rule, str):
        if not _clean_text(rule):
            return None
        return _keyword_to_rule(rule)

    if not isinstance(rule, dict):
        return None

    positive_keywords = _clean_keywords(rule.get("positiveKeywords"))
    if not positive_keywords:
        return None

    weight = _to_float(rule.get("weight"), 1.0)
    min_score = _to_float(rule.get("minScore"), weight)
    return {
        "positiveKeywords": positive_keywords,
        "requiredCoKeywords": _clean_keywords(rule.get("requiredCoKeywords")),
        "negativeKeywords": _clean_keywords(rule.get("negativeKeywords")),
        "weight": weight,
        "minScore": min_score,
        "fields": _clean_fields(rule.get("fields")),
    }


def normalize_event_rules(data: Any) -> dict[str, list[Rule]]:
    if not isinstance(data, dict):
        raise ValueError("事件规则配置必须是对象")

    normalized: dict[str, list[Rule]] = {}
    for category, rules in data.items():
        category_name = _clean_text(category)
        if not category_name:
            continue

        if isinstance(rules, str | dict):
            raw_rules = [rules]
        elif isinstance(rules, list | tuple):
            raw_rules = list(rules)
        else:
            raw_rules = []

        category_rules: list[Rule] = []
        for rule in raw_rules:
            normalized_rule = normalize_rule(rule)
            if normalized_rule is not None:
                category_rules.append(normalized_rule)
        normalized[category_name] = category_rules
    return normalized


def extract_rule_keywords(rules: list[Any]) -> list[str]:
    keywords: list[str] = []
    for rule in rules:
        normalized_rule = normalize_rule(rule)
        if normalized_rule is None:
            continue
        for keyword in normalized_rule["positiveKeywords"]:
            if keyword not in keywords:
                keywords.append(keyword)
    return keywords


def _field_text(news: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        for alias in FIELD_ALIASES[field]:
            text = _clean_text(news.get(alias))
            if text:
                parts.append(text)
                break
    return " ".join(parts).casefold()


def _contains_keyword(text: str, keyword: str) -> bool:
    return _clean_text(keyword).casefold() in text


def classify_news_item(
    news: dict[str, Any],
    category: str,
    rules: list[Any],
) -> ClassificationResult:
    normalized_rules = [
        normalized_rule
        for rule in rules
        if (normalized_rule := normalize_rule(rule)) is not None
    ]
    matched_keywords: list[str] = []
    total_score = 0.0

    for rule in normalized_rules:
        text = _field_text(news, rule["fields"])
        if not text:
            continue

        if any(_contains_keyword(text, keyword) for keyword in rule["negativeKeywords"]):
            continue

        if rule["requiredCoKeywords"] and not any(
            _contains_keyword(text, keyword) for keyword in rule["requiredCoKeywords"]
        ):
            continue

        rule_score = 0.0
        rule_keywords: list[str] = []
        for keyword in rule["positiveKeywords"]:
            if _contains_keyword(text, keyword):
                rule_score += rule["weight"]
                rule_keywords.append(keyword)

        if rule_score >= rule["minScore"]:
            total_score += rule_score
            matched_keywords.extend(rule_keywords)

    if total_score <= 0:
        return ClassificationResult(False, category, 0.0, 0.0)

    confidence = min(1.0, total_score / 2.0)
    return ClassificationResult(
        matched=True,
        category=category,
        score=total_score,
        confidence=confidence,
        matched_keywords=tuple(dict.fromkeys(matched_keywords)),
        reason=f"规则命中：{','.join(dict.fromkeys(matched_keywords))}",
    )


def should_validate_with_llm(
    category: str,
    result: ClassificationResult,
    high_impact_categories: set[str] | None = None,
) -> bool:
    if not result.matched:
        return False
    if high_impact_categories is None:
        high_impact_categories = HIGH_IMPACT_CATEGORIES
    matched_keywords = set(result.matched_keywords)
    generic_risk = bool(matched_keywords & GENERIC_RISK_KEYWORDS)
    high_risk_low_confidence = (
        category in HIGH_RISK_CATEGORIES and result.confidence < LOW_CONFIDENCE_THRESHOLD
    )
    high_impact_generic_risk = category in high_impact_categories and generic_risk
    near_min_score = result.score <= NEAR_MIN_SCORE_THRESHOLD
    return high_risk_low_confidence or high_impact_generic_risk or near_min_score


def build_llm_evidence(
    news: dict[str, Any],
    category: str,
    result: ClassificationResult,
) -> dict[str, Any]:
    matched_keywords = list(result.matched_keywords)
    generic_risk_keywords = [
        keyword for keyword in matched_keywords if keyword in GENERIC_RISK_KEYWORDS
    ]
    return {
        "category": category,
        "score": result.score,
        "confidence": result.confidence,
        "matchedKeywords": matched_keywords,
        "genericRiskKeywords": generic_risk_keywords,
        "reason": result.reason,
        "shouldDropOnFailure": result.confidence < LOW_CONFIDENCE_THRESHOLD
        or result.score <= NEAR_MIN_SCORE_THRESHOLD,
    }


def _verifier_metadata(verifier: Any = None) -> dict[str, str]:
    return {
        "provider": _clean_text(getattr(verifier, "provider", "custom")) or "custom",
        "model": _clean_text(getattr(verifier, "model", "unknown")) or "unknown",
        "prompt_version": _clean_text(getattr(verifier, "prompt_version", "unknown"))
        or "unknown",
        "rule_version": _clean_text(getattr(verifier, "rule_version", "unknown"))
        or "unknown",
    }


def _validation_cache_key(
    news: dict[str, Any],
    category: str,
    verifier: Any = None,
) -> str:
    metadata = _verifier_metadata(verifier)
    payload = {
        "category": category,
        **metadata,
        "title": _clean_text(news.get("title") or news.get("标题")),
        "link": _clean_text(news.get("link") or news.get("原文链接")),
        "content": _clean_text(news.get("content") or news.get("新闻内容"))[:500],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_llm_validation(value: str | dict[str, Any] | LLMValidationResult) -> LLMValidationResult:
    if isinstance(value, LLMValidationResult):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value

    if not isinstance(parsed, dict):
        raise ValueError("LLM 返回内容必须是 JSON 对象")

    required_keys = {"shouldKeep", "category", "confidence", "reason"}
    missing_keys = required_keys - set(parsed)
    if missing_keys:
        raise ValueError(f"LLM 返回 JSON 缺少字段：{', '.join(sorted(missing_keys))}")

    should_keep = parsed["shouldKeep"]
    if not isinstance(should_keep, bool):
        raise ValueError("LLM 返回 shouldKeep 必须是布尔值")

    category = parsed["category"]
    if not isinstance(category, str):
        raise ValueError("LLM 返回 category 必须是字符串")

    confidence_value = parsed["confidence"]
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, int | float):
        raise ValueError("LLM 返回 confidence 必须是数字")
    confidence = float(confidence_value)
    if confidence < 0 or confidence > 1:
        raise ValueError("LLM 返回 confidence 必须在 0 到 1 之间")

    reason = parsed["reason"]
    if not isinstance(reason, str):
        raise ValueError("LLM 返回 reason 必须是字符串")

    return LLMValidationResult(
        should_keep=should_keep,
        category=_clean_text(category),
        confidence=confidence,
        reason=_clean_text(reason),
    )


def _run_llm_verifier(
    verifier: LLMVerifier | LegacyLLMVerifier,
    news: dict[str, Any],
    category: str,
    evidence: dict[str, Any],
) -> LLMValidationResult | str | dict[str, Any]:
    if hasattr(verifier, "verify"):
        return verifier.verify(news, category, evidence)
    return verifier(news, category)


def validate_classification_with_llm(
    news: dict[str, Any],
    category: str,
    result: ClassificationResult,
    verifier: LLMVerifier | LegacyLLMVerifier | None = None,
    cache: LLMValidationCache | None = None,
) -> LLMValidationResult:
    if not should_validate_with_llm(category, result) or verifier is None:
        return LLMValidationResult(True, category, result.confidence, result.reason)

    evidence = build_llm_evidence(news, category, result)
    cache_key = _validation_cache_key(news, category, verifier)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, LLMValidationResult):
            return cached

    try:
        validation = parse_llm_validation(_run_llm_verifier(verifier, news, category, evidence))
    except Exception as exc:
        should_keep = not bool(evidence.get("shouldDropOnFailure"))
        validation = LLMValidationResult(
            should_keep,
            category,
            result.confidence if should_keep else 0.0,
            f"LLM 校验 JSON 解析失败或调用失败：{exc}",
        )

    if cache is not None:
        cache.set(cache_key, validation)
    return validation
