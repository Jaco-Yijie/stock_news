from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import pandas as pd

import fetcher
from classifier import (
    LLMValidationCache,
    classify_news_item,
    normalize_event_rules,
    validate_classification_with_llm,
    _validation_cache_key,
)
from config_store import (
    _normalize_sectors_payload,
    _replace_sector_keywords,
    extract_sector_keywords,
    extract_sector_rule_id,
)
from news_store import refilter_external_event_cache
from sectors import EXTERNAL_EVENTS
from llm_provider import (
    ChatCompletionsLLMVerifier,
    LLMProviderConfig,
    load_llm_verifier_from_env,
)


EVENT_RULES = normalize_event_rules(EXTERNAL_EVENTS)


class CountingVerifier:
    def __init__(
        self,
        response: str | dict[str, Any],
        provider: str = "test",
        model: str = "test-model",
        prompt_version: str = "test-prompt",
        rule_version: str = "test-rule",
    ) -> None:
        self.response = response
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version
        self.rule_version = rule_version
        self.calls = 0
        self.last_evidence: dict[str, Any] | None = None

    def verify(
        self,
        news_item: dict[str, Any],
        category: str,
        evidence: dict[str, Any],
    ) -> str | dict[str, Any]:
        self.calls += 1
        self.last_evidence = evidence
        return self.response


class FakeResponse:
    def __init__(self, content: str | dict[str, Any]) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        if isinstance(self.content, dict):
            return self.content
        return {"choices": [{"message": {"content": self.content}}]}


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _chat_verifier(
    provider: str,
    content: str,
) -> tuple[ChatCompletionsLLMVerifier, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(
            {
                "url": url,
                "model": kwargs["json"]["model"],
                "timeout": kwargs["timeout"],
                "payload": kwargs["json"],
            }
        )
        return FakeResponse(content)

    config = LLMProviderConfig(
        provider=provider,
        api_key="test-secret",
        endpoint=f"https://example.com/{provider}",
        model=f"{provider}-model",
        timeout=1,
    )
    return ChatCompletionsLLMVerifier(config, post=fake_post), calls


def _news(title: str, content: str = "") -> dict[str, str]:
    return {
        "标题": title,
        "新闻内容": content,
        "原文链接": f"https://example.com/{abs(hash(title))}",
    }


def _matched(category: str, title: str, content: str = "") -> bool:
    result = classify_news_item(_news(title, content), category, EVENT_RULES[category])
    return result.matched


def test_sector_config_compatibility() -> None:
    old_config = _normalize_sectors_payload({"半导体芯片": ["半导体", "芯片"]})
    new_config = _normalize_sectors_payload(
        {"苹果概念": {"keywords": ["Apple"], "rule_id": "apple_supply_chain"}}
    )

    assert extract_sector_keywords(old_config["半导体芯片"]) == ["半导体", "芯片"]
    assert extract_sector_rule_id(old_config["半导体芯片"]) == ""
    assert extract_sector_keywords(new_config["苹果概念"]) == ["Apple"]
    assert extract_sector_rule_id(new_config["苹果概念"]) == "apple_supply_chain"
    assert "apple_supply_chain" not in extract_sector_keywords(new_config["苹果概念"])


def test_sector_keyword_update_preserves_rule_id() -> None:
    config_value = {"keywords": ["Apple"], "rule_id": "apple_supply_chain"}

    added = _replace_sector_keywords(config_value, ["Apple", "iPhone"])
    removed = _replace_sector_keywords(added, ["iPhone"])

    assert extract_sector_keywords(added) == ["Apple", "iPhone"]
    assert extract_sector_rule_id(added) == "apple_supply_chain"
    assert extract_sector_keywords(removed) == ["iPhone"]
    assert extract_sector_rule_id(removed) == "apple_supply_chain"


def test_unknown_rule_id_does_not_fallback_to_sector_name() -> None:
    assert not fetcher.high_risk_sector_rules(
        "苹果产业链",
        {"keywords": ["Apple"], "rule_id": "unknown_rule"},
    )
    assert fetcher.high_risk_sector_rules("苹果产业链", ["Apple"])


def test_regression_false_positives() -> None:
    assert not _matched("苹果产业链", "海马汽车连续亏损")
    assert not _matched("美联储/利率", "中美元首会谈结束")
    assert not _matched("美联储/利率", "印度卢比兑美元跌破")


def test_vision_pro_requires_complete_phrase() -> None:
    assert _matched("苹果产业链", "苹果发布 Vision Pro 新版本")
    assert not _matched("苹果产业链", "新品 Pro 版本即将发布")
    assert not _matched("苹果产业链", "Vision 概念产品更新")


def test_high_confidence_normal_news_skips_llm() -> None:
    calls = 0
    news = _news("苹果发布 Vision Pro 新版本")
    result = classify_news_item(news, "苹果产业链", EVENT_RULES["苹果产业链"])

    def verifier(payload: dict[str, str], category: str) -> str:
        nonlocal calls
        calls += 1
        return '{"shouldKeep": true, "category": "苹果产业链", "confidence": 1.0, "reason": "ok"}'

    validation = validate_classification_with_llm(
        news,
        "苹果产业链",
        result,
        verifier=verifier,
        cache=LLMValidationCache(),
    )

    assert validation.should_keep
    assert calls == 0


def test_low_confidence_news_uses_llm() -> None:
    calls = 0
    news = _news("GPU 需求增长")
    result = classify_news_item(news, "AI产业链", EVENT_RULES["AI产业链"])

    def verifier(payload: dict[str, str], category: str) -> str:
        nonlocal calls
        calls += 1
        return '{"shouldKeep": true, "category": "AI产业链", "confidence": 0.8, "reason": "ok"}'

    validation = validate_classification_with_llm(
        news,
        "AI产业链",
        result,
        verifier=verifier,
        cache=LLMValidationCache(),
    )

    assert validation.should_keep
    assert calls == 1


def test_high_impact_news_uses_llm_cache() -> None:
    calls = 0
    news = _news("美元 利率走势分化")
    result = classify_news_item(news, "美联储/利率", EVENT_RULES["美联储/利率"])
    cache = LLMValidationCache()

    def verifier(payload: dict[str, str], category: str) -> str:
        nonlocal calls
        calls += 1
        return '{"shouldKeep": true, "category": "美联储/利率", "confidence": 0.9, "reason": "ok"}'

    first = validate_classification_with_llm(
        news,
        "美联储/利率",
        result,
        verifier=verifier,
        cache=cache,
    )
    second = validate_classification_with_llm(
        news,
        "美联储/利率",
        result,
        verifier=verifier,
        cache=cache,
    )

    assert first.should_keep
    assert second.should_keep
    assert calls == 1


def test_invalid_cache_item_is_ignored() -> None:
    calls = 0
    news = _news("美元 利率走势分化")
    result = classify_news_item(news, "美联储/利率", EVENT_RULES["美联储/利率"])
    cache = LLMValidationCache()
    cache._items[_validation_cache_key(news, "美联储/利率")] = "bad cache item"

    def verifier(payload: dict[str, str], category: str) -> str:
        nonlocal calls
        calls += 1
        return '{"shouldKeep": true, "category": "美联储/利率", "confidence": 0.9, "reason": "ok"}'

    validation = validate_classification_with_llm(
        news,
        "美联储/利率",
        result,
        verifier=verifier,
        cache=cache,
    )

    assert validation.should_keep
    assert calls == 1


def test_invalid_llm_json_downgrades_to_drop() -> None:
    news = _news("美元 利率走势分化")
    result = classify_news_item(news, "美联储/利率", EVENT_RULES["美联储/利率"])

    validation = validate_classification_with_llm(
        news,
        "美联储/利率",
        result,
        verifier=lambda payload, category: "not json",
        cache=LLMValidationCache(),
    )

    assert not validation.should_keep
    assert "JSON" in validation.reason


def test_disabled_provider_reports_notice_and_skips_llm() -> None:
    with temporary_env(
        {
            "LLM_VERIFY_PROVIDER": "disabled",
            "DOUBAO_API_KEY": None,
            "DEEPSEEK_API_KEY": None,
        }
    ):
        verifier, notice = load_llm_verifier_from_env()

    news = _news("美元 利率走势分化")
    result = classify_news_item(news, "美联储/利率", EVENT_RULES["美联储/利率"])
    validation = validate_classification_with_llm(
        news,
        "美联储/利率",
        result,
        verifier=verifier,
        cache=LLMValidationCache(),
    )

    assert verifier is None
    assert notice is not None and "未启用" in notice
    assert validation.should_keep


def test_missing_api_key_disables_provider_without_crash() -> None:
    with temporary_env(
        {
            "LLM_VERIFY_PROVIDER": "doubao",
            "DOUBAO_API_KEY": None,
        }
    ):
        verifier, notice = load_llm_verifier_from_env()

    assert verifier is None
    assert notice is not None and "DOUBAO_API_KEY" in notice


def test_missing_endpoint_or_model_disables_provider_without_crash() -> None:
    with temporary_env(
        {
            "LLM_VERIFY_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_ENDPOINT": None,
            "DEEPSEEK_MODEL": None,
        }
    ):
        verifier, notice = load_llm_verifier_from_env()

    assert verifier is None
    assert notice is not None and "DEEPSEEK_ENDPOINT" in notice

    with temporary_env(
        {
            "LLM_VERIFY_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_ENDPOINT": "https://example.com/deepseek",
            "DEEPSEEK_MODEL": None,
        }
    ):
        verifier, notice = load_llm_verifier_from_env()

    assert verifier is None
    assert notice is not None and "DEEPSEEK_MODEL" in notice


def test_provider_env_config_overrides_model_and_endpoint() -> None:
    with temporary_env(
        {
            "LLM_VERIFY_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_ENDPOINT": "https://example.com/deepseek",
            "DEEPSEEK_MODEL": "deepseek-test",
            "LLM_VERIFY_TIMEOUT": "3",
        }
    ):
        verifier, notice = load_llm_verifier_from_env()

    assert notice is None
    assert verifier is not None
    assert verifier.provider == "deepseek"
    assert verifier.endpoint == "https://example.com/deepseek"
    assert verifier.model == "deepseek-test"
    assert verifier.timeout == 3


def test_invalid_llm_json_fields_downgrade_to_drop() -> None:
    news = _news("美元 利率走势分化")
    result = classify_news_item(news, "美联储/利率", EVENT_RULES["美联储/利率"])
    invalid_payloads = [
        '{"shouldKeep": true, "category": "美联储/利率", "confidence": 0.9}',
        '{"shouldKeep": "true", "category": "美联储/利率", "confidence": 0.9, "reason": "bad"}',
        '{"shouldKeep": true, "category": 123, "confidence": 0.9, "reason": "bad"}',
        '{"shouldKeep": true, "category": "美联储/利率", "confidence": "0.9", "reason": "bad"}',
        '{"shouldKeep": true, "category": "美联储/利率", "confidence": 1.2, "reason": "bad"}',
        '{"shouldKeep": true, "category": "美联储/利率", "confidence": 0.9, "reason": 123}',
    ]

    for payload in invalid_payloads:
        validation = validate_classification_with_llm(
            news,
            "美联储/利率",
            result,
            verifier=lambda item, category, payload=payload: payload,
            cache=LLMValidationCache(),
        )
        assert not validation.should_keep


def _fetch_external_with_mocked_title(
    event_category: str,
    title: str,
    llm_verifier=None,
) -> fetcher.SectorResult:
    original_fetch = fetcher.fetch_keyword_news

    def fake_fetch(keyword: str, timeout: int = fetcher.DEFAULT_TIMEOUT) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "标题": title,
                    "来源媒体": "测试媒体",
                    "发布时间": "2026-05-14 10:00:00",
                    "原文链接": f"https://example.com/{event_category}",
                    "匹配关键词": keyword,
                    "新闻内容": "",
                }
            ]
        )

    fetcher.fetch_keyword_news = fake_fetch
    try:
        return fetcher.fetch_external_event_news(
            event_category,
            EVENT_RULES[event_category],
            llm_verifier=llm_verifier,
            llm_cache=LLMValidationCache(),
        )
    finally:
        fetcher.fetch_keyword_news = original_fetch


def test_doubao_verifier_can_reject_low_confidence_candidate() -> None:
    verifier, calls = _chat_verifier(
        "doubao",
        '{"shouldKeep": false, "category": "苹果产业链", "confidence": 0.9, "reason": "泛词误匹配"}',
    )

    result = _fetch_external_with_mocked_title(
        "苹果产业链",
        "Apple 产业链调整",
        llm_verifier=verifier,
    )

    assert result.data.empty
    assert len(calls) == 1


def test_deepseek_verifier_can_reject_low_confidence_candidate() -> None:
    verifier, calls = _chat_verifier(
        "deepseek",
        '{"shouldKeep": false, "category": "美联储/利率", "confidence": 0.9, "reason": "泛词误匹配"}',
    )

    result = _fetch_external_with_mocked_title(
        "美联储/利率",
        "美元 利率走势分化",
        llm_verifier=verifier,
    )

    assert result.data.empty
    assert len(calls) == 1


def test_doubao_responses_endpoint_uses_input_payload_and_output_text() -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, "payload": kwargs["json"]})
        return FakeResponse(
            {
                "output_text": (
                    '{"shouldKeep": false, "category": "美联储/利率", '
                    '"confidence": 0.9, "reason": "泛词误匹配"}'
                )
            }
        )

    verifier = ChatCompletionsLLMVerifier(
        LLMProviderConfig(
            provider="doubao",
            api_key="test-secret",
            endpoint="https://example.com/api/v3/responses",
            model="ep-test",
        ),
        post=fake_post,
    )
    validation = verifier.verify(
        _news("中美元首会谈结束"),
        "美联储/利率",
        {"matchedKeywords": ["美元"]},
    )

    assert not validation.should_keep
    assert len(calls) == 1
    assert "input" in calls[0]["payload"]
    assert "messages" not in calls[0]["payload"]


def test_doubao_responses_endpoint_reads_output_content_text() -> None:
    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "text": (
                                    '{"shouldKeep": false, "category": "美联储/利率", '
                                    '"confidence": 0.8, "reason": "泛词误匹配"}'
                                )
                            }
                        ]
                    }
                ]
            }
        )

    verifier = ChatCompletionsLLMVerifier(
        LLMProviderConfig(
            provider="doubao",
            api_key="test-secret",
            endpoint="https://example.com/api/v3/responses",
            model="ep-test",
        ),
        post=fake_post,
    )
    validation = verifier.verify(
        _news("中美元首会谈结束"),
        "美联储/利率",
        {"matchedKeywords": ["美元"]},
    )

    assert not validation.should_keep


def test_doubao_chat_endpoint_omits_response_format() -> None:
    verifier, calls = _chat_verifier(
        "doubao",
        '{"shouldKeep": false, "category": "美联储/利率", "confidence": 0.8, "reason": "ok"}',
    )

    validation = verifier.verify(
        _news("中美元首会谈结束"),
        "美联储/利率",
        {"matchedKeywords": ["美元"]},
    )

    assert not validation.should_keep
    assert "messages" in calls[0]["payload"]
    assert "response_format" not in calls[0]["payload"]


def test_deepseek_chat_endpoint_keeps_response_format() -> None:
    verifier, calls = _chat_verifier(
        "deepseek",
        '{"shouldKeep": false, "category": "美联储/利率", "confidence": 0.8, "reason": "ok"}',
    )

    validation = verifier.verify(
        _news("中美元首会谈结束"),
        "美联储/利率",
        {"matchedKeywords": ["美元"]},
    )

    assert not validation.should_keep
    assert "messages" in calls[0]["payload"]
    assert calls[0]["payload"]["response_format"] == {"type": "json_object"}


def test_low_confidence_high_risk_news_uses_llm() -> None:
    verifier = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.8, "reason": "ok"}'
    )
    news = _news("Apple 产业链调整")
    result = classify_news_item(news, "苹果产业链", EVENT_RULES["苹果产业链"])

    validation = validate_classification_with_llm(
        news,
        "苹果产业链",
        result,
        verifier=verifier,
        cache=LLMValidationCache(),
    )

    assert validation.should_keep
    assert verifier.calls == 1
    assert verifier.last_evidence is not None
    assert verifier.last_evidence["shouldDropOnFailure"]


def test_llm_cache_key_separates_provider_metadata() -> None:
    news = _news("Apple 产业链调整")
    result = classify_news_item(news, "苹果产业链", EVENT_RULES["苹果产业链"])
    cache = LLMValidationCache()
    doubao = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.9, "reason": "ok"}',
        provider="doubao",
        model="doubao-lite",
    )
    deepseek = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.9, "reason": "ok"}',
        provider="deepseek",
        model="deepseek-chat",
    )

    validate_classification_with_llm(
        news,
        "苹果产业链",
        result,
        verifier=doubao,
        cache=cache,
    )
    validate_classification_with_llm(
        news,
        "苹果产业链",
        result,
        verifier=doubao,
        cache=cache,
    )
    validate_classification_with_llm(
        news,
        "苹果产业链",
        result,
        verifier=deepseek,
        cache=cache,
    )

    assert doubao.calls == 1
    assert deepseek.calls == 1


def test_llm_cache_key_separates_prompt_and_rule_versions() -> None:
    news = _news("Apple 产业链调整")
    result = classify_news_item(news, "苹果产业链", EVENT_RULES["苹果产业链"])
    cache = LLMValidationCache()
    prompt_v1 = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.9, "reason": "ok"}',
        provider="doubao",
        model="doubao-lite",
        prompt_version="prompt-v1",
        rule_version="rule-v1",
    )
    prompt_v2 = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.9, "reason": "ok"}',
        provider="doubao",
        model="doubao-lite",
        prompt_version="prompt-v2",
        rule_version="rule-v1",
    )
    rule_v2 = CountingVerifier(
        '{"shouldKeep": true, "category": "苹果产业链", "confidence": 0.9, "reason": "ok"}',
        provider="doubao",
        model="doubao-lite",
        prompt_version="prompt-v1",
        rule_version="rule-v2",
    )

    for verifier in (prompt_v1, prompt_v1, prompt_v2, rule_v2):
        validate_classification_with_llm(
            news,
            "苹果产业链",
            result,
            verifier=verifier,
            cache=cache,
        )

    assert prompt_v1.calls == 1
    assert prompt_v2.calls == 1
    assert rule_v2.calls == 1


def test_fetch_external_event_news_filters_mocked_fetch() -> None:
    original_fetch = fetcher.fetch_keyword_news

    def fake_fetch(keyword: str, timeout: int = fetcher.DEFAULT_TIMEOUT) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "标题": "海马汽车连续亏损",
                    "来源媒体": "测试媒体",
                    "发布时间": "2026-05-14 10:00:00",
                    "原文链接": f"https://example.com/{keyword}",
                    "匹配关键词": keyword,
                    "新闻内容": "",
                }
            ]
        )

    fetcher.fetch_keyword_news = fake_fetch
    try:
        result = fetcher.fetch_external_event_news(
            "苹果产业链",
            EVENT_RULES["苹果产业链"],
        )
    finally:
        fetcher.fetch_keyword_news = original_fetch

    assert result.data.empty


def _fetch_sector_with_mocked_title(sector: str, keywords, title: str, llm_verifier=None):
    original_fetch = fetcher.fetch_keyword_news

    def fake_fetch(keyword: str, timeout: int = fetcher.DEFAULT_TIMEOUT) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "标题": title,
                    "来源媒体": "测试媒体",
                    "发布时间": "2026-05-14 10:00:00",
                    "原文链接": f"https://example.com/{sector}/{keyword}",
                    "匹配关键词": keyword,
                    "新闻内容": "",
                }
            ]
        )

    fetcher.fetch_keyword_news = fake_fetch
    try:
        return fetcher.fetch_sector_news(
            sector,
            keywords,
            llm_verifier=llm_verifier,
            llm_cache=LLMValidationCache(),
        )
    finally:
        fetcher.fetch_keyword_news = original_fetch


def test_fetch_sector_news_filters_high_risk_apple_sector() -> None:
    assert _fetch_sector_with_mocked_title(
        "苹果产业链",
        ["Apple"],
        "海马汽车连续亏损",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "苹果产业链",
        ["Apple"],
        "Apple 发布新品",
    ).data.empty
    assert not _fetch_sector_with_mocked_title(
        "苹果产业链",
        ["Apple"],
        "Apple 供应链订单改善",
    ).data.empty


def test_fetch_sector_news_filters_high_risk_fed_sector() -> None:
    assert _fetch_sector_with_mocked_title(
        "美联储/利率",
        ["美元"],
        "中美元首会谈结束",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "美联储/利率",
        ["美元"],
        "印度卢比兑美元跌破",
    ).data.empty
    assert not _fetch_sector_with_mocked_title(
        "美联储/利率",
        ["美联储"],
        "美联储宣布降息",
    ).data.empty


def test_fetch_sector_news_filters_vision_pro_sector() -> None:
    assert not _fetch_sector_with_mocked_title(
        "Vision Pro",
        ["Vision Pro"],
        "Vision Pro 需求回暖",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "Vision Pro",
        ["Pro"],
        "公司发布 Pro 版本",
    ).data.empty


def test_fetch_sector_news_uses_rule_id_for_custom_sector_names() -> None:
    assert _fetch_sector_with_mocked_title(
        "苹果概念",
        {"keywords": ["Apple"], "rule_id": "apple_supply_chain"},
        "海马汽车连续亏损",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "Fed利率",
        {"keywords": ["美元"], "rule_id": "fed_rate"},
        "中美元首会谈结束",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "Fed利率",
        {"keywords": ["美元"], "rule_id": "fed_rate"},
        "印度卢比兑美元跌破",
    ).data.empty
    assert not _fetch_sector_with_mocked_title(
        "头显观察",
        {"keywords": ["Vision Pro"], "rule_id": "vision_pro"},
        "Vision Pro 需求回暖",
    ).data.empty
    assert _fetch_sector_with_mocked_title(
        "头显观察",
        {"keywords": ["Pro"], "rule_id": "vision_pro"},
        "公司发布 Pro 版本",
    ).data.empty


def test_fetch_sector_news_uses_llm_for_low_confidence_high_risk_candidate() -> None:
    verifier = CountingVerifier(
        '{"shouldKeep": false, "category": "苹果产业链", "confidence": 0.9, "reason": "泛词误匹配"}'
    )

    result = _fetch_sector_with_mocked_title(
        "苹果产业链",
        ["Apple"],
        "Apple 产业链调整",
        llm_verifier=verifier,
    )

    assert result.data.empty
    assert verifier.calls == 1


def test_fetch_sector_news_skips_llm_for_high_confidence_candidate() -> None:
    verifier = CountingVerifier(
        '{"shouldKeep": false, "category": "Vision Pro", "confidence": 0.9, "reason": "should not call"}'
    )

    result = _fetch_sector_with_mocked_title(
        "Vision Pro",
        {"keywords": ["Vision Pro"], "rule_id": "vision_pro"},
        "Vision Pro 需求回暖",
        llm_verifier=verifier,
    )

    assert not result.data.empty
    assert verifier.calls == 0


def test_old_external_event_cache_is_refiltered() -> None:
    cache_df = pd.DataFrame(
        [
            {
                "news_type": "external_event",
                "sector": "",
                "title": "印度卢比兑美元跌破",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/usd-inr",
                "keyword": "美元",
                "content": "",
                "event_category": "美联储/利率",
                "related_sectors": "银行、证券",
                "reason": "旧缓存",
                "fetched_at": "2026-05-14 10:00:00",
            },
            {
                "news_type": "sector_news",
                "sector": "半导体芯片",
                "title": "半导体设备订单增长",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/chip",
                "keyword": "半导体",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
        ]
    )

    filtered_df, warnings = refilter_external_event_cache(cache_df, EVENT_RULES)

    assert filtered_df[filtered_df["news_type"].eq("external_event")].empty
    assert len(filtered_df[filtered_df["news_type"].eq("sector_news")]) == 1
    assert warnings


def test_old_high_risk_sector_cache_is_refiltered() -> None:
    cache_df = pd.DataFrame(
        [
            {
                "news_type": "sector_news",
                "sector": "苹果产业链",
                "title": "海马汽车连续亏损",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/haima",
                "keyword": "Apple",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
            {
                "news_type": "sector_news",
                "sector": "Vision Pro",
                "title": "公司发布 Pro 版本",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/pro",
                "keyword": "Pro",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
            {
                "news_type": "sector_news",
                "sector": "半导体芯片",
                "title": "半导体设备订单增长",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/chip",
                "keyword": "半导体",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
        ]
    )

    filtered_df, warnings = refilter_external_event_cache(cache_df, EVENT_RULES)

    assert len(filtered_df[filtered_df["sector"].eq("苹果产业链")]) == 0
    assert len(filtered_df[filtered_df["sector"].eq("Vision Pro")]) == 0
    assert len(filtered_df[filtered_df["sector"].eq("半导体芯片")]) == 1
    assert warnings


def test_old_custom_sector_cache_uses_rule_id() -> None:
    cache_df = pd.DataFrame(
        [
            {
                "news_type": "sector_news",
                "sector": "苹果概念",
                "title": "海马汽车连续亏损",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/haima",
                "keyword": "Apple",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
            {
                "news_type": "sector_news",
                "sector": "半导体芯片",
                "title": "半导体设备订单增长",
                "source": "测试媒体",
                "publish_time": "2026-05-14 10:00:00",
                "link": "https://example.com/chip",
                "keyword": "半导体",
                "content": "",
                "event_category": "",
                "related_sectors": "",
                "reason": "",
                "fetched_at": "2026-05-14 10:00:00",
            },
        ]
    )
    sectors_config = {
        "苹果概念": {"keywords": ["Apple"], "rule_id": "apple_supply_chain"},
        "半导体芯片": ["半导体"],
    }

    filtered_df, warnings = refilter_external_event_cache(
        cache_df,
        EVENT_RULES,
        sectors_config=sectors_config,
    )

    assert len(filtered_df[filtered_df["sector"].eq("苹果概念")]) == 0
    assert len(filtered_df[filtered_df["sector"].eq("半导体芯片")]) == 1
    assert warnings


if __name__ == "__main__":
    test_sector_config_compatibility()
    test_sector_keyword_update_preserves_rule_id()
    test_unknown_rule_id_does_not_fallback_to_sector_name()
    test_regression_false_positives()
    test_vision_pro_requires_complete_phrase()
    test_high_confidence_normal_news_skips_llm()
    test_low_confidence_news_uses_llm()
    test_high_impact_news_uses_llm_cache()
    test_invalid_cache_item_is_ignored()
    test_invalid_llm_json_downgrades_to_drop()
    test_disabled_provider_reports_notice_and_skips_llm()
    test_missing_api_key_disables_provider_without_crash()
    test_missing_endpoint_or_model_disables_provider_without_crash()
    test_provider_env_config_overrides_model_and_endpoint()
    test_invalid_llm_json_fields_downgrade_to_drop()
    test_doubao_verifier_can_reject_low_confidence_candidate()
    test_deepseek_verifier_can_reject_low_confidence_candidate()
    test_doubao_responses_endpoint_uses_input_payload_and_output_text()
    test_doubao_responses_endpoint_reads_output_content_text()
    test_doubao_chat_endpoint_omits_response_format()
    test_deepseek_chat_endpoint_keeps_response_format()
    test_low_confidence_high_risk_news_uses_llm()
    test_llm_cache_key_separates_provider_metadata()
    test_llm_cache_key_separates_prompt_and_rule_versions()
    test_fetch_external_event_news_filters_mocked_fetch()
    test_fetch_sector_news_filters_high_risk_apple_sector()
    test_fetch_sector_news_filters_high_risk_fed_sector()
    test_fetch_sector_news_filters_vision_pro_sector()
    test_fetch_sector_news_uses_rule_id_for_custom_sector_names()
    test_fetch_sector_news_uses_llm_for_low_confidence_high_risk_candidate()
    test_fetch_sector_news_skips_llm_for_high_confidence_candidate()
    test_old_external_event_cache_is_refiltered()
    test_old_high_risk_sector_cache_is_refiltered()
    test_old_custom_sector_cache_uses_rule_id()
    print("test_classifier.py: ok")
