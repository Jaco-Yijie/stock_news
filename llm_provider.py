from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import requests

from classifier import LLMValidationResult, parse_llm_validation


PROMPT_VERSION = "news-classification-reverse-v1"
RULE_VERSION = "structured-rules-v1"
DEFAULT_TIMEOUT = 8

PROVIDER_DEFAULTS = {
    "doubao": {
        "api_key_env": "DOUBAO_API_KEY",
        "endpoint_env": "DOUBAO_ENDPOINT",
        "model_env": "DOUBAO_MODEL",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "endpoint_env": "DEEPSEEK_ENDPOINT",
        "model_env": "DEEPSEEK_MODEL",
    },
}


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    api_key: str
    endpoint: str
    model: str
    timeout: int = DEFAULT_TIMEOUT
    prompt_version: str = PROMPT_VERSION
    rule_version: str = RULE_VERSION


class ChatCompletionsLLMVerifier:
    def __init__(
        self,
        config: LLMProviderConfig,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.provider = config.provider
        self.model = config.model
        self.endpoint = config.endpoint
        self.prompt_version = config.prompt_version
        self.rule_version = config.rule_version
        self.timeout = config.timeout
        self._api_key = config.api_key
        self._post = post or requests.post

    def verify(
        self,
        news_item: dict[str, Any],
        category: str,
        evidence: dict[str, Any],
    ) -> LLMValidationResult:
        system_prompt = _system_prompt()
        user_prompt = _user_prompt(news_item, category, evidence)
        if self.provider == "doubao" and "/responses" in self.endpoint.casefold():
            payload = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            data = self._post_json(payload)
            return parse_llm_validation(_extract_responses_content(data))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0,
        }
        if self.provider != "doubao":
            payload["response_format"] = {"type": "json_object"}
        data = self._post_json(payload)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return parse_llm_validation(content)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """通用文本生成（用于日报归纳等场景，不强制 JSON 输出）。"""
        if self.provider == "doubao" and "/responses" in self.endpoint.casefold():
            payload = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            return _extract_responses_content(self._post_json(payload))

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }
        data = self._post_json(payload)
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("LLM provider 返回内容必须是 JSON 对象")
        return data


def load_llm_verifier_from_env() -> tuple[ChatCompletionsLLMVerifier | None, str | None]:
    provider = os.getenv("LLM_VERIFY_PROVIDER", "disabled").strip().casefold()
    if not provider or provider == "disabled":
        return None, "LLM 反向校验未启用：LLM_VERIFY_PROVIDER=disabled。"

    defaults = PROVIDER_DEFAULTS.get(provider)
    if defaults is None:
        return None, f"LLM 反向校验未启用：未知 provider「{provider}」。"

    api_key_env = defaults["api_key_env"]
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        return None, f"LLM 反向校验未启用：缺少环境变量 {api_key_env}。"

    endpoint_env = defaults["endpoint_env"]
    endpoint = os.getenv(endpoint_env, "").strip()
    if not endpoint:
        return None, f"LLM 反向校验未启用：缺少环境变量 {endpoint_env}。"

    model_env = defaults["model_env"]
    model = os.getenv(model_env, "").strip()
    if not model:
        return None, f"LLM 反向校验未启用：缺少环境变量 {model_env}。"

    timeout = _env_int("LLM_VERIFY_TIMEOUT", DEFAULT_TIMEOUT)
    config = LLMProviderConfig(
        provider=provider,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
    )
    return ChatCompletionsLLMVerifier(config), None


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _clean_text(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _extract_responses_content(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)

    return ""


def _system_prompt() -> str:
    return (
        "你是新闻分类反向校验器。只判断新闻是否真的属于给定分类，"
        "不要创造新分类。你必须只输出 JSON，不要输出解释文本。"
    )


def _user_prompt(
    news_item: dict[str, Any],
    category: str,
    evidence: dict[str, Any],
) -> str:
    prompt_payload = {
        "task": "判断这条新闻是否真的属于给定分类，而不是被泛词误匹配。",
        "outputSchema": {
            "shouldKeep": True,
            "category": "string",
            "confidence": 0.0,
            "reason": "string",
        },
        "rules": [
            "只能确认保留或反向否决。",
            "如果不属于给定分类，shouldKeep=false。",
            "不要输出 JSON 以外的任何文本。",
        ],
        "negativeExamples": [
            {"title": "中美元首会谈结束", "category": "美联储/利率", "shouldKeep": False},
            {"title": "印度卢比兑美元跌破", "category": "美联储/利率", "shouldKeep": False},
            {"title": "公司发布 Pro 版本", "category": "Vision Pro", "shouldKeep": False},
            {"title": "海马汽车连续亏损", "category": "苹果产业链", "shouldKeep": False},
        ],
        "category": category,
        "news": {
            "title": _clean_text(news_item.get("标题") or news_item.get("title"), 200),
            "summary": _clean_text(news_item.get("摘要") or news_item.get("summary"), 300),
            "content": _clean_text(news_item.get("新闻内容") or news_item.get("content"), 500),
            "link": _clean_text(news_item.get("原文链接") or news_item.get("link"), 300),
        },
        "evidence": evidence,
    }
    return json.dumps(prompt_payload, ensure_ascii=False)
