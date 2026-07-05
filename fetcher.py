from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import unescape
from threading import Lock
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests

try:
    from pandas.core.strings.accessor import StringMethods
except ImportError:
    StringMethods = None

try:
    import akshare as ak
except ImportError as exc:
    ak = None
    AKSHARE_IMPORT_ERROR: ImportError | None = exc
else:
    AKSHARE_IMPORT_ERROR = None

from classifier import (
    LLMValidationCache,
    LLMVerifier,
    classify_news_item,
    extract_rule_keywords,
    normalize_event_rules,
    validate_classification_with_llm,
)
from config_store import extract_sector_keywords, extract_sector_rule_id
from sectors import EVENT_TO_SECTORS, EXTERNAL_EVENTS, SECTORS


DEFAULT_TIMEOUT = 10
MAX_FETCH_WORKERS = 8
AKSHARE_FAILURE_THRESHOLD = 3
AKSHARE_SKIP_SECONDS = 600
EASTMONEY_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
DISPLAY_COLUMNS = ["标题", "来源媒体", "发布时间", "原文链接"]
EXTERNAL_EVENT_COLUMNS = ["event_category", "related_sectors", "reason"]
TITLE_COLUMNS = ["新闻标题", "标题", "title"]
SOURCE_COLUMNS = ["文章来源", "来源", "媒体", "source"]
TIME_COLUMNS = ["发布时间", "时间", "日期", "publish_time"]
LINK_COLUMNS = ["新闻链接", "链接", "url", "link"]
CONTENT_COLUMNS = ["新闻内容", "内容", "摘要", "content"]
REQUEST_HEADERS = {
    "Accept": "*/*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
}
PANDAS_REPLACE_PATCH_LOCK = Lock()
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
TITLE_MARKET_MOVE_PATTERN = re.compile(
    r"(?:上涨|下跌|涨近|涨超|涨逾|跌近|跌超|跌逾|大涨|大跌|拉升|跳水|走强|走弱|涨|跌)"
    r"\s*[+-]?\d+(?:\.\d+)?\s*(?:%|％|个百分点|点)?"
)
TITLE_PERCENT_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?\s*(?:%|％|个百分点)")
TITLE_NOISE_WORDS = (
    "最新",
    "突发",
    "快讯",
    "午评",
    "盘中",
    "异动",
    "拉升",
    "走强",
    "跳水",
)
TITLE_SIMILARITY_THRESHOLD = 0.88
DEFAULT_LLM_VALIDATION_CACHE = LLMValidationCache()
VISION_PRO_SECTOR_RULES = [
    {
        "positiveKeywords": ["Vision Pro"],
        "requiredCoKeywords": [],
        "negativeKeywords": [],
        "weight": 2,
        "minScore": 2,
        "fields": ["title", "summary", "content"],
    }
]


@dataclass(frozen=True)
class SectorResult:
    data: pd.DataFrame
    error: str | None = None
    warnings: tuple[str, ...] = ()


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*DISPLAY_COLUMNS, "匹配关键词", "新闻内容"])


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = unescape(str(value))
    for old, new in (
        ("\u3000", " "),
        ("\xa0", " "),
        ("\r", " "),
        ("\n", " "),
        ("\t", " "),
        ("<em>", ""),
        ("</em>", ""),
        ("<span>", ""),
        ("</span>", ""),
    ):
        text = text.replace(old, new)
    return " ".join(text.split()).strip()


def _normalize_link_for_dedupe(value: Any) -> str:
    link = _clean_text(value)
    if not link:
        return ""

    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc:
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))

    return link.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def _normalize_text_for_dedupe(value: Any) -> str:
    text = _clean_text(value).casefold()
    text = HTML_TAG_PATTERN.sub("", text)
    text = text.replace("\u3000", " ")
    text = NON_WORD_PATTERN.sub("", text)
    return text


def _normalize_title_for_dedupe(value: Any) -> str:
    text = _clean_text(value).casefold()
    text = HTML_TAG_PATTERN.sub("", text)
    text = TITLE_MARKET_MOVE_PATTERN.sub("", text)
    text = TITLE_PERCENT_PATTERN.sub("", text)
    for noise_word in TITLE_NOISE_WORDS:
        text = text.replace(noise_word, "")
    text = text.replace("\u3000", " ")
    return NON_WORD_PATTERN.sub("", text)


def _series_or_default(df: pd.DataFrame, column: str | None, default: str) -> pd.Series:
    if column is None:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].map(_clean_text)


def _normalize_news_frame(raw_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return _empty_news_frame()

    title_col = _pick_column(raw_df, TITLE_COLUMNS)
    link_col = _pick_column(raw_df, LINK_COLUMNS)
    source_col = _pick_column(raw_df, SOURCE_COLUMNS)
    time_col = _pick_column(raw_df, TIME_COLUMNS)
    content_col = _pick_column(raw_df, CONTENT_COLUMNS)

    if title_col is None:
        raise ValueError(f"AKShare 返回数据缺少标题字段，实际字段：{list(raw_df.columns)}")
    if link_col is None:
        raise ValueError(f"AKShare 返回数据缺少链接字段，实际字段：{list(raw_df.columns)}")

    news_df = pd.DataFrame(
        {
            "标题": _series_or_default(raw_df, title_col, ""),
            "来源媒体": _series_or_default(raw_df, source_col, "未知来源"),
            "发布时间": _series_or_default(raw_df, time_col, ""),
            "原文链接": _series_or_default(raw_df, link_col, ""),
            "匹配关键词": keyword,
            "新闻内容": _series_or_default(raw_df, content_col, ""),
        }
    )

    news_df = news_df[
        news_df["标题"].str.strip().ne("") & news_df["原文链接"].str.strip().ne("")
    ]
    return news_df


def _sort_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sorted_df = df.copy()
    parsed_time = pd.to_datetime(sorted_df["发布时间"], errors="coerce")
    sorted_df["_发布时间排序"] = parsed_time
    sorted_df = sorted_df.sort_values(
        by=["_发布时间排序", "发布时间"],
        ascending=[False, False],
        na_position="last",
    )
    return sorted_df.drop(columns=["_发布时间排序"]).reset_index(drop=True)


def _is_similar_title(title: str, kept_titles: list[str]) -> bool:
    if not title:
        return False

    matcher = SequenceMatcher()
    matcher.set_seq2(title)
    title_len = len(title)
    for kept_title in kept_titles:
        if not kept_title:
            continue
        # 长度差过大时 ratio 不可能达到阈值，直接跳过昂贵的比较
        if 2 * min(title_len, len(kept_title)) < TITLE_SIMILARITY_THRESHOLD * (
            title_len + len(kept_title)
        ):
            continue
        matcher.set_seq1(kept_title)
        if matcher.real_quick_ratio() < TITLE_SIMILARITY_THRESHOLD:
            continue
        if matcher.quick_ratio() < TITLE_SIMILARITY_THRESHOLD:
            continue
        if matcher.ratio() >= TITLE_SIMILARITY_THRESHOLD:
            return True
    return False


def _column_values(df: pd.DataFrame, column: str) -> list[Any]:
    if column in df.columns:
        return df[column].tolist()
    return [""] * len(df)


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sorted_df = _sort_news(df)
    link_keys = [_normalize_link_for_dedupe(v) for v in _column_values(sorted_df, "原文链接")]
    title_keys = [_normalize_title_for_dedupe(v) for v in _column_values(sorted_df, "标题")]
    content_keys = [_normalize_text_for_dedupe(v) for v in _column_values(sorted_df, "新闻内容")]

    kept_positions: list[int] = []
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    seen_contents: set[str] = set()
    kept_titles: list[str] = []

    for position in range(len(sorted_df)):
        link_key = link_keys[position]
        title_key = title_keys[position]
        content_key = content_keys[position]

        is_duplicate = False
        if link_key and link_key in seen_links:
            is_duplicate = True
        elif title_key and title_key in seen_titles:
            is_duplicate = True
        elif content_key and content_key in seen_contents:
            is_duplicate = True
        elif _is_similar_title(title_key, kept_titles):
            is_duplicate = True

        if is_duplicate:
            continue

        kept_positions.append(position)
        if link_key:
            seen_links.add(link_key)
        if title_key:
            seen_titles.add(title_key)
            kept_titles.append(title_key)
        if content_key:
            seen_contents.add(content_key)

    return sorted_df.iloc[kept_positions].reset_index(drop=True)


_ORIGINAL_STRING_REPLACE = StringMethods.replace if StringMethods is not None else None
_PANDAS_REPLACE_PATCH_DEPTH = 0


def _patched_string_replace(
    self,
    pat,
    repl,
    n=-1,
    case=None,
    flags=0,
    regex=False,
):
    if pat == r"\u3000" and regex is True:
        return _ORIGINAL_STRING_REPLACE(
            self,
            "\u3000",
            repl,
            n=n,
            case=case,
            flags=flags,
            regex=False,
        )
    return _ORIGINAL_STRING_REPLACE(
        self,
        pat,
        repl,
        n=n,
        case=case,
        flags=flags,
        regex=regex,
    )


@contextmanager
def _akshare_regex_compat():
    # \u5f15\u7528\u8ba1\u6570\u5f0f\u8865\u4e01\uff1a\u9501\u53ea\u4fdd\u62a4\u8ba1\u6570\u5668\uff0c\u6293\u53d6\u672c\u8eab\u53ef\u4ee5\u5e76\u53d1\u8fdb\u884c
    global _PANDAS_REPLACE_PATCH_DEPTH
    if StringMethods is None:
        yield
        return

    with PANDAS_REPLACE_PATCH_LOCK:
        if _PANDAS_REPLACE_PATCH_DEPTH == 0:
            StringMethods.replace = _patched_string_replace
        _PANDAS_REPLACE_PATCH_DEPTH += 1
    try:
        yield
    finally:
        with PANDAS_REPLACE_PATCH_LOCK:
            _PANDAS_REPLACE_PATCH_DEPTH -= 1
            if _PANDAS_REPLACE_PATCH_DEPTH == 0:
                StringMethods.replace = _ORIGINAL_STRING_REPLACE


def _fetch_from_akshare(keyword: str, timeout: int = DEFAULT_TIMEOUT) -> pd.DataFrame:
    if ak is None:
        raise ImportError(f"akshare 未安装：{AKSHARE_IMPORT_ERROR}")

    with _akshare_regex_compat():
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(ak.stock_news_em, symbol=keyword)
        try:
            raw_df = future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"AKShare 超过 {timeout} 秒未返回") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return _normalize_news_frame(raw_df, keyword)


def _parse_jsonp(text: str) -> dict[str, Any]:
    stripped = text.strip()
    left = stripped.find("(")
    right = stripped.rfind(")")

    if left != -1 and right != -1 and left < right:
        payload = stripped[left + 1 : right]
    else:
        json_start = stripped.find("{")
        json_end = stripped.rfind("}")
        if json_start == -1 or json_end == -1 or json_start > json_end:
            raise ValueError("东方财富 fallback 返回内容不是 JSONP/JSON")
        payload = stripped[json_start : json_end + 1]

    return json.loads(payload)


def _extract_eastmoney_items(payload: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "title" in value:
                items.append(value)
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return items


def _eastmoney_param_variants(keyword: str) -> list[dict[str, Any]]:
    timestamp = int(time.time() * 1000)
    callback = f"jQuery35101792940631092459_{timestamp}"
    search_payload = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 10,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }

    return [
        {
            "cb": callback,
            "param": json.dumps(search_payload, ensure_ascii=False),
            "_": timestamp,
        },
    ]


def _normalize_eastmoney_items(
    items: list[dict[str, Any]], keyword: str
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for item in items:
        title = _clean_text(item.get("title") or item.get("新闻标题"))
        code = _clean_text(item.get("code"))
        link = _clean_text(
            item.get("url")
            or item.get("link")
            or item.get("articleUrl")
            or item.get("newsUrl")
        )
        if not link and code:
            link = f"http://finance.eastmoney.com/a/{code}.html"

        if not title or not link:
            continue

        rows.append(
            {
                "标题": title,
                "来源媒体": _clean_text(
                    item.get("mediaName") or item.get("source") or "东方财富"
                ),
                "发布时间": _clean_text(item.get("date") or item.get("time")),
                "原文链接": link,
                "匹配关键词": keyword,
                "新闻内容": _clean_text(item.get("content") or item.get("summary")),
            }
        )

    if not rows:
        return _empty_news_frame()
    return pd.DataFrame(rows, columns=[*DISPLAY_COLUMNS, "匹配关键词", "新闻内容"])


def _fetch_from_eastmoney(keyword: str, timeout: int = DEFAULT_TIMEOUT) -> pd.DataFrame:
    last_error: Exception | None = None
    headers = {
        **REQUEST_HEADERS,
        "Referer": f"https://so.eastmoney.com/news/s?keyword={quote(keyword)}",
    }

    for params in _eastmoney_param_variants(keyword):
        try:
            response = requests.get(
                EASTMONEY_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = _parse_jsonp(response.text)
            items = _extract_eastmoney_items(payload)
            return _normalize_eastmoney_items(items, keyword)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"东方财富 fallback 失败：{last_error}") from last_error
    return _empty_news_frame()


# AKShare 熔断器：连续失败若干次后，在一段时间内直接使用东方财富数据源，
# 避免在 AKShare 不可用的环境（如云端海外 IP）中每个关键词都等满超时时间。
_AKSHARE_BREAKER_LOCK = Lock()
_AKSHARE_CONSECUTIVE_FAILURES = 0
_AKSHARE_SKIP_UNTIL = 0.0


def _akshare_available() -> bool:
    with _AKSHARE_BREAKER_LOCK:
        return time.time() >= _AKSHARE_SKIP_UNTIL


def _record_akshare_result(success: bool) -> None:
    global _AKSHARE_CONSECUTIVE_FAILURES, _AKSHARE_SKIP_UNTIL
    with _AKSHARE_BREAKER_LOCK:
        if success:
            _AKSHARE_CONSECUTIVE_FAILURES = 0
            _AKSHARE_SKIP_UNTIL = 0.0
            return
        _AKSHARE_CONSECUTIVE_FAILURES += 1
        if _AKSHARE_CONSECUTIVE_FAILURES >= AKSHARE_FAILURE_THRESHOLD:
            _AKSHARE_SKIP_UNTIL = time.time() + AKSHARE_SKIP_SECONDS


# 补充数据源：财联社电报、新浪全球快讯。整流抓取一次后按关键词本地匹配，
# 进程内缓存 5 分钟（含失败结果），避免每个关键词重复请求或反复等待超时。
STREAM_TTL_SECONDS = 300
_STREAM_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_STREAM_CACHE_LOCK = Lock()


def _stream_link(base_link: str, title: str, publish_time: str) -> str:
    # 快讯类接口没有单条 URL，用真实列表页 + 内容指纹片段保证链接唯一可去重
    digest = hashlib.sha1(f"{title}|{publish_time}".encode("utf-8")).hexdigest()[:12]
    return f"{base_link}#{digest}"


def _normalize_stream_frame(
    raw_df: pd.DataFrame,
    source_name: str,
    base_link: str,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return _empty_news_frame()

    title_col = _pick_column(raw_df, TITLE_COLUMNS)
    content_col = _pick_column(raw_df, CONTENT_COLUMNS)
    time_col = _pick_column(raw_df, TIME_COLUMNS)
    date_col = "发布日期" if "发布日期" in raw_df.columns else None

    rows: list[dict[str, str]] = []
    for record in raw_df.to_dict("records"):
        content = _clean_text(record.get(content_col)) if content_col else ""
        title = _clean_text(record.get(title_col)) if title_col else ""
        if not title:
            title = content[:40]
        if not title:
            continue
        publish_time = _clean_text(record.get(time_col)) if time_col else ""
        if date_col:
            publish_date = _clean_text(record.get(date_col))
            if publish_date and publish_date not in publish_time:
                publish_time = f"{publish_date} {publish_time}".strip()
        rows.append(
            {
                "标题": title,
                "来源媒体": source_name,
                "发布时间": publish_time,
                "原文链接": _stream_link(base_link, title, publish_time),
                "匹配关键词": "",
                "新闻内容": content,
            }
        )

    if not rows:
        return _empty_news_frame()
    return pd.DataFrame(rows, columns=[*DISPLAY_COLUMNS, "匹配关键词", "新闻内容"])


STREAM_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("cls_telegraph", "stock_telegraph_cls", "财联社电报", "https://www.cls.cn/telegraph"),
    ("sina_global", "stock_info_global_sina", "新浪财经快讯", "https://finance.sina.com.cn/7x24/"),
)


def _fetch_stream_frame(
    cache_key: str,
    akshare_fn_name: str,
    source_name: str,
    base_link: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    now = time.time()
    with _STREAM_CACHE_LOCK:
        cached = _STREAM_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < STREAM_TTL_SECONDS:
            return cached[1]

    frame = _empty_news_frame()
    try:
        if ak is None:
            raise ImportError(f"akshare 未安装：{AKSHARE_IMPORT_ERROR}")
        fetch_fn = getattr(ak, akshare_fn_name, None)
        if fetch_fn is None:
            raise RuntimeError(f"akshare 缺少 {akshare_fn_name} 接口")
        with _akshare_regex_compat():
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(fetch_fn)
            try:
                raw_df = future.result(timeout=timeout)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        frame = _normalize_stream_frame(raw_df, source_name, base_link)
    except Exception:
        # 失败也写入缓存（空表），TTL 内不再重试，避免逐关键词等待超时
        frame = _empty_news_frame()

    with _STREAM_CACHE_LOCK:
        _STREAM_CACHE[cache_key] = (time.time(), frame)
    return frame


def fetch_supplemental_keyword_news(
    keyword: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cache_key, fn_name, source_name, base_link in STREAM_SOURCES:
        stream_df = _fetch_stream_frame(cache_key, fn_name, source_name, base_link, timeout)
        if stream_df.empty:
            continue
        mask = stream_df["标题"].str.contains(
            keyword, case=False, regex=False
        ) | stream_df["新闻内容"].str.contains(keyword, case=False, regex=False)
        matched = stream_df[mask]
        if not matched.empty:
            matched = matched.copy()
            matched["匹配关键词"] = keyword
            frames.append(matched)
    if not frames:
        return _empty_news_frame()
    return pd.concat(frames, ignore_index=True)


def fetch_keyword_news(keyword: str, timeout: int = DEFAULT_TIMEOUT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    akshare_error: Exception | None = None
    primary_error: Exception | None = None

    if _akshare_available():
        try:
            frames.append(_fetch_from_akshare(keyword, timeout=timeout))
            _record_akshare_result(True)
        except Exception as exc:
            _record_akshare_result(False)
            akshare_error = exc
    else:
        akshare_error = RuntimeError("AKShare 近期连续失败，已暂时切换到东方财富数据源")

    if not frames:
        try:
            frames.append(_fetch_from_eastmoney(keyword, timeout=timeout))
        except Exception as exc:
            primary_error = RuntimeError(
                f"关键词「{keyword}」抓取失败；AKShare 错误：{akshare_error}；"
                f"东方财富 fallback 错误：{exc}"
            )

    supplemental_df = fetch_supplemental_keyword_news(keyword, timeout=timeout)
    if not supplemental_df.empty:
        frames.append(supplemental_df)

    if frames:
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)
    if primary_error is not None:
        raise primary_error
    return _empty_news_frame()


def _fetch_keywords_news(
    query_keywords: list[str],
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    if not query_keywords:
        return frames, errors

    if len(query_keywords) == 1:
        keyword = query_keywords[0]
        try:
            frames.append(fetch_keyword_news(keyword, timeout=timeout))
        except Exception as exc:
            errors.append(f"{keyword}: {exc}")
        return frames, errors

    max_workers = min(MAX_FETCH_WORKERS, len(query_keywords))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            keyword: executor.submit(fetch_keyword_news, keyword, timeout=timeout)
            for keyword in query_keywords
        }
        for keyword, future in futures.items():
            try:
                frames.append(future.result())
            except Exception as exc:
                errors.append(f"{keyword}: {exc}")
    return frames, errors


def _rules_for_sector_rule_id(rule_id: str) -> list[Any]:
    normalized = str(rule_id or "").strip().casefold().replace(" ", "").replace("／", "/")
    if normalized in {"visionpro", "vision_pro", "vision-pro"}:
        return VISION_PRO_SECTOR_RULES
    if normalized in {"苹果产业链", "apple_supply_chain", "apple-supply-chain"}:
        return EXTERNAL_EVENTS.get("苹果产业链", [])
    if normalized in {"美联储/利率", "美联储利率", "fed_rate", "fed-rate", "fed/rate"}:
        return EXTERNAL_EVENTS.get("美联储/利率", [])
    return []


def high_risk_sector_rules(sector: str, sector_config: Any = None) -> list[Any]:
    rule_id = extract_sector_rule_id(sector_config)
    if rule_id:
        return _rules_for_sector_rule_id(rule_id)

    normalized = str(sector or "").strip().casefold().replace(" ", "").replace("／", "/")
    if normalized == "visionpro":
        return VISION_PRO_SECTOR_RULES
    if normalized == "苹果产业链":
        return EXTERNAL_EVENTS.get("苹果产业链", [])
    if normalized in {"美联储/利率", "美联储利率"}:
        return EXTERNAL_EVENTS.get("美联储/利率", [])
    return []


def fetch_sector_news(
    sector: str,
    keywords: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
    llm_verifier: LLMVerifier | None = None,
    llm_cache: LLMValidationCache | None = None,
) -> SectorResult:
    if keywords is None:
        keywords = SECTORS[sector]
    if llm_verifier is not None and llm_cache is None:
        llm_cache = DEFAULT_LLM_VALIDATION_CACHE
    sector_config = keywords
    query_keywords = extract_sector_keywords(sector_config)

    frames, errors = _fetch_keywords_news(query_keywords, timeout=timeout)

    if not frames:
        return SectorResult(
            data=_empty_news_frame(),
            error=f"{sector} 抓取失败；" + "；".join(errors),
        )

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = deduplicate_news(combined_df)
    sector_rules = high_risk_sector_rules(sector, sector_config)
    if sector_rules:
        combined_df = filter_external_event_news(
            combined_df,
            sector,
            sector_rules,
            llm_verifier=llm_verifier,
            llm_cache=llm_cache,
        )

    if errors:
        return SectorResult(data=combined_df, warnings=tuple(errors))
    return SectorResult(data=combined_df)


def build_event_metadata(
    event_category: str,
    event_to_sectors: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    if event_to_sectors is None:
        event_to_sectors = EVENT_TO_SECTORS

    related_sectors = event_to_sectors.get(event_category, [])
    if not related_sectors:
        return {
            "event_category": event_category,
            "related_sectors": "未映射",
            "reason": "影响关系不确定",
        }

    related_text = "、".join(related_sectors)
    return {
        "event_category": event_category,
        "related_sectors": related_text,
        "reason": f"根据「{event_category}」事件类别映射，可能影响：{related_text}",
    }


def annotate_external_event_news(
    news_df: pd.DataFrame,
    event_category: str,
    event_to_sectors: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        frame = _empty_news_frame()
    else:
        frame = news_df.copy()

    metadata = build_event_metadata(event_category, event_to_sectors)
    for column, value in metadata.items():
        frame[column] = value
    return frame


def filter_external_event_news(
    news_df: pd.DataFrame,
    event_category: str,
    event_rules: list[Any],
    llm_verifier: LLMVerifier | None = None,
    llm_cache: LLMValidationCache | None = None,
) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return _empty_news_frame()
    if llm_verifier is not None and llm_cache is None:
        llm_cache = DEFAULT_LLM_VALIDATION_CACHE

    kept_positions: list[int] = []
    for position, news in enumerate(news_df.to_dict("records")):
        classification = classify_news_item(news, event_category, event_rules)
        if not classification.matched:
            continue

        validation = validate_classification_with_llm(
            news,
            event_category,
            classification,
            verifier=llm_verifier,
            cache=llm_cache,
        )
        if not validation.should_keep:
            continue
        if validation.category and validation.category != event_category:
            continue

        kept_positions.append(position)

    if not kept_positions:
        return _empty_news_frame()
    return news_df.iloc[kept_positions].reset_index(drop=True)


def fetch_external_event_news(
    event_category: str,
    keywords: list[Any] | None = None,
    event_to_sectors: dict[str, list[str]] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    llm_verifier: LLMVerifier | None = None,
    llm_cache: LLMValidationCache | None = None,
) -> SectorResult:
    if keywords is None:
        keywords = EXTERNAL_EVENTS[event_category]
    if llm_verifier is not None and llm_cache is None:
        llm_cache = DEFAULT_LLM_VALIDATION_CACHE
    event_rules = normalize_event_rules({event_category: keywords}).get(event_category, [])
    query_keywords = extract_rule_keywords(event_rules)

    frames, errors = _fetch_keywords_news(query_keywords, timeout=timeout)

    if not frames:
        return SectorResult(
            data=annotate_external_event_news(
                _empty_news_frame(),
                event_category,
                event_to_sectors,
            ),
            error=f"{event_category} 抓取失败；" + "；".join(errors),
        )

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = deduplicate_news(combined_df)
    combined_df = filter_external_event_news(
        combined_df,
        event_category,
        event_rules,
        llm_verifier=llm_verifier,
        llm_cache=llm_cache,
    )
    combined_df = annotate_external_event_news(
        combined_df,
        event_category,
        event_to_sectors,
    )

    if errors:
        return SectorResult(data=combined_df, warnings=tuple(errors))
    return SectorResult(data=combined_df)


def fetch_all_sectors() -> dict[str, SectorResult]:
    results: dict[str, SectorResult] = {}
    for sector, keywords in SECTORS.items():
        results[sector] = fetch_sector_news(sector, keywords)
    return results


def fetch_all_external_events(
    external_events: dict[str, list[Any]] | None = None,
    event_to_sectors: dict[str, list[str]] | None = None,
) -> dict[str, SectorResult]:
    if external_events is None:
        external_events = EXTERNAL_EVENTS

    results: dict[str, SectorResult] = {}
    for event_category, keywords in external_events.items():
        results[event_category] = fetch_external_event_news(
            event_category,
            keywords,
            event_to_sectors=event_to_sectors,
        )
    return results


def result_to_dict(result: SectorResult) -> dict[str, Any]:
    return {
        "data": result.data,
        "error": result.error,
        "warnings": list(result.warnings),
    }
