from __future__ import annotations

from stock_search import (
    format_market_cap,
    match_sectors_for_stock,
    normalize_stock_query,
    search_stocks,
)

STOCK_PAIRS = [
    ("600519", "贵州茅台"),
    ("300750", "宁德时代"),
    ("688981", "中芯国际"),
    ("600036", "招商银行"),
    ("601398", "工商银行"),
    ("000001", "平安银行"),
]

SECTORS_CONFIG = {
    "半导体芯片": ["半导体", "芯片", "集成电路", "晶圆"],
    "白酒消费": ["白酒", "消费", "食品饮料", "零售"],
    "新能源汽车": ["新能源汽车", "电动车", "锂电池", "充电桩"],
    "储能": ["储能", "新型储能", "电池储能", "储能系统"],
    "银行": ["银行", "商业银行", "存款", "贷款"],
}


def test_search_by_exact_code() -> None:
    matches = search_stocks("600519", STOCK_PAIRS)
    assert matches and matches[0].name == "贵州茅台"


def test_search_by_name_substring() -> None:
    matches = search_stocks("茅台", STOCK_PAIRS)
    assert matches and matches[0].code == "600519"


def test_search_exact_name_ranks_first() -> None:
    """完整名称命中时应排在模糊匹配之前。"""
    pairs = STOCK_PAIRS + [("999999", "贵州茅台二号")]
    matches = search_stocks("贵州茅台", pairs)
    assert matches[0].code == "600519"


def test_search_code_prefix_matches_multiple() -> None:
    matches = search_stocks("60", STOCK_PAIRS)
    codes = {match.code for match in matches}
    assert {"600519", "600036"} <= codes


def test_search_blank_query_returns_empty() -> None:
    assert search_stocks("   ", STOCK_PAIRS) == []
    assert normalize_stock_query("  60 0519 ") == "600519"


def test_match_sectors_by_stock_hint() -> None:
    """个股直接映射优先：宁德时代 -> 新能源汽车 + 储能。"""
    sectors = match_sectors_for_stock("宁德时代", "电池", SECTORS_CONFIG)
    assert sectors[:2] == ["新能源汽车", "储能"]


def test_match_sectors_by_industry_bridge() -> None:
    """行业名桥接：未知个股按行业“半导体”映射到半导体芯片板块。"""
    sectors = match_sectors_for_stock("某芯片公司", "半导体", SECTORS_CONFIG)
    assert "半导体芯片" in sectors


def test_match_sectors_by_keyword_overlap() -> None:
    """行业名与板块关键词互相包含：行业“白酒”命中白酒消费板块。"""
    sectors = match_sectors_for_stock("某酒企", "白酒", SECTORS_CONFIG)
    assert "白酒消费" in sectors


def test_match_sectors_only_returns_configured() -> None:
    """映射结果里不在当前板块配置中的板块要被过滤掉。"""
    config = {"银行": ["银行", "商业银行"]}
    sectors = match_sectors_for_stock("宁德时代", "电池", config)
    assert sectors == []


def test_format_market_cap() -> None:
    assert format_market_cap("2100000000000") == "2.10 万亿"
    assert format_market_cap("35000000000") == "350.00 亿"
    assert format_market_cap("") == "-"
    assert format_market_cap("未知") == "未知"


if __name__ == "__main__":
    test_search_by_exact_code()
    test_search_by_name_substring()
    test_search_exact_name_ranks_first()
    test_search_code_prefix_matches_multiple()
    test_search_blank_query_returns_empty()
    test_match_sectors_by_stock_hint()
    test_match_sectors_by_industry_bridge()
    test_match_sectors_by_keyword_overlap()
    test_match_sectors_only_returns_configured()
    test_format_market_cap()
    print("test_stock_search.py: ok")
