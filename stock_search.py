"""股票搜索：股票代码/名称查找、个股信息获取、股票与板块的关联映射。

模块本身不依赖 Streamlit；网络请求（akshare）由调用方负责缓存。
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from config_store import extract_sector_keywords

# akshare 部分接口不支持超时参数，统一用后台线程兜底，避免页面被网络请求卡死
FETCH_TIMEOUT_SECONDS = 15


def _call_with_timeout(func: Callable[[], Any], timeout: float = FETCH_TIMEOUT_SECONDS) -> Any:
    """在 daemon 线程中执行网络请求，超时抛 TimeoutError。

    必须用 daemon 线程：卡住的请求线程不能阻塞页面，也不能阻塞解释器退出。
    """
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = func()
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError("请求超时")
    if "error" in result:
        raise result["error"]
    return result["value"]

# 常见个股 -> 项目板块 的直接映射（优先级最高）
STOCK_SECTOR_HINTS: dict[str, list[str]] = {
    "贵州茅台": ["白酒消费"],
    "五粮液": ["白酒消费"],
    "泸州老窖": ["白酒消费"],
    "宁德时代": ["新能源汽车", "储能"],
    "比亚迪": ["新能源汽车"],
    "中芯国际": ["半导体芯片"],
    "北方华创": ["半导体芯片"],
    "韦尔股份": ["半导体芯片", "消费电子"],
    "寒武纪": ["人工智能", "半导体芯片"],
    "海光信息": ["人工智能", "半导体芯片", "信创软件"],
    "科大讯飞": ["人工智能"],
    "工业富联": ["算力数据中心", "消费电子"],
    "中际旭创": ["算力数据中心"],
    "浪潮信息": ["算力数据中心", "信创软件"],
    "立讯精密": ["消费电子"],
    "歌尔股份": ["消费电子"],
    "隆基绿能": ["光伏"],
    "通威股份": ["光伏"],
    "阳光电源": ["光伏", "储能"],
    "金风科技": ["风电"],
    "明阳智能": ["风电"],
    "国电南瑞": ["电力设备"],
    "特变电工": ["电力设备"],
    "恒瑞医药": ["创新药", "医药医疗"],
    "百济神州": ["创新药"],
    "药明康德": ["CXO", "医药医疗"],
    "凯莱英": ["CXO"],
    "迈瑞医疗": ["医药医疗"],
    "埃斯顿": ["机器人"],
    "绿的谐波": ["机器人"],
    "汇川技术": ["机器人", "电力设备"],
    "中航沈飞": ["军工"],
    "航发动力": ["军工"],
    "中国船舶": ["军工"],
    "中国卫星": ["商业航天", "军工"],
    "航天电子": ["商业航天", "军工"],
    "亿航智能": ["低空经济"],
    "万丰奥威": ["低空经济"],
    "招商银行": ["银行"],
    "工商银行": ["银行"],
    "中信证券": ["证券"],
    "东方财富": ["证券", "数字经济"],
    "中国平安": ["保险"],
    "中国人寿": ["保险"],
    "紫金矿业": ["有色金属", "黄金"],
    "山东黄金": ["黄金"],
    "赣锋锂业": ["有色金属", "新能源汽车"],
    "中国神华": ["煤炭"],
    "陕西煤业": ["煤炭"],
    "宝钢股份": ["钢铁"],
    "万华化学": ["化工"],
    "牧原股份": ["农业"],
    "温氏股份": ["农业"],
    "中远海控": ["航运港口"],
    "顺丰控股": ["物流快递"],
    "中国中免": ["旅游酒店", "白酒消费"],
    "格力电器": ["家电"],
    "美的集团": ["家电"],
    "海尔智家": ["家电"],
    "万科A": ["房地产"],
    "保利发展": ["房地产"],
    "三六零": ["网络安全", "信创软件"],
    "深信服": ["网络安全"],
    "奇安信": ["网络安全"],
    "金山办公": ["信创软件"],
    "恺英网络": ["传媒游戏"],
    "三七互娱": ["传媒游戏"],
    "分众传媒": ["传媒游戏"],
    "国盾量子": ["量子计算"],
    "光迅科技": ["量子计算", "算力数据中心"],
}

# 东方财富行业名称 -> 项目板块（行业名与板块名不一致时的桥接）
INDUSTRY_TO_SECTORS: dict[str, list[str]] = {
    "半导体": ["半导体芯片"],
    "电子元件": ["消费电子", "半导体芯片"],
    "消费电子": ["消费电子"],
    "光学光电子": ["消费电子"],
    "软件开发": ["信创软件"],
    "计算机设备": ["算力数据中心", "信创软件"],
    "互联网服务": ["数字经济"],
    "通信设备": ["算力数据中心"],
    "通信服务": ["数字经济"],
    "游戏": ["传媒游戏"],
    "文化传媒": ["传媒游戏"],
    "电池": ["新能源汽车", "储能"],
    "汽车整车": ["新能源汽车"],
    "汽车零部件": ["新能源汽车"],
    "能源金属": ["有色金属", "新能源汽车"],
    "光伏设备": ["光伏"],
    "风电设备": ["风电"],
    "电网设备": ["电力设备"],
    "电力行业": ["电力设备"],
    "航天航空": ["军工", "商业航天"],
    "船舶制造": ["军工"],
    "地面兵装": ["军工"],
    "通用设备": ["机器人"],
    "专用设备": ["机器人"],
    "仪器仪表": ["机器人"],
    "化学制药": ["创新药", "医药医疗"],
    "生物制品": ["创新药", "医药医疗"],
    "中药": ["医药医疗"],
    "医疗器械": ["医药医疗"],
    "医疗服务": ["医药医疗", "CXO"],
    "医药商业": ["医药医疗"],
    "白酒": ["白酒消费"],
    "酿酒行业": ["白酒消费"],
    "食品饮料": ["白酒消费"],
    "商业百货": ["白酒消费"],
    "家电行业": ["家电"],
    "房地产开发": ["房地产"],
    "房地产服务": ["房地产"],
    "银行": ["银行"],
    "证券": ["证券"],
    "保险": ["保险"],
    "多元金融": ["证券"],
    "有色金属": ["有色金属"],
    "小金属": ["有色金属"],
    "贵金属": ["黄金"],
    "煤炭行业": ["煤炭"],
    "钢铁行业": ["钢铁"],
    "化学制品": ["化工"],
    "化学原料": ["化工"],
    "化肥行业": ["化工"],
    "农牧饲渔": ["农业"],
    "种植业与林业": ["农业"],
    "航运港口": ["航运港口"],
    "物流行业": ["物流快递"],
    "旅游酒店": ["旅游酒店"],
    "环保行业": ["环保水务"],
    "电源设备": ["储能", "光伏"],
}

# 网络不可用时的兜底股票列表（代码, 名称）
FALLBACK_STOCKS: list[tuple[str, str]] = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("300750", "宁德时代"),
    ("002594", "比亚迪"),
    ("688981", "中芯国际"),
    ("002371", "北方华创"),
    ("688256", "寒武纪"),
    ("002230", "科大讯飞"),
    ("601138", "工业富联"),
    ("300308", "中际旭创"),
    ("000977", "浪潮信息"),
    ("002475", "立讯精密"),
    ("601012", "隆基绿能"),
    ("300274", "阳光电源"),
    ("600276", "恒瑞医药"),
    ("603259", "药明康德"),
    ("300760", "迈瑞医疗"),
    ("600036", "招商银行"),
    ("601398", "工商银行"),
    ("600030", "中信证券"),
    ("300059", "东方财富"),
    ("601318", "中国平安"),
    ("601899", "紫金矿业"),
    ("600547", "山东黄金"),
    ("601088", "中国神华"),
    ("600309", "万华化学"),
    ("002714", "牧原股份"),
    ("601919", "中远海控"),
    ("002352", "顺丰控股"),
    ("601888", "中国中免"),
    ("000651", "格力电器"),
    ("000333", "美的集团"),
    ("000002", "万科A"),
    ("601601", "中国太保"),
    ("600050", "中国联通"),
    ("688111", "金山办公"),
    ("601127", "赛力斯"),
    ("002050", "三花智控"),
    ("300124", "汇川技术"),
    ("688027", "国盾量子"),
]


@dataclass
class StockMatch:
    code: str
    name: str

    @property
    def label(self) -> str:
        return f"{self.name}（{self.code}）"


@dataclass
class StockProfile:
    code: str
    name: str
    industry: str = ""
    info: dict[str, str] = field(default_factory=dict)
    quote: dict[str, str] = field(default_factory=dict)
    error: str = ""


def load_stock_list() -> list[tuple[str, str]]:
    """获取 A 股代码-名称列表；akshare 网络失败时退回内置列表。"""
    try:
        import akshare as ak

        stock_df = _call_with_timeout(ak.stock_info_a_code_name, timeout=30)
        pairs = [
            (str(row["code"]).zfill(6), str(row["name"]).strip())
            for _, row in stock_df.iterrows()
            if str(row.get("name", "")).strip()
        ]
        if pairs:
            return pairs
    except Exception:
        pass
    return list(FALLBACK_STOCKS)


def normalize_stock_query(query: str) -> str:
    return str(query or "").strip().replace(" ", "").upper()


def search_stocks(
    query: str,
    stock_pairs: list[tuple[str, str]],
    limit: int = 8,
) -> list[StockMatch]:
    """按代码前缀或名称子串匹配股票，精确命中排最前。"""
    normalized = normalize_stock_query(query)
    if not normalized:
        return []

    exact: list[StockMatch] = []
    partial: list[StockMatch] = []
    for code, name in stock_pairs:
        clean_name = name.replace(" ", "")
        if normalized == code or normalized == clean_name.upper():
            exact.append(StockMatch(code=code, name=name))
        elif code.startswith(normalized) or normalized in clean_name.upper():
            partial.append(StockMatch(code=code, name=name))
        if len(exact) + len(partial) >= limit * 3:
            break
    return (exact + partial)[:limit]


def fetch_stock_profile(code: str, name: str) -> StockProfile:
    """通过 akshare 获取个股基本信息与实时行情；失败时返回带错误说明的档案。"""
    profile = StockProfile(code=code, name=name)
    errors: list[str] = []

    try:
        import akshare as ak
    except Exception:
        profile.error = "akshare 未安装，无法获取个股详情"
        return profile

    try:
        info_df = _call_with_timeout(
            lambda: ak.stock_individual_info_em(symbol=code)
        )
        info = {
            str(row["item"]): str(row["value"])
            for _, row in info_df.iterrows()
        }
        profile.industry = info.get("行业", "")
        profile.info = info
    except Exception:
        errors.append("基本信息获取失败")

    try:
        quote_df = _call_with_timeout(
            lambda: ak.stock_bid_ask_em(symbol=code)
        )
        profile.quote = {
            str(row["item"]): str(row["value"])
            for _, row in quote_df.iterrows()
        }
    except Exception:
        errors.append("实时行情获取失败")

    if errors:
        profile.error = "；".join(errors) + "（可能是网络原因，稍后重试）"
    return profile


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def match_sectors_for_stock(
    stock_name: str,
    industry: str,
    sectors_config: dict[str, Any],
) -> list[str]:
    """推断股票关联的项目板块。

    优先级：个股直接映射 > 行业名桥接映射 > 行业名与板块关键词互相包含。
    只返回当前板块配置中存在的板块。
    """
    candidates: list[str] = []
    candidates.extend(STOCK_SECTOR_HINTS.get(str(stock_name).strip(), []))

    industry = str(industry or "").strip()
    if industry:
        for mapped_industry, sectors in INDUSTRY_TO_SECTORS.items():
            if mapped_industry in industry or industry in mapped_industry:
                candidates.extend(sectors)

        for sector, value in sectors_config.items():
            if sector in industry or industry in sector:
                candidates.append(sector)
                continue
            for keyword in extract_sector_keywords(value):
                if keyword and (keyword in industry or industry in keyword):
                    candidates.append(sector)
                    break

    return [
        sector
        for sector in _dedupe_keep_order(candidates)
        if sector in sectors_config
    ]


def format_market_cap(value: str) -> str:
    """把以元为单位的市值转成“x.xx 亿 / x.xx 万亿”。非数字原样返回。"""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value or "-")
    if amount <= 0:
        return "-"
    if amount >= 1e12:
        return f"{amount / 1e12:.2f} 万亿"
    if amount >= 1e8:
        return f"{amount / 1e8:.2f} 亿"
    return f"{amount / 1e4:.2f} 万"
