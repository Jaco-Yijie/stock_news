SECTORS = {
    "半导体芯片": ["半导体", "芯片", "集成电路", "晶圆"],
    "商业航天": ["商业航天", "卫星互联网", "火箭", "航天器"],
    "脑机接口": ["脑机接口", "BCI", "神经接口"],
    "新能源汽车": ["新能源汽车", "电动车", "锂电池", "充电桩"],
    "人工智能": ["人工智能", "大模型", "AI芯片", "算力"],
    "创新药": ["创新药", "生物医药", "ADC", "临床试验"],
    "量子计算": ["量子计算", "量子芯片", "量子通信"],
    "低空经济": ["低空经济", "eVTOL", "飞行汽车", "无人机"],
    "机器人": ["机器人", "人形机器人", "工业机器人", "减速器"],
    "光伏": ["光伏", "太阳能", "硅片", "组件"],
    "风电": ["风电", "海上风电", "风机", "风电设备"],
    "储能": ["储能", "新型储能", "电池储能", "储能系统"],
    "电力设备": ["电力设备", "特高压", "智能电网", "变压器"],
    "军工": ["军工", "国防军工", "航空发动机", "导弹"],
    "消费电子": ["消费电子", "智能手机", "MR", "可穿戴设备"],
    "算力数据中心": ["算力", "数据中心", "服务器", "液冷"],
    "信创软件": ["信创", "国产软件", "操作系统", "数据库"],
    "网络安全": ["网络安全", "数据安全", "信息安全", "密码"],
    "数字经济": ["数字经济", "数据要素", "数字中国", "政务云"],
    "传媒游戏": ["传媒", "游戏", "影视", "短剧"],
    "医药医疗": ["医药", "医疗器械", "医院", "医保"],
    "CXO": ["CXO", "CRO", "CDMO", "医药外包"],
    "白酒消费": ["白酒", "消费", "食品饮料", "零售"],
    "家电": ["家电", "白色家电", "空调", "冰箱"],
    "房地产": ["房地产", "地产", "楼市", "保障房"],
    "银行": ["银行", "商业银行", "存款", "贷款"],
    "证券": ["证券", "券商", "资本市场", "投行"],
    "保险": ["保险", "险资", "寿险", "财险"],
    "有色金属": ["有色金属", "铜", "铝", "锂"],
    "黄金": ["黄金", "金价", "贵金属", "央行购金"],
    "煤炭": ["煤炭", "煤价", "焦煤", "动力煤"],
    "钢铁": ["钢铁", "钢材", "铁矿石", "螺纹钢"],
    "化工": ["化工", "化学品", "磷化工", "氟化工"],
    "农业": ["农业", "种业", "猪肉", "粮食"],
    "航运港口": ["航运", "港口", "集运", "海运"],
    "物流快递": ["物流", "快递", "供应链", "仓储"],
    "旅游酒店": ["旅游", "酒店", "景区", "免税"],
    "环保水务": ["环保", "水务", "污水处理", "固废"],
}


EXTERNAL_EVENTS = {
    "苹果产业链": [
        {
            "positiveKeywords": ["苹果订单", "iPhone", "苹果供应链", "Apple 供应链"],
            "requiredCoKeywords": [],
            "negativeKeywords": ["海马汽车"],
            "weight": 2,
            "minScore": 2,
            "fields": ["title", "summary", "content"],
        },
        {
            "positiveKeywords": ["Apple"],
            "requiredCoKeywords": ["iPhone", "Vision Pro", "苹果供应链", "Apple 供应链", "产业链", "供应链"],
            "negativeKeywords": ["海马汽车"],
            "weight": 1,
            "minScore": 1,
            "fields": ["title", "summary", "content"],
        },
        {
            "positiveKeywords": ["Vision Pro"],
            "requiredCoKeywords": [],
            "negativeKeywords": ["海马汽车"],
            "weight": 2,
            "minScore": 2,
            "fields": ["title", "summary", "content"],
        },
    ],
    "AI产业链": [
        {
            "positiveKeywords": ["NVIDIA", "英伟达", "GPU", "AI服务器", "HBM"],
            "requiredCoKeywords": [],
            "negativeKeywords": [],
            "weight": 1,
            "minScore": 1,
            "fields": ["title", "summary", "content"],
        }
    ],
    "中美关系/制裁": [
        {
            "positiveKeywords": ["美国制裁", "出口管制", "关税", "中美关系", "美国总统"],
            "requiredCoKeywords": [],
            "negativeKeywords": [],
            "weight": 1,
            "minScore": 1,
            "fields": ["title", "summary", "content"],
        }
    ],
    "美联储/利率": [
        {
            "positiveKeywords": ["美联储", "FOMC", "降息", "加息", "鲍威尔", "美债收益率"],
            "requiredCoKeywords": [],
            "negativeKeywords": [],
            "weight": 2,
            "minScore": 2,
            "fields": ["title", "summary", "content"],
        },
        {
            "positiveKeywords": ["美元"],
            "requiredCoKeywords": ["美联储", "FOMC", "利率", "降息", "加息", "鲍威尔"],
            "negativeKeywords": ["中美元首", "中美关系", "卢比兑美元", "兑美元"],
            "weight": 1,
            "minScore": 1,
            "fields": ["title", "summary", "content"],
        },
    ],
    "油价能源": [
        {
            "positiveKeywords": ["油价", "原油", "OPEC", "布伦特原油", "WTI"],
            "requiredCoKeywords": [],
            "negativeKeywords": [],
            "weight": 1,
            "minScore": 1,
            "fields": ["title", "summary", "content"],
        }
    ],
}


EVENT_TO_SECTORS = {
    "苹果产业链": ["消费电子", "半导体芯片"],
    "AI产业链": ["人工智能", "算力数据中心", "半导体芯片"],
    "中美关系/制裁": ["半导体芯片", "信创软件", "军工"],
    "美联储/利率": ["银行", "证券", "黄金", "有色金属"],
    "油价能源": ["石油石化", "化工", "航运港口"],
}
