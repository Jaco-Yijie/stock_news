# stock_news

A Streamlit dashboard for collecting, caching, filtering, and deduplicating A-share sector news.

这个项目用于收集 A 股相关板块新闻，并提供本地缓存、增量刷新、关键词管理、外部事件映射和新闻筛选功能。适合用作个人新闻监控面板。

## Quick Start / 如何使用

### 1. Download project / 下载项目

```bash
git clone https://github.com/wangyijie072022-cloud/stock_news.git
cd stock_news
```

如果不会用 Git，也可以在 GitHub 页面点击：

```text
Code → Download ZIP
```

下载后解压，再进入项目文件夹。

### 2. Create a Python virtual environment / 创建 Python 虚拟环境

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 4. Run the app / 启动应用

```bash
streamlit run main.py
```

启动后浏览器打开：

```text
http://localhost:8501
```

## Features

- Collect A-share sector news with AKShare
- Group news by sector
- Support incremental refresh to avoid repeated full crawling
- Save news cache locally
- Deduplicate repeated news from different links or sources
- Filter by sector, keyword, source, and news type
- Edit sector keywords from the web interface
- Add or remove custom sectors
- Track external events such as Apple orders, NVIDIA GPU, US sanctions, Fed rate decisions, and oil price changes
- Map external events to related A-share sectors
- Streamlit-based web dashboard
- Deployment-ready with configurable `DATA_DIR`

## Project Structure

```text
stock_news/
├── main.py              # Streamlit web interface
├── fetcher.py           # News fetching and deduplication logic
├── sectors.py           # Default sector and external event configuration
├── news_store.py        # Local news cache read/write logic
├── config_store.py      # Editable sector/event config storage
├── paths.py             # Shared data directory path config
├── requirements.txt     # Python dependencies
├── DEPLOY.md            # Deployment instructions
├── .gitignore           # Git ignore rules
└── data/                # Local cache/config directory
```

## Data Sources

The app mainly uses AKShare to fetch Eastmoney news data.

Current data source:

- AKShare
- Eastmoney news search data through AKShare/fallback logic

The app does not provide investment advice. All news should be treated as information references only.

## Core Concepts

### Sector News

Sector news is collected by matching sector keywords.

Example:

```text
半导体芯片 → 半导体、芯片、集成电路、晶圆
新能源汽车 → 新能源汽车、电动车、锂电池、充电桩
人工智能 → 人工智能、大模型、AI芯片、算力
```

### External Events

External events are news that may indirectly affect A-share sectors.

Examples:

```text
Apple orders → 消费电子、半导体芯片
NVIDIA GPU → 人工智能、算力数据中心、半导体芯片
US sanctions → 半导体芯片、信创软件、军工
Fed rate decisions → 银行、证券、黄金、有色金属
Oil price changes → 石油石化、化工、航运港口
```

These events are shown separately from regular sector news and mapped to related A-share sectors.

## Data Directory

By default, the app stores local data under:

```text
./data
```

It may contain:

```text
data/news_cache.csv
data/sectors_config.json
data/events_config.json
```

For deployment, you can override the data directory with:

```bash
DATA_DIR=/var/data
```

## Notes

- The app depends on third-party data sources, so news fetching may fail when the upstream API is unstable.
- Incremental refresh depends on `publish_time`.
- Local cache files should not be committed to Git.
- This project is for information monitoring only and does not provide financial advice.

## Deploy

See [DEPLOY.md](DEPLOY.md) for deployment instructions.
