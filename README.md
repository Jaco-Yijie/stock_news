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
Code -> Download ZIP
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
- Filter by sector, keyword, source, news type, and time range
- Edit sector keywords from the web interface
- Add or remove custom sectors
- Track external events such as Apple orders, NVIDIA GPU, US sanctions, Fed rate decisions, and oil price changes
- Map external events to related A-share sectors
- Optionally use LLM verification through environment variables
- Streamlit-based web dashboard
- Deployment-ready with configurable `DATA_DIR`

利好利空规则优先判断标题中的已确认事件。标题没有明确方向时才用正文补充。
价格、货币政策和贸易事件按相关板块判断；同一条新闻对上下游方向相反时，
看板同时显示各板块方向和整体中性判断。

## Streamlit Community Cloud Deployment

推荐优先使用 Streamlit Community Cloud。

创建应用时填写：

- Repository：`stock_news`
- Branch：`main`
- Main file path：`main.py`
- Start Command：不需要填写，Streamlit Cloud 会自动根据 `main.py` 启动

注意事项：

- Streamlit Community Cloud 12 小时无访问会休眠。
- Streamlit Cloud 上的本地 `./data` 文件不适合长期线上持久化；应用重建、重启或环境变化时，本地文件缓存和在线编辑的配置可能丢失。
- 新闻缓存持久化已支持 Supabase：配置 `SUPABASE_URL` 和 `SUPABASE_KEY` 后缓存保存在 Supabase Postgres 中，重启不丢失。配置步骤见 [DEPLOY.md](DEPLOY.md)。

## Agent 对比与 bad case 评测

```bash
python -m agent.run_agent --sectors 半导体芯片 人工智能 --dry-run --compare
python -m agent.compare_report
python -m agent.bad_case stats
python -m agent.bad_case evaluate --trace agent_trace.jsonl
```

- 两侧使用同批候选，比较排序取前 10 条之前的分类结果。报告分别统计决策模型和工具调用。
- 对比报告把正常走模型决策的差异加入 `data/bad_cases.jsonl`。旧 trace 或规则兜底产生的差异不会入库。
- 人工审核后，将 `type` 从 `pending` 改为 `miss`（应该保留但被剔除）、`false_positive`（不该保留但被保留）或 `dismissed`（差异合理），在 `note` 记录判断依据。同一板块和链接的既有标签会保留。
- Critic 会读取同板块最多 10 条已确认的 `false_positive` 标题。评测命令按候选快照核对人工标注的去留，输出通过、失败和未覆盖数量。
- Actions 的对比任务会随报告上传当次的案例文件。可下载其中的报告和 trace，用 `python -m agent.bad_case collect --report <下载的报告路径>` 合并到本地人工标注库。正式比对前需在运行环境配置已有的 LLM 环境变量。

### Telegram 人工反馈

- 重要新闻和早报按条目编号显示「该留／不该留」按钮。Agent 日报会另发一条带原文链接的审核列表。
- 「该留」记录为 `kept`，「不该留」记录为 `false_positive`。同一聊天、同一天、同一板块和链接保留最近一次选择。漏推（`miss`）由对比报告审核。
- `python feedback_bot.py --export` 将反馈导出为 `data/telegram_feedback_cases.jsonl`。Agent 与离线评测默认读取此文件。同一案例有已审核的本地标签时，以本地标签为准。
- Actions 的对比任务在已配置 Supabase 时自动读取反馈。反馈 JSONL 保存在当前运行环境，不作为 Actions 产物上传。部署步骤见 [DEPLOY.md](DEPLOY.md#telegram-人工反馈)。

## Project Structure

```text
stock_news/
├── main.py              # Streamlit web interface
├── fetcher.py           # News fetching and deduplication logic
├── sectors.py           # Default sector and external event configuration
├── news_store.py        # News cache read/write logic (local CSV or Supabase)
├── supabase_store.py    # Supabase (PostgREST) cache backend
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
半导体芯片 -> 半导体、芯片、集成电路、晶圆
新能源汽车 -> 新能源汽车、电动车、锂电池、充电桩
人工智能 -> 人工智能、大模型、AI芯片、算力
```

### External Events

External events are news that may indirectly affect A-share sectors.

Examples:

```text
Apple orders -> 消费电子、半导体芯片
NVIDIA GPU -> 人工智能、算力数据中心、半导体芯片
US sanctions -> 半导体芯片、信创软件、军工
Fed rate decisions -> 银行、证券、黄金、有色金属
Oil price changes -> 石油石化、化工、航运港口
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

You can still override the data directory with `DATA_DIR` when running on your own infrastructure:

```bash
DATA_DIR=/path/to/data
```

Do not commit `data/news_cache.csv` to Git.

## Notes

- The app depends on third-party data sources, so news fetching may fail when the upstream API is unstable.
- Incremental refresh depends on `publish_time`.
- Local cache files should not be committed to Git.
- Local files under `./data` are not reliable long-term storage on Streamlit Cloud.
- This project is for information monitoring only and does not provide financial advice.

## Deploy

See [DEPLOY.md](DEPLOY.md) for deployment instructions.
