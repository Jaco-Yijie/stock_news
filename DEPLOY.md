# A 股板块新闻系统部署说明

## 推荐部署方式：Streamlit Community Cloud

本项目优先部署到 Streamlit Community Cloud。

创建应用时填写：

- Repository：`stock_news`
- Branch：`main`
- Main file path：`main.py`
- Start Command：不需要填写

Streamlit Community Cloud 会根据 `requirements.txt` 安装依赖，并自动运行 `main.py`，不需要配置 Render 这类平台使用的专属 start command。

## 部署步骤

1. 将项目推送到 GitHub 仓库 `stock_news`。
2. 打开 Streamlit Community Cloud，创建新应用。
3. Repository 选择 `stock_news`。
4. Branch 选择 `main`。
5. Main file path 填写 `main.py`。
6. 不填写 Start Command。
7. 部署完成后访问 Cloud 生成的应用地址。

## 数据目录和持久化

应用默认使用项目目录下的 `./data` 保存缓存和配置：

- `data/news_cache.csv`
- `data/sectors_config.json`
- `data/events_config.json`

项目仍然支持通过环境变量 `DATA_DIR` 指定数据目录：

```bash
DATA_DIR=/path/to/data
```

但在 Streamlit Community Cloud 上，不建议沿用 Render 这类平台的持久盘路径配置。Streamlit Cloud 的本地文件系统不适合长期线上持久化；应用重建、重启或环境变化时，本地文件缓存和页面中编辑的配置可能丢失。

新闻缓存的长期持久化请使用下面的 Supabase 配置。

## 新闻缓存持久化：Supabase

配置了 Supabase 后，新闻缓存改为保存在 Supabase 的 Postgres 表中，应用重启、休眠唤醒后缓存不再丢失。未配置时自动回退到本地 `data/news_cache.csv`，本地开发不受影响。

### 1. 创建 Supabase 项目

打开 [supabase.com](https://supabase.com)，用 GitHub 账号登录，创建一个免费项目（Region 建议选 Singapore 或 Tokyo，离国内数据源更近）。

### 2. 创建 news_cache 表

进入项目的 SQL Editor，执行：

```sql
create table if not exists news_cache (
  id text primary key,
  news_type text not null default 'sector_news',
  sector text not null default '',
  title text not null default '',
  source text not null default '',
  publish_time text not null default '',
  link text not null default '',
  keyword text not null default '',
  content text not null default '',
  event_category text not null default '',
  related_sectors text not null default '',
  reason text not null default '',
  fetched_at text not null default ''
);

alter table news_cache enable row level security;
```

开启 RLS（Row Level Security，行级安全）且不加任何策略，意味着匿名 key 无法读写这张表，只有下面的 service_role key（服务端密钥）可以，这是预期的安全配置。

### 3. 获取连接信息

在 Supabase 项目的 Settings 中获取两个值：

- `SUPABASE_URL`：Settings → Data API 中的 Project URL，形如 `https://xxxx.supabase.co`
- `SUPABASE_KEY`：Settings → API Keys 中的 `service_role` key（注意不是 anon key）

service_role key 拥有完整读写权限，只能放在服务端环境变量或 Streamlit secrets 中，绝不能提交到 Git 或暴露给浏览器。

### 4. 配置密钥

Streamlit Community Cloud：应用页面右下角 Manage app → Settings → Secrets，填入：

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "service_role key"
```

本地运行：在 shell 中导出环境变量（或写入不提交的 `.env` 并自行加载）：

```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_KEY="service_role key"
```

配置生效后，页面上的「缓存后端」会从「本地文件」变为「Supabase」。

## Streamlit Cloud 休眠说明

Streamlit Community Cloud 12 小时无访问会休眠。休眠后再次访问会自动唤醒，但本地文件不应被视为可靠的长期存储。

## Render 说明

Render 不再作为本项目的首选部署方式。若后续确实要使用 Render，应按 Render 官方文档自行配置 Web Service、端口和持久化磁盘；本项目文档不再主推 Render 专属 start command。
