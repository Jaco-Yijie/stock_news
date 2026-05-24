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

但在 Streamlit Community Cloud 上，不建议沿用 Render 这类平台的持久盘路径配置。Streamlit Cloud 的本地文件系统不适合长期线上持久化；应用重建、重启或环境变化时，缓存和页面中编辑的配置可能丢失。

长期保存新闻缓存和配置，建议后续接 Supabase 或 Neon。

## Streamlit Cloud 休眠说明

Streamlit Community Cloud 12 小时无访问会休眠。休眠后再次访问会自动唤醒，但本地文件不应被视为可靠的长期存储。

## Render 说明

Render 不再作为本项目的首选部署方式。若后续确实要使用 Render，应按 Render 官方文档自行配置 Web Service、端口和持久化磁盘；本项目文档不再主推 Render 专属 start command。
