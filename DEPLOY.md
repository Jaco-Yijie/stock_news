# A 股板块新闻系统部署说明

## Streamlit Community Cloud

1. 将项目推送到 GitHub 仓库。
2. 确认仓库包含：
   - `main.py`
   - `requirements.txt`
   - `fetcher.py`
   - `news_store.py`
   - `config_store.py`
   - `sectors.py`
   - `paths.py`
3. 在 Streamlit Community Cloud 创建新应用。
4. 选择仓库、分支，并设置入口文件为：
   ```bash
   main.py
   ```
5. 部署后应用会默认使用项目目录下的 `./data` 保存缓存和配置。

说明：Streamlit Community Cloud 的本地文件存储不适合作为长期持久化存储，应用重建后缓存和在线编辑的配置可能会丢失。

## Render

1. 将项目推送到 GitHub 仓库。
2. 在 Render 创建 Web Service。
3. Build command 可留空，或设置为：
   ```bash
   pip install -r requirements.txt
   ```
4. Start command 设置为：
   ```bash
   streamlit run main.py --server.port $PORT --server.address 0.0.0.0
   ```
5. 如果需要保留新闻缓存和页面中编辑的配置，建议添加 Persistent Disk，并设置环境变量：
   ```bash
   DATA_DIR=/var/data
   ```

## 数据目录

应用默认使用：
```bash
./data
```

如果设置了环境变量 `DATA_DIR`，则会改用该目录保存：
- `news_cache.csv`
- `sectors_config.json`
- `events_config.json`

Render persistent disk 推荐：
```bash
DATA_DIR=/var/data
```
