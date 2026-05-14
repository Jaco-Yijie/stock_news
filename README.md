## Quick Start / How to use

### 1. Download project

```bash
git clone https://github.com/wangyijie072022-cloud/stock_news.git
cd stock_news

### 2. Create a Python virtual environment
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

# stock_news

A Streamlit dashboard for collecting, caching, filtering, and deduplicating A-share sector news.

This project is used to collect A-share related sector news, and provides local caching, incremental refresh, keyword management, external event mapping and news filtering functions. It is suitable for use as a personal news monitoring panel to help you quickly view the latest information in different sections.

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
