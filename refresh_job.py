"""后台定时抓取脚本：供 GitHub Actions 或本地 cron 调用。

用法：
    python refresh_job.py                       # 增量抓取全部板块 + 外部事件
    python refresh_job.py --mode full           # 全量重建缓存
    python refresh_job.py --retention-days 60   # 调整缓存保留天数（0 表示不清理）

需要配置 SUPABASE_URL / SUPABASE_KEY 环境变量才能持久化到 Supabase，
未配置时写入本地 data/news_cache.csv（在 CI 上不会保留）。
"""
from __future__ import annotations

import argparse
import sys

from config_store import try_load_events_config, try_load_sectors_config
from llm_provider import load_llm_verifier_from_env
from news_store import cache_backend_name, save_cache
from refresh import prune_old_cache, run_full_refresh, run_incremental_refresh


def _print_warnings(label: str, warnings_by_key: dict[str, list[str]]) -> None:
    for key, warnings in warnings_by_key.items():
        for warning in warnings:
            print(f"[warning] {label}「{key}」：{warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="后台抓取新闻并写入缓存")
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
        help="incremental：合并新增新闻（默认）；full：全量重建缓存",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="缓存保留天数，超过的旧新闻会被清理；0 表示不清理（默认 30）",
    )
    args = parser.parse_args(argv)

    sectors_config, sectors_config_error = try_load_sectors_config()
    events_config, events_config_error = try_load_events_config()
    for error in (sectors_config_error, events_config_error):
        if error:
            print(f"[warning] {error}")

    llm_verifier, llm_notice = load_llm_verifier_from_env()
    if llm_notice:
        print(f"[info] {llm_notice}")

    backend = cache_backend_name()
    selected_sectors = list(sectors_config)
    print(
        f"[info] 缓存后端：{backend}，模式：{args.mode}，"
        f"板块 {len(selected_sectors)} 个，事件类别 {len(events_config['external_events'])} 个"
    )
    if backend != "Supabase":
        print("[warning] 未配置 Supabase，本次结果写入本地文件，在 CI 环境中不会保留。")

    refresh = run_full_refresh if args.mode == "full" else run_incremental_refresh
    outcome = refresh(
        selected_sectors,
        sectors_config,
        events_config["external_events"],
        events_config["event_to_sectors"],
        llm_verifier=llm_verifier,
    )

    _print_warnings("板块", outcome.sector_warnings)
    _print_warnings("事件", outcome.event_warnings)

    if outcome.fetch_failed:
        print("[error] 全量抓取未获取到有效数据，保留现有缓存，本次不写入。")
        return 1

    final_cache = prune_old_cache(outcome.cache, args.retention_days)
    pruned_count = len(outcome.cache) - len(final_cache)
    save_cache(final_cache)
    print(
        f"[info] 完成：新增 {outcome.added_count} 条，"
        f"清理过期 {pruned_count} 条，缓存共 {len(final_cache)} 条。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
