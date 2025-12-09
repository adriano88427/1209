# -*- coding: utf-8 -*-
"""
从原始数据构造价格面板（宽表/长表），便于外部基准使用。
说明：
- 使用 FactorAnalysis 预处理，读取配置中的价格列别名。
- 仅利用信号日及后两交易日的价格列，按 (信号日 + offset) 生成“伪日期”。
  若需真实交易日对齐，请使用完整的行情时间序列替换此结果。
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from typing import Dict, List

import pandas as pd

from yinzifenxi.fa_config import DATA_PARSE_CONFIG, REPORT_OUTPUT_DIR
from yinzifenxi.fa_nonparam_analysis import FactorAnalysis


def normalize_price_columns(df: pd.DataFrame) -> Dict[str, str]:
    """返回实际存在的价格列映射：标准名 -> 实际列名"""
    aliases = DATA_PARSE_CONFIG.get("price_aliases", {}) or {}
    mapping: Dict[str, str] = {}
    for std_name, alias_list in aliases.items():
        candidates = [std_name] + list(alias_list or [])
        for c in candidates:
            if c in df.columns:
                mapping[std_name] = c
                break
    return mapping


def build_price_long(df: pd.DataFrame, code_col: str, price_map: Dict[str, str]) -> pd.DataFrame:
    """构造长表：date(伪), asset, price, horizon"""
    records: List[Dict] = []
    base_dates = pd.to_datetime(df["信号日期"], errors="coerce")
    codes = df[code_col]
    # 价格优先级（避免同日 open/close 等重复时随机覆盖，收盘优先）
    priority = {
        "当日收盘价": 0,
        "当日开盘价": 1,
        "后第1交易日收盘价": 0,
        "后第1交易日开盘价": 1,
        "后第2交易日收盘价": 0,
        "后第2交易日开盘价": 1,
    }

    def add_rows(std_name: str, offset_days: int):
        if std_name not in price_map:
            return
        col = price_map[std_name]
        prices = df[col]
        for d, code, px in zip(base_dates, codes, prices):
            if pd.isna(d) or pd.isna(px):
                continue
            records.append(
                {
                    "date": d + timedelta(days=offset_days),
                    "asset": str(code),
                    "price": float(px),
                    "horizon": offset_days,
                    "source_col": col,
                    "source_priority": priority.get(std_name, 99),
                }
            )

    add_rows("当日收盘价", 0)
    add_rows("当日开盘价", 0)
    add_rows("后第1交易日开盘价", 1)
    add_rows("后第1交易日收盘价", 1)
    add_rows("后第2交易日开盘价", 2)
    add_rows("后第2交易日收盘价", 2)

    price_long = pd.DataFrame(records)
    # 去重：同 date/asset 保留优先级最高的价格（收盘优先）
    if not price_long.empty:
        price_long = price_long.sort_values(["date", "asset", "source_priority"])
        before = len(price_long)
        price_long = price_long.drop_duplicates(subset=["date", "asset"], keep="first")
        after = len(price_long)
        price_long = price_long.drop(columns=["source_priority"])
        print(f"[PRICE] 长表去重 {before-after}/{before} (date+asset，收盘优先)")
    return price_long


def export_long_wide(price_long: pd.DataFrame, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    long_path = os.path.join(output_dir, f"{prefix}_long.csv")
    wide_path = os.path.join(output_dir, f"{prefix}_wide.csv")

    price_long.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"[PRICE] 长表输出: {long_path} rows={len(price_long)}")

    wide = price_long.pivot_table(index="date", columns="asset", values="price")
    wide.to_csv(wide_path, encoding="utf-8-sig")
    print(f"[PRICE] 宽表输出: {wide_path} shape={wide.shape}")


def main():
    parser = argparse.ArgumentParser(description="构造价格面板（宽表/长表）")
    parser.add_argument("--output-dir", default=REPORT_OUTPUT_DIR, help="输出目录，默认 baogao/")
    parser.add_argument("--prefix", default="benchmark_price", help="输出文件前缀")
    args = parser.parse_args()

    fa = FactorAnalysis()
    ok = fa.preprocess_data()
    if not ok or fa.data is None:
        raise SystemExit("预处理失败，无法构造价格面板")
    df = fa.data.copy()
    if "信号日期" not in df.columns:
        raise SystemExit("缺少列 信号日期")
    code_col = "股票代码" if "股票代码" in df.columns else ("证券代码" if "证券代码" in df.columns else None)
    if not code_col:
        raise SystemExit("缺少股票代码列")

    price_map = normalize_price_columns(df)
    if not price_map:
        raise SystemExit("未找到任何价格列，请检查配置中的 price_aliases 或源表列名")
    print(f"[PRICE] 识别到价格列映射: {price_map}")

    price_long = build_price_long(df, code_col, price_map)
    if price_long.empty:
        raise SystemExit("价格长表为空，无法导出")
    export_long_wide(price_long, args.output_dir, args.prefix)
    print("[PRICE] 注意：使用信号日及后两日价格生成的伪日期；若需真实交易日对齐，请使用完整行情序列。")


if __name__ == "__main__":
    main()
