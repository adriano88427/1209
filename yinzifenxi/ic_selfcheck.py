# -*- coding: utf-8 -*-
"""
轻量 IC 自校验工具：
- 读取日度 IC 明细（由主流程开启 IC_DUMP_DAILY_ENABLED 生成的 ic_daily_dump.csv）
- 计算等权/样本加权 IC 均值
- 与主流程汇总对比，输出差异表
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def load_daily_ic(path: str, view: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"daily IC file not found: {path}")
    df = pd.read_csv(path)
    required = {"factor", "ic", "samples"}
    if not required.issubset(df.columns):
        raise ValueError(f"daily IC file缺少列: {required - set(df.columns)}")
    if view:
        df = df[df.get("view", "").astype(str).str.lower() == view.lower()]
    return df


def compute_daily_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    grouped = daily_df.groupby("factor", as_index=False)
    agg = grouped.agg(
        ic_mean_self=("ic", "mean"),
        daily_points_self=("ic", "size"),
        samples_total_self=("samples", "sum"),
    )
    weighted = (
        daily_df.assign(ic=lambda d: d["ic"].astype(float), samples=lambda d: d["samples"].astype(float))
        .groupby("factor")
        .apply(lambda g: np.average(g["ic"], weights=g["samples"]) if g["samples"].sum() > 0 else np.nan)
        .reset_index(name="ic_mean_weighted_self")
    )
    result = agg.merge(weighted, on="factor", how="left")
    result.rename(columns={"factor": "因子名称"}, inplace=True)
    return result


def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"summary file not found: {path}")
    df = pd.read_csv(path)
    required = {"因子名称", "IC均值", "加权IC均值"}
    if not required.issubset(df.columns):
        raise ValueError(f"summary file缺少列: {required - set(df.columns)}")
    return df[["因子名称", "IC均值", "加权IC均值"]]


def build_compare(summary_df: pd.DataFrame, self_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(self_df, on="因子名称", how="left")
    merged["ic_mean_diff"] = merged["IC均值"] - merged["ic_mean_self"]
    merged["ic_mean_weighted_diff"] = merged["加权IC均值"] - merged["ic_mean_weighted_self"]
    return merged


def main():
    parser = argparse.ArgumentParser(description="IC 自校验（基于日度 IC 落盘文件）")
    parser.add_argument("--daily-ic", required=True, help="日度 IC 落盘文件（IC日度明细.csv）")
    parser.add_argument("--summary", required=True, help="主流程汇总 CSV（因子分析汇总_*.csv）")
    parser.add_argument("--output", default="baogao/IC自校验对比.csv", help="对比输出 CSV")
    parser.add_argument("--view", default="neutralized", help="过滤日度IC视图（默认 neutralized），如需 raw 可指定 raw")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="差异容忍阈值（绝对值），默认1e-4")
    args = parser.parse_args()

    daily_df = load_daily_ic(args.daily_ic, view=args.view)
    self_summary = compute_daily_summary(daily_df)
    summary_df = load_summary(args.summary)
    compare_df = build_compare(summary_df, self_summary)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    compare_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[SELFCHK] compare saved to {args.output}")
    max_diff = compare_df[["ic_mean_diff", "ic_mean_weighted_diff"]].abs().max().max()
    if max_diff <= args.tolerance:
        print(f"[SELFCHK] diffs within tolerance (<= {args.tolerance})")
    else:
        print(f"[SELFCHK] found differences (max={max_diff}), please review the output CSV or adjust tolerance")


if __name__ == "__main__":
    main()
