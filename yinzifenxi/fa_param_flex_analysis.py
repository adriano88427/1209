# -*- coding: utf-8 -*-
"""
带参数单因子自由区间挖掘模块
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .fa_config import RETURN_COLUMN, SINGLE_PARAM_FLEX_SETTINGS, validate_single_param_flex_config
from .fa_logging import detect_debug_enabled
from .fa_stat_utils import calc_max_drawdown
from .fa_param_flex_report import generate_single_param_flex_reports


def _debug(msg: str):
    if detect_debug_enabled():
        print(f"[DEBUG][SINGLE-PARAM-FLEX] {msg}")


def _compute_min_samples(cfg: Dict[str, Any], total_samples: int) -> int:
    mode = str(cfg.get("min_samples_mode", "auto")).lower()
    if mode == "fixed":
        return max(1, int(cfg.get("min_samples_fixed", 500)))
    # auto: 总样本 / 20
    return max(1, int(total_samples / 20))


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _calc_metrics(df: pd.DataFrame, return_col: str) -> Dict[str, Any]:
    returns = _safe_numeric(df[return_col]).dropna()
    if returns.empty:
        return {}
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=1)
    annual_return = float(daily_mean * 252) if pd.notna(daily_mean) else np.nan
    if pd.notna(daily_std) and daily_std > 0 and pd.notna(daily_mean):
        sharpe = float((daily_mean / daily_std) * math.sqrt(252))
    elif pd.notna(daily_mean):
        sharpe = float("inf") if daily_mean > 0 else 0.0
    else:
        sharpe = np.nan
    max_dd = calc_max_drawdown(returns)
    return {
        "annual_return": annual_return,
        "avg_return": float(daily_mean) if pd.notna(daily_mean) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": float(max_dd) if max_dd is not None else np.nan,
        "samples": len(returns),
    }


def _build_intervals(series: pd.Series, bins: int, strategy: str) -> List[pd.Interval]:
    if series.empty:
        return []
    intervals: List[pd.Interval] = []
    if strategy == "quantile":
        try:
            cat = pd.qcut(series, bins, duplicates="drop")
            intervals = list(cat.cat.categories)
        except Exception:
            strategy = "equal"
    if strategy == "equal":
        try:
            cat = pd.cut(series, bins, duplicates="drop")
            intervals = list(cat.cat.categories)
        except Exception:
            intervals = []
    return intervals


def _format_interval(interval: pd.Interval, as_percent: bool = True) -> str:
    if pd.isna(interval):
        return "--"
    def _fmt(x):
        if not isinstance(x, (int, float)):
            return str(x)
        return f"{x*100:.2f}%" if as_percent else f"{x:.4f}"
    left = _fmt(interval.left)
    right = _fmt(interval.right)
    return f"[{left}, {right}]"


def run_single_param_flex(parameterized_analyzer, flex_config: Dict[str, Any], report_options: Dict[str, Any]):
    start = time.perf_counter()
    cfg = validate_single_param_flex_config(flex_config)
    df = getattr(parameterized_analyzer, "processed_data", None)
    if df is None or df.empty:
        print("[WARN] 单因子自由区间挖掘缺少 processed_data，已跳过")
        return {}

    if RETURN_COLUMN not in df.columns:
        print(f"[WARN] 缺少收益列 {RETURN_COLUMN}，已跳过单因子自由区间挖掘")
        return {}

    df = df.copy()
    df[RETURN_COLUMN] = _safe_numeric(df[RETURN_COLUMN])
    df = df.dropna(subset=[RETURN_COLUMN])
    total_samples = len(df)
    min_samples = _compute_min_samples(cfg, total_samples)
    print(f"[FLEX] 总样本 {total_samples}，样本下限 {min_samples}（mode={cfg.get('min_samples_mode','auto')}）")
    overall_metrics = _calc_metrics(df, RETURN_COLUMN)

    factors = getattr(parameterized_analyzer, "factors", [])
    if not factors:
        print("[WARN][FLEX] 无有效因子列表")
        return {}

    default_bins = cfg.get("default_bins", 8)
    bin_strategy = cfg.get("bin_strategy", "quantile")
    max_ranges = cfg.get("max_ranges_per_factor", 5)
    enable_user = cfg.get("enable_user_ranges", True)
    user_ranges = cfg.get("user_ranges", {}) or {}

    results: List[Dict[str, Any]] = []
    filter_stats = {"low_sample": 0, "total_ranges": 0, "merged": 0}

    for factor in factors:
        if factor not in df.columns:
            _debug(f"{factor} 不在数据列中，跳过")
            continue
        series = _safe_numeric(df[factor])
        returns = df[RETURN_COLUMN]
        base_mask = series.notna() & returns.notna()
        if "信号日期" in df.columns:
            dates = pd.to_datetime(df["信号日期"], errors="coerce")
        else:
            dates = None
        factor_series = series[base_mask]
        return_series = returns[base_mask]
        if factor_series.empty:
            continue

        ranges: List[Tuple[str, pd.Index, bool]] = []

        # 0 值单独成档
        zero_mask = factor_series == 0
        if zero_mask.any():
            ranges.append(("[0]", factor_series[zero_mask].index, False))

        nonzero = factor_series[~zero_mask]
        intervals = _build_intervals(nonzero, default_bins, bin_strategy)
        for interval in intervals:
            idx = nonzero[(nonzero >= interval.left) & (nonzero <= interval.right)].index
            ranges.append((_format_interval(interval, as_percent=True), idx, False))

        # 用户区间
        if enable_user and factor in user_ranges:
            for rng in user_ranges.get(factor, []):
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    continue
                low, high = rng
                idx = factor_series[(factor_series >= low) & (factor_series <= high)].index
                label = f"[{low*100:.2f}%,{high*100:.2f}%]"
                ranges.append((label, idx, True))

        # 去重并截断上限
        dedup = []
        seen = set()
        non_user_count = 0
        for label, idx, is_user in ranges:
            if not len(idx):
                continue
            key = (label, is_user)
            if key in seen:
                continue
            # 非用户区间受 max_ranges 限制，用户区间不受限制
            if (not is_user) and non_user_count >= max_ranges:
                continue
            seen.add(key)
            dedup.append((label, idx, is_user))
            if not is_user:
                non_user_count += 1

        for label, idx, is_user in dedup:
            subset = df.loc[idx]
            samples = len(subset)
            filter_stats["total_ranges"] += 1
            if (samples < min_samples) and (not is_user):
                filter_stats["low_sample"] += 1
                continue

            metrics = _calc_metrics(subset, RETURN_COLUMN)
            if not metrics:
                continue
            trade_days = subset["信号日期"].nunique() if "信号日期" in subset.columns else samples
            years = None
            if dates is not None:
                year_vals = dates.loc[idx].dt.year.dropna().unique()
                if len(year_vals):
                    years = f"{year_vals.min()}-{year_vals.max()}" if len(year_vals) > 1 else str(year_vals[0])
            results.append(
                {
                    "factor": factor,
                    "range_display": label,
                    "annual_return": metrics["annual_return"],
                    "avg_return": metrics.get("avg_return"),
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "samples": samples,
                    "trade_days": trade_days,
                    "data_years": years or "--",
                    "is_user_range": bool(is_user),
                }
            )

    if not results:
        print("[WARN][FLEX] 未产生任何有效区间（可能样本不足），请检查分档数或下限设置。")
        return {}

    report_opts = report_options.copy() if report_options else {}
    report_opts.setdefault("report_top_n", cfg.get("report_top_n", 20))
    report_opts.setdefault("bin_strategy", bin_strategy)
    report_opts["default_bins"] = default_bins
    report_opts["min_samples"] = min_samples
    report_opts["filter_stats"] = filter_stats
    report_opts["overall_metrics"] = overall_metrics
    duration = time.perf_counter() - start
    paths = generate_single_param_flex_reports(results, report_opts)
    paths["duration_sec"] = duration
    print(f"[INFO][FLEX] Workflow completed in {duration:.2f}s with {len(results)} ranges.")
    return paths
