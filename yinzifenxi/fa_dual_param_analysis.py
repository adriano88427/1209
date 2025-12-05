# -*- coding: utf-8 -*-
"""
带参数双因子分析管道。
"""
from __future__ import annotations

import itertools
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fa_config import RETURN_COLUMN, DATA_PARSE_CONFIG, FACTOR_META
from .fa_logging import detect_debug_enabled
from .fa_stat_utils import calc_max_drawdown
from .fa_dual_param_report import generate_dual_param_reports


def _debug(message: str):
    if detect_debug_enabled():
        print(f"[DEBUG][DUAL-PARAM] {message}")


def run_dual_param_pipeline(
    parameterized_analyzer,
    dual_config: Dict[str, Any],
    report_options: Dict[str, Any],
    logger=None,
) -> Dict[str, str]:
    section_start = time.perf_counter()
    data = getattr(parameterized_analyzer, "processed_data", None)
    if data is None or data.empty:
        print("[WARN] 双因子带参数分析缺少 processed_data，已跳过")
        elapsed = time.perf_counter() - section_start
        print(f"[INFO][DUAL-PARAM] Workflow stopped after {elapsed:.2f}s due to missing data.")
        return {}

    _debug(f"接收到预处理数据: {len(data)} 行 × {len(data.columns)} 列")
    print(f"[DUAL-PARAM] 接收到预处理数据: {len(data)} 行, {len(data.columns)} 列")
    configured_pairs = dual_config.get("param_factor_pairs") or []
    if configured_pairs:
        print(f"[DUAL-PARAM] 用户配置的 param_factor_pairs 共 {len(configured_pairs)} 组: {configured_pairs}")
        _debug(f"手动组合列表: {configured_pairs}")
    else:
        print("[DUAL-PARAM] 未提供 param_factor_pairs, 将退回自动选择")
    param_ranges = dual_config.get("param_ranges") or {}
    if param_ranges:
        print("[DUAL-PARAM] 用户自定义区间如下:")
        for factor_name, ranges in param_ranges.items():
            print(f"  - {factor_name}: {ranges}")
        _debug(f"区间配置: {param_ranges}")
    else:
        print("[DUAL-PARAM] 未提供 param_ranges, 将按默认分箱")
    print(
        f"[DUAL-PARAM] 参数: min_samples={dual_config.get('param_min_samples', 300)}, "
        f"default_bins={dual_config.get('param_default_bins', 3)}"
    )

    factor_pairs = _select_param_pairs(parameterized_analyzer, dual_config)
    if not factor_pairs:
        print("[WARN] 未找到用于带参数双因子的因子对")
        elapsed = time.perf_counter() - section_start
        print(f"[INFO][DUAL-PARAM] Workflow stopped after {elapsed:.2f}s because no factor pairs qualified.")
        return {}

    _debug(f"最终需要分析的因子对数量: {len(factor_pairs)}")
    print(f"[DUAL-PARAM] 实际需要分析的因子组合共 {len(factor_pairs)} 组: {factor_pairs}")

    min_samples = dual_config.get("param_min_samples", 300)
    results: List[Dict[str, Any]] = []
    default_bins = dual_config.get("param_default_bins", 3)
    manual_pairs = {
        (str(pair[0]), str(pair[1]))
        for pair in (dual_config.get("param_factor_pairs") or [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }
    for pair in factor_pairs:
        pair_key = (str(pair[0]), str(pair[1]))
        pair_result = _analyze_parameterized_pair(
            data,
            pair,
            dual_config.get("param_ranges", {}),
            min_samples,
            default_bins,
            force_inclusion=pair_key in manual_pairs,
        )
        if pair_result:
            results.append(pair_result)
        else:
            _debug(f"{pair} 未通过筛选，已跳过")

    if not results:
        print("[WARN] 双因子带参数分析未产生有效结果")
        elapsed = time.perf_counter() - section_start
        print(f"[INFO][DUAL-PARAM] Workflow stopped after {elapsed:.2f}s without any valid result.")
        return {}

    report_paths = generate_dual_param_reports(results, report_options)
    report_paths["result_count"] = len(results)
    duration = time.perf_counter() - section_start
    report_paths["duration_sec"] = duration
    print(f"[INFO][DUAL-PARAM] Workflow completed in {duration:.2f}s with {len(results)} results.")
    return report_paths


def _select_param_pairs(parameterized_analyzer, dual_config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """返回需要分析的因子组合：优先包含手动指定组合，再补充自动生成组合。"""
    max_pairs = max(1, int(dual_config.get("max_factor_pairs", 20)))
    top_n = max(2, int(dual_config.get("nonparam_top_n", 6)))

    manual_pairs: Sequence[Tuple[str, str]] = dual_config.get("param_factor_pairs") or []
    filtered_manual: List[Tuple[str, str]] = []
    if manual_pairs:
        for a, b in manual_pairs:
            if a in parameterized_analyzer.factors and b in parameterized_analyzer.factors:
                filtered_manual.append((a, b))
        removed = len(manual_pairs) - len(filtered_manual)
        if removed > 0:
            print(f"[DUAL-PARAM] 有 {removed} 个 param_factor_pairs 不在可用因子列表中, 已忽略")
        if filtered_manual:
            print(f"[DUAL-PARAM] 采用用户指定的 {len(filtered_manual)} 组组合: {filtered_manual}")
            _debug(f"手动组合入选: {filtered_manual}")
        else:
            print("[DUAL-PARAM] 用户指定组合均未命中有效因子, 将回退为纯自动组合")

    factors = parameterized_analyzer.factors[:top_n]
    auto_candidates = list(itertools.combinations(factors, 2))
    _debug(f"自动候选组合数量: {len(auto_candidates)} (top_n={len(factors)})")
    print(f"[DUAL-PARAM] 自动候选组合数量: {len(auto_candidates)} (基于前 {len(factors)} 个因子)")

    final_pairs: List[Tuple[str, str]] = []
    seen = set()

    def _append_pair(pair: Tuple[str, str]):
        key = tuple(pair)
        if key in seen:
            return False
        final_pairs.append(key)
        seen.add(key)
        return True

    for pair in filtered_manual:
        _append_pair(pair)

    remaining_slots = max_pairs - len(final_pairs)
    auto_added = 0
    if remaining_slots <= 0:
        print(f"[DUAL-PARAM] 手动组合已达到上限 {max_pairs}，自动组合不再补充")
    else:
        for pair in auto_candidates:
            if _append_pair(pair):
                auto_added += 1
                if auto_added >= remaining_slots:
                    break

    if not final_pairs:
        print("[DUAL-PARAM] 未能生成任何可分析的因子组合")
    else:
        print(
            f"[DUAL-PARAM] 实际需要分析的因子组合共 {len(final_pairs)} 组 "
            f"(手动 {len(filtered_manual)} 组, 自动 {auto_added} 组, 上限 {max_pairs})"
        )
        _debug(f"组合明细: {final_pairs}")
    return final_pairs


def _analyze_parameterized_pair(
    df: pd.DataFrame,
    pair: Tuple[str, str],
    range_config: Dict[str, Any],
    min_samples: int,
    default_bins: int,
    force_inclusion: bool = False,
) -> Optional[Dict[str, Any]]:
    factor_a, factor_b = pair
    _debug(
        f"准备分析 {pair}: min_samples={min_samples}, force_inclusion={force_inclusion}, "
        f"default_bins={default_bins}"
    )
    if factor_a not in df.columns or factor_b not in df.columns or RETURN_COLUMN not in df.columns:
        print(f"[DUAL-PARAM] 组合 {pair} 缺少必要列，已跳过")
        _debug(f"{pair} 缺少必要列，跳过")
        return None

    ranges_a = _resolve_ranges(df[factor_a], range_config.get(factor_a), default_bins)
    ranges_b = _resolve_ranges(df[factor_b], range_config.get(factor_b), default_bins)
    if not ranges_a or not ranges_b:
        print(
            f"[DUAL-PARAM] 组合 {pair} 无可用区间 "
            f"(ranges_a={len(ranges_a) if ranges_a else 0}, ranges_b={len(ranges_b) if ranges_b else 0})，跳过"
        )
        _debug(f"{pair} 无可用区间 (ranges_a={len(ranges_a) if ranges_a else 0}, ranges_b={len(ranges_b) if ranges_b else 0})")
        return None

    print(
        f"[DUAL-PARAM] 开始分析组合 {pair}，区间数量: A={len(ranges_a)}, B={len(ranges_b)}, "
        f"force_inclusion={force_inclusion}, min_samples={min_samples}"
    )

    records: List[Dict[str, Any]] = []
    total_cells = 0
    skipped_no_sample = 0
    skipped_min_sample = 0
    for idx_a, range_a in enumerate(ranges_a):
        mask_a = _build_mask(df[factor_a], range_a)
        for idx_b, range_b in enumerate(ranges_b):
            total_cells += 1
            mask_b = _build_mask(df[factor_b], range_b)
            subset = df[mask_a & mask_b]
            returns = subset[RETURN_COLUMN].dropna()
            if len(returns) == 0:
                skipped_no_sample += 1
                continue
            if len(returns) < min_samples and not force_inclusion:
                skipped_min_sample += 1
                continue
            avg = float(returns.mean())
            std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
            annual_return = avg * 252
            annual_std = std * np.sqrt(252) if std > 0 else np.nan
            sharpe = (annual_return / annual_std) if annual_std and np.isfinite(annual_std) else np.nan
            drawdown = calc_max_drawdown(returns)
            win_rate = float((returns > 0).mean())
            years_value = _extract_years(subset)

            range_a_label = _format_range_label(idx_a)
            range_b_label = _format_range_label(idx_b)
            range_a_display = _format_range_values(factor_a, range_a)
            range_b_display = _format_range_values(factor_b, range_b)

            records.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "range_a": range_a,
                    "range_b": range_b,
                    "range_a_label": range_a_label,
                    "range_b_label": range_b_label,
                    "range_a_display": range_a_display,
                    "range_b_display": range_b_display,
                    "avg_return": avg,
                    "annual_return": annual_return,
                    "annual_std": annual_std,
                    "sharpe": sharpe,
                    "samples": len(returns),
                    "win_rate": win_rate,
                    "max_drawdown": drawdown,
                    "data_years": years_value,
                }
            )

    if not records:
        print(
            f"[DUAL-PARAM] 组合 {pair} 所有 {total_cells} 个交叉区间均未满足条件 "
            f"(空样本 {skipped_no_sample}, 样本不足 {skipped_min_sample})"
        )
        _debug(
            f"{pair} 全部交叉区间失效 total={total_cells}, 空样本={skipped_no_sample}, "
            f"样本不足={skipped_min_sample}"
        )
        return None

    print(
        f"[DUAL-PARAM] 组合 {pair} 共有 {len(records)} 个区间满足条件 "
        f"(总交叉 {total_cells}, 空样本 {skipped_no_sample}, 样本不足 {skipped_min_sample})"
    )
    _debug(
        f"{pair} 满足条件区间={len(records)}/{total_cells}, 空样本={skipped_no_sample}, "
        f"样本不足={skipped_min_sample}"
    )

    pair_df = pd.DataFrame(records)
    best = pair_df.sort_values("annual_return", ascending=False).iloc[0]
    worst = pair_df.sort_values("annual_return", ascending=True).iloc[0]

    single_a_return = _calc_single_factor_return(df, factor_a, best["range_a"])
    single_b_return = _calc_single_factor_return(df, factor_b, best["range_b"])
    comparison = [value for value in (single_a_return, single_b_return) if value is not None and np.isfinite(value)]
    baseline_value = max(comparison) if comparison else None
    baseline_safe = baseline_value if baseline_value is not None else 0.0
    synergy = (
        best["annual_return"] - baseline_safe
        if np.isfinite(best["annual_return"]) and np.isfinite(baseline_safe)
        else np.nan
    )

    summary = {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "range_a_display": best["range_a_display"],
        "range_b_display": best["range_b_display"],
        "best_range": f"{factor_a}{best['range_a_display']} & {factor_b}{best['range_b_display']}",
        "best_annual_return": best["annual_return"],
        "best_annual_std": best["annual_std"],
        "best_sharpe": best["sharpe"],
        "best_max_drawdown": best["max_drawdown"],
        "best_avg_return": best["avg_return"],
        "samples": best.get("samples"),
        "data_years": best.get("data_years"),
        "single_a_annual_return": single_a_return,
        "single_b_annual_return": single_b_return,
        "worst_range": f"{worst['range_a_label']} & {worst['range_b_label']}",
        "worst_annual_return": worst["annual_return"],
        "synergy": synergy,
    }
    print(
        f"[DUAL-PARAM] 组合 {pair} 最佳区间 {summary['best_range']} "
        f"年化收益 {summary['best_annual_return']:.4f}, 样本 {summary['samples']}"
    )
    baseline_str = (
        f"{baseline_value:.4f}" if baseline_value is not None and np.isfinite(baseline_value) else "N/A"
    )
    synergy_str = f"{synergy:.4f}" if np.isfinite(summary["synergy"]) else "N/A"
    _debug(
        f"{pair} 最佳区间={summary['best_range']} 样本={summary['samples']} "
        f"baseline单因子={baseline_str} synergy={synergy_str}"
    )
    return {"summary": summary, "grid": pair_df}


def _resolve_ranges(series: pd.Series, config_ranges, default_bins: int) -> List[Tuple[float, float]]:
    """将用户配置或默认分位转换为数值区间。"""
    if config_ranges:
        normalized = []
        for item in config_ranges:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                normalized.append((float(item[0]), float(item[1])))
        if normalized:
            return normalized

    bins = max(2, int(default_bins))
    quantile_points = np.linspace(0, 1, bins + 1)
    values = series.quantile(quantile_points).tolist()
    ranges = []
    for idx in range(bins):
        low, high = values[idx], values[idx + 1]
        if low == high:
            continue
        ranges.append((float(low), float(high)))
    return ranges


def _build_mask(series: pd.Series, value_range: Tuple[float, float]) -> pd.Series:
    low, high = value_range
    return (series >= low) & (series <= high)


def _format_range_label(index: int) -> str:
    return f"区间{index + 1}"


def _calc_single_factor_return(df: pd.DataFrame, factor: str, value_range: Tuple[float, float]) -> Optional[float]:
    if factor not in df.columns or value_range is None:
        return None
    mask = _build_mask(df[factor], value_range)
    returns = df.loc[mask, RETURN_COLUMN].dropna()
    if returns.empty:
        return None
    return float(returns.mean()) * 252


def _extract_years(subset: pd.DataFrame) -> Optional[str]:
    if subset.empty:
        return None
    if "数据年份" in subset.columns:
        years = subset["数据年份"].dropna().unique()
    elif "信号日期" in subset.columns:
        dates = pd.to_datetime(subset["信号日期"], errors="coerce")
        years = dates.dt.year.dropna().unique()
    else:
        return None
    years = [int(y) for y in years if pd.notna(y)]
    if not years:
        return None
    return ",".join(str(year) for year in sorted(years))


def _infer_factor_semantic(factor: str) -> Optional[str]:
    meta = FACTOR_META.get(factor)
    if meta and meta.get("semantic"):
        return meta["semantic"]
    column_types = (DATA_PARSE_CONFIG.get("column_types") or {}) if isinstance(DATA_PARSE_CONFIG, dict) else {}
    semantic = column_types.get(factor)
    return semantic


def _format_range_values(factor: str, value_range: Tuple[float, float]) -> str:
    low, high = value_range
    semantic = _infer_factor_semantic(factor)
    if semantic == "percent":
        return f"[{low * 100:.2f}%~{high * 100:.2f}%]"
    if semantic == "amount":
        return f"[{low / 1e8:.2f}亿~{high / 1e8:.2f}亿]"
    return f"[{low:.4f}~{high:.4f}]"
