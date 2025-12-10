# -*- coding: utf-8 -*-
"""
带参数单因子自由区间报告
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .fa_report_utils import (
    HTMLReportBuilder,
    render_metric_cards,
    render_table,
    render_alert,
    render_text_block,
    render_report_notes,
)


def _fmt_percent(value, decimals=2):
    if pd.isna(value):
        return "--"
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "--"


def _fmt_float(value, decimals=3):
    if pd.isna(value):
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "--"


def _calc_bounds(series: pd.Series, higher_better=True):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return (1.0, 0.0) if higher_better else (0.0, 1.0)
    q_low = clean.quantile(0.05)
    q_high = clean.quantile(0.95)
    if higher_better:
        best, worst = q_high, q_low
    else:
        best, worst = q_low, q_high
    if best == worst:
        span = abs(best) * 0.1 + 1e-6
        if higher_better:
            best += span
            worst -= span
        else:
            best -= span
            worst += span
    return best, worst


def _score_linear(value, best, worst, higher_better=True):
    if pd.isna(value):
        return 0.0
    val = float(value)
    if higher_better:
        if val <= worst:
            return 0.0
        if val >= best:
            return 100.0
        return (val - worst) / (best - worst) * 100.0
    else:
        if val >= worst:
            return 0.0
        if val <= best:
            return 100.0
        return (worst - val) / (worst - best) * 100.0


def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ret_best, ret_worst = _calc_bounds(df["annual_return"], True)
    sh_best, sh_worst = _calc_bounds(df["sharpe"], True)
    dd_best, dd_worst = _calc_bounds(df["max_drawdown"], False)
    sp_best, sp_worst = _calc_bounds(df["samples"], True)

    df = df.copy()
    df["score_return"] = df["annual_return"].apply(lambda v: _score_linear(v, ret_best, ret_worst, True))
    df["score_sharpe"] = df["sharpe"].apply(lambda v: _score_linear(v, sh_best, sh_worst, True))
    df["score_drawdown"] = df["max_drawdown"].apply(lambda v: _score_linear(v, dd_best, dd_worst, False))
    df["score_samples"] = df["samples"].apply(lambda v: _score_linear(v, sp_best, sp_worst, True))
    df["composite_score"] = (
        df["score_return"] * 0.25
        + df["score_sharpe"] * 0.25
        + df["score_drawdown"] * 0.15
        + df["score_samples"] * 0.35
    )
    return df


def _prepare_views(scored: pd.DataFrame, top_n: int):
    scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
    top_limit = max(1, int(top_n))
    full_view = scored.head(top_limit * 2).copy()

    # 去重视图：每因子仅保留综合得分最高的区间
    dedup = scored.sort_values("composite_score", ascending=False).groupby("factor", as_index=False).first()
    dedup = dedup.sort_values("composite_score", ascending=False).head(top_limit * 2)
    return full_view, dedup


def generate_single_param_flex_reports(results: List[Dict[str, Any]], report_options: Dict[str, Any]) -> Dict[str, str]:
    df = pd.DataFrame(results)
    scored = _score_dataframe(df)
    top_n = int(report_options.get("report_top_n", 20) or 20)
    full_view, dedup_view = _prepare_views(scored, top_n)

    output_dir = report_options.get("output_dir") or os.path.join(os.getcwd(), "baogao")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 按需求重命名：带参数的单因子自由区间挖掘 + 时间戳（去掉“分析”字样）
    prefix = "带参数的单因子自由区间挖掘"

    # CSV / Excel
    csv_path = os.path.join(output_dir, f"{prefix}_汇总_{timestamp}.csv")
    scored.to_csv(csv_path, index=False, encoding="utf-8-sig")
    excel_path = os.path.join(output_dir, f"{prefix}_数据_{timestamp}.xlsx")
    scored.to_excel(excel_path, index=False)

    html_path = os.path.join(output_dir, f"{prefix}_{timestamp}.html")
    html_content = _build_html(scored, full_view, dedup_view, report_options, timestamp)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK][FLEX] CSV: {csv_path}")
    print(f"[OK][FLEX] Excel: {excel_path}")
    print(f"[OK][FLEX] HTML: {html_path}")
    return {"csv_path": csv_path, "excel_path": excel_path, "html_path": html_path}


def _build_html(scored, full_view, dedup_view, report_options, timestamp):
    builder = HTMLReportBuilder("带参数单因子自由区间挖掘报告", f"生成时间 {timestamp}")
    top_n = int(report_options.get("report_top_n", 20) or 20)
    min_samples = report_options.get("min_samples")
    filter_stats = report_options.get("filter_stats") or {}
    overall = report_options.get("overall_metrics") or {}
    overall_return = overall.get("annual_return") if isinstance(overall, dict) else None
    threshold = None
    if overall_return is not None and not pd.isna(overall_return):
        threshold = overall_return * 1.2
    bin_strategy = report_options.get("bin_strategy", "quantile")

    cards = render_metric_cards(
        [
            ("有效区间数", str(len(scored))),
            (
                "整体年化收益率(全样本)",
                _fmt_percent(overall.get("annual_return"), 2)
                if overall
                else _fmt_percent(scored["annual_return"].mean(), 2),
            ),
            (
                "整体最大回撤(全样本)",
                _fmt_percent(overall.get("max_drawdown"), 1)
                if overall
                else _fmt_percent(scored["max_drawdown"].mean(), 1),
            ),
            ("样本中位数", _fmt_float(scored["samples"].median(), 1)),
        ]
    )
    builder.add_section("核心指标", cards)

    desc_lines = [
        f"- 分档策略：{bin_strategy}，默认分档 {report_options.get('default_bins','?')}；样本下限 {min_samples}",
        f"- 过滤统计：总区间 {filter_stats.get('total_ranges',0)}，样本不足 {filter_stats.get('low_sample',0)}",
    ]
    builder.add_section("运行说明", render_text_block("\n".join(desc_lines)))

    if scored.empty:
        builder.add_section("排行榜", render_alert("暂无满足条件的区间"))
        return builder.build()

    def _apply_threshold(df: pd.DataFrame) -> pd.DataFrame:
        if threshold is None:
            return df
        return df[df["annual_return"] > threshold]

    def _render_table(df):
        base = _apply_threshold(df)
        user_rows = df[df.get("is_user_range", False)]
        # 合并后去重，保证用户区间始终展示
        merged = pd.concat([base, user_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=["factor", "range_display"])
        display = merged.head(top_n * 2).copy()
        if display.empty:
            return render_alert("暂无满足条件的区间")
        cols = [
            "factor",
            "range_display",
            "composite_score",
            "annual_return",
            "avg_return",
            "max_drawdown",
            "samples",
            "trade_days",
            "data_years",
            "is_user_range",
        ]
        display = display[cols]
        display = display.rename(
            columns={
                "factor": "因子",
                "range_display": "区间",
                "composite_score": "综合得分",
                "annual_return": "年化收益率",
                "avg_return": "平均每笔交易收益率",
                "max_drawdown": "最大回撤",
                "samples": "样本数量",
                "trade_days": "交易日数",
                "data_years": "数据年份",
            }
        )
        # 行样式：用户区间标黄
        row_classes = {}
        is_user_flags = display["is_user_range"].copy()
        display = display.drop(columns=["is_user_range"])
        display = display[
            [
                "因子",
                "区间",
                "综合得分",
                "年化收益率",
                "平均每笔交易收益率",
                "最大回撤",
                "样本数量",
                "交易日数",
                "数据年份",
            ]
        ]
        for idx, is_user in is_user_flags.items():
            if bool(is_user):
                row_classes[idx] = "highlight-yellow"
        # 标记各数值列的前三名为蓝色背景
        metric_columns = {
            "综合得分": False,    # 越大越好
            "年化收益率": False,  # 越大越好
            "平均每笔交易收益率": False,  # 越大越好
            "最大回撤": True,     # 越小越好
        }
        cell_classes = {}
        for col, ascending in metric_columns.items():
            if col not in display.columns:
                continue
            top_idx = (
                display[col]
                .dropna()
                .sort_values(ascending=ascending)
                .index[:3]
            )
            for idx in top_idx:
                cell_classes.setdefault(idx, {})[col] = "highlight-blue"
        formatters = {
            "综合得分": _fmt_float,
            "年化收益率": lambda v: _fmt_percent(v, 2),
            "平均每笔交易收益率": lambda v: _fmt_percent(v, 2),
            "最大回撤": lambda v: _fmt_percent(v, 1),
            "样本数量": lambda v: str(int(v)) if pd.notna(v) else "--",
            "交易日数": lambda v: str(int(v)) if pd.notna(v) else "--",
        }
        return render_table(
            display,
            list(display.columns),
            formatters=formatters,
            cell_classes=cell_classes,
            row_classes=row_classes,
        )

    builder.add_section("排行榜（全量区间）", _render_table(full_view))
    builder.add_section("排行榜（按因子去重）", _render_table(dedup_view))

    user_view = scored[scored["is_user_range"]]
    if user_view.empty:
        builder.add_section("用户区间", render_alert("暂无用户指定区间或无区间满足样本下限。", level="info"))
    else:
        builder.add_section("用户区间", _render_table(user_view))

    builder.add_section("报告说明", render_report_notes())

    return builder.render()
