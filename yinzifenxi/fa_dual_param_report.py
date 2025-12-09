# -*- coding: utf-8 -*-
"""
带参数双因子报告生成模块
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .fa_report_utils import (
    HTMLReportBuilder,
    build_run_metadata_section,
    render_metric_cards,
    render_table,
    render_alert,
)


def _fmt_percent(value, decimals=2):
    if pd.isna(value):
        return "--"
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "--"


def _format_detail_grid(grid: pd.DataFrame) -> pd.DataFrame:
    percent_columns = {
        "avg_return",
        "annual_return",
        "annual_std",
        "win_rate",
        "max_drawdown",
    }
    formatted = grid.copy()
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(lambda v: _fmt_percent(v, 2))
    return formatted


def generate_dual_param_reports(results: List[Dict], report_options: Dict[str, any]) -> Dict[str, str]:
    output_dir = report_options.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    prefix = report_options.get("param_prefix", "双因子带参数")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries = [item["summary"] for item in results]
    summary_df = pd.DataFrame(summaries)
    csv_name = f"{prefix}汇总_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    html_name = f"{prefix}分析_{timestamp}.html"
    html_path = os.path.join(output_dir, html_name)
    html_content = _build_html(summary_df, results, timestamp, report_options)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    excel_name = f"{prefix}数据_{timestamp}.xlsx"
    excel_path = os.path.join(output_dir, excel_name)
    _write_excel(results, excel_path)

    return {"csv_path": csv_path, "html_path": html_path, "excel_path": excel_path}


def _build_html(
    summary_df: pd.DataFrame,
    results: List[Dict],
    timestamp: str,
    report_options: Dict[str, any],
) -> str:
    builder = HTMLReportBuilder("带参数双因子分析报告", f"生成时间 {timestamp}")
    result_count = len(results)
    meta_items, run_info_html = build_run_metadata_section(report_options, result_count)
    builder.set_meta(meta_items)
    builder.add_section("运行与调试信息", run_info_html)
    cards = render_metric_cards(
        [
            ("因子对数", str(len(results))),
            ("平均最佳年化收益率", _fmt_percent(summary_df["best_annual_return"].mean(), 2)),
            ("平均协同增益", _fmt_percent(summary_df["synergy"].mean(), 2)),
        ]
    )
    builder.add_section("核心指标", cards)

    rank_limit = int(report_options.get("max_rank_display", 10) or 10)
    ranked = summary_df.copy().reset_index(drop=True)
    if ranked.empty:
        builder.add_section("最佳双因子区间", render_alert("暂无有效区间"))
    else:
        manual_pairs = report_options.get("param_factor_pairs") or []

        def _pair_key(a, b):
            return (str(a), str(b))

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

        def _calc_bounds(series, higher_better=True):
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

        ret_best, ret_worst = _calc_bounds(ranked["best_annual_return"], True)
        sharpe_best, sharpe_worst = _calc_bounds(ranked["best_sharpe"], True)
        std_best, std_worst = _calc_bounds(ranked["best_annual_std"], False)
        draw_best, draw_worst = _calc_bounds(ranked["best_max_drawdown"], False)
        avg_best, avg_worst = _calc_bounds(ranked["best_avg_return"], True)
        sample_best, sample_worst = _calc_bounds(ranked["samples"], True)

        ranked = ranked.copy()
        ranked["score_return"] = ranked["best_annual_return"].apply(
            lambda v: _score_linear(v, ret_best, ret_worst, True)
        )
        ranked["score_sharpe"] = ranked["best_sharpe"].apply(
            lambda v: _score_linear(v, sharpe_best, sharpe_worst, True)
        )
        ranked["score_std"] = ranked["best_annual_std"].apply(
            lambda v: _score_linear(v, std_best, std_worst, False)
        )
        ranked["score_drawdown"] = ranked["best_max_drawdown"].apply(
            lambda v: _score_linear(v, draw_best, draw_worst, False)
        )
        ranked["score_avg"] = ranked["best_avg_return"].apply(
            lambda v: _score_linear(v, avg_best, avg_worst, True)
        )
        ranked["score_samples"] = ranked["samples"].apply(
            lambda v: _score_linear(v, sample_best, sample_worst, True)
        )
        ranked["composite_score"] = (
            ranked["score_return"] * 0.20
            + ranked["score_sharpe"] * 0.30
            + ranked["score_std"] * 0.10
            + ranked["score_drawdown"] * 0.20
            + ranked["score_avg"] * 0.10
            + ranked["score_samples"] * 0.10
        )
        ranked = ranked.sort_values("composite_score", ascending=False).reset_index(drop=True)

        display_df = ranked.head(rank_limit).copy()
        existing = {
            _pair_key(row["factor_a"], row["factor_b"])
            for _, row in display_df.iterrows()
        }
        for pair in manual_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            key = _pair_key(pair[0], pair[1])
            if key in existing:
                continue
            match = ranked[
                (ranked["factor_a"] == pair[0]) & (ranked["factor_b"] == pair[1])
            ]
            if not match.empty:
                display_df = pd.concat([display_df, match.iloc[[0]]], ignore_index=True)
                existing.add(key)

        display_df = display_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        manual_pair_keys = {
            _pair_key(pair[0], pair[1])
            for pair in manual_pairs
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        row_class_map = {}
        for idx, row in display_df.iterrows():
            if _pair_key(row["factor_a"], row["factor_b"]) in manual_pair_keys:
                row_class_map[idx] = "highlight-manual"

        def _format_years_multiline(value) -> str:
            if pd.isna(value):
                return "--"
            tokens = [token.strip() for token in str(value).split(",") if token.strip()]
            if not tokens:
                return "--"
            chunk = max(1, math.ceil(len(tokens) / 3))
            lines = ["、".join(tokens[i:i + chunk]) for i in range(0, len(tokens), chunk)]
            lines = lines[:3]
            return f"<div class='years-multiline'>{'<br>'.join(lines)}</div>"

        metric_configs = [
            ("best_annual_return", True),
            ("best_annual_std", False),
            ("best_sharpe", True),
            ("best_max_drawdown", False),
            ("best_avg_return", True),
            ("samples", True),
            ("synergy", True),
        ]
        cell_classes = defaultdict(dict)
        for column, higher_better in metric_configs:
            numeric = pd.to_numeric(display_df[column], errors="coerce")
            numeric = numeric.dropna()
            if numeric.empty:
                continue
            top_indices = numeric.sort_values(ascending=not higher_better).index[:3]
            for idx in top_indices:
                cell_classes[idx][column] = "highlight-blue"

        table = render_table(
            display_df,
            columns=[
                "factor_a",
                "factor_b",
                "best_range",
                "best_annual_return",
                "best_annual_std",
                "best_sharpe",
                "best_max_drawdown",
                "best_avg_return",
                "single_a_annual_return",
                "single_b_annual_return",
                "samples",
                "data_years",
                "composite_score",
                "synergy",
            ],
            headers=[
                "因子A",
                "因子B",
                "最佳区间",
                "最佳年化收益率",
                "年化波动",
                "夏普比率",
                "最大回撤",
                "平均每笔收益率",
                "因子A区间年化收益率",
                "因子B区间年化收益率",
                "样本数量",
                "数据年份",
                "综合得分",
                "协同增益",
            ],
            formatters={
                "best_annual_return": lambda v: _fmt_percent(v, 2),
                "best_annual_std": lambda v: _fmt_percent(v, 2),
                "best_sharpe": lambda v: "--" if pd.isna(v) else f"{float(v):.2f}",
                "best_max_drawdown": lambda v: _fmt_percent(v, 2),
                "best_avg_return": lambda v: _fmt_percent(v, 2),
                "single_a_annual_return": lambda v: _fmt_percent(v, 2),
                "single_b_annual_return": lambda v: _fmt_percent(v, 2),
                "samples": lambda v: "--" if pd.isna(v) else f"{int(v)}",
                "data_years": _format_years_multiline,
                "composite_score": lambda v: "--" if pd.isna(v) else f"{float(v):.1f}",
                "synergy": lambda v: _fmt_percent(v, 2),
            },
            cell_classes=dict(cell_classes),
            row_classes=row_class_map,
            column_classes={"data_years": "years-cell"},
            html_columns=["data_years"],
        )
        builder.add_section("最佳双因子区间", table)

    detail_sections = []
    for item in results[:3]:
        grid = _format_detail_grid(item["grid"])
        summary = item["summary"]
        detail_sections.append(
            f"<h3>{summary['factor_a']} × {summary['factor_b']}</h3>" + grid.to_html(index=False)
        )
    if detail_sections:
        builder.add_section("部分区间详情", "".join(detail_sections))

    return builder.render()


def _write_excel(results: List[Dict], excel_path: str):
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    for item in results:
        sheet_name = f"{item['summary']['factor_a']}_{item['summary']['factor_b']}"
        clean_name = sheet_name[:30]
        item["grid"].to_excel(writer, sheet_name=clean_name, index=False)
    writer.close()
