# -*- coding: utf-8 -*-
"""
HTML 鎶ュ憡杈呭姪缁勪欢銆?
"""

from __future__ import annotations

import math
from html import escape
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


HTML_BASE_STYLE = """
:root {
    --bg: #eef2fb;
    --card-bg: #ffffff;
    --primary: #295cdb;
    --secondary: #8b5cf6;
    --text: #1e2430;
    --muted: #7d889f;
    --border: #e1e6f5;
    --positive: #1abc9c;
    --negative: #e74c3c;
    --info: #3498db;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 0;
    font-family: "HarmonyOS Sans","Microsoft YaHei","Segoe UI",sans-serif;
    background: linear-gradient(135deg, #f4f8ff 0%, #e9efff 45%, #f9f4ff 100%);
    color: var(--text);
}
.container {
    max-width: 1150px;
    margin: 0 auto;
    padding: 36px 22px 60px;
}
.hero {
    background: linear-gradient(120deg, #1f4ad8 0%, #7b5dff 50%, #c15dff 100%);
    color: #fff;
    border-radius: 20px;
    padding: 32px 28px 36px;
    box-shadow: 0 25px 70px rgba(37,75,208,0.35);
    margin-bottom: 28px;
}
.hero h1 {
    margin: 0;
    font-size: 32px;
    letter-spacing: 0.5px;
}
.hero p {
    margin: 10px 0 0;
    font-size: 16px;
    opacity: 0.9;
}
.meta-info {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 18px;
}
.meta-item {
    background: rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 8px 16px;
    font-size: 13px;
    border: 1px solid rgba(255,255,255,0.25);
}
.meta-item strong {
    margin-left: 6px;
    color: #fff;
}
.section {
    background: var(--card-bg);
    border-radius: 18px;
    padding: 26px 26px 24px;
    margin-bottom: 32px;
    box-shadow: 0 15px 45px rgba(32,65,153,0.12);
    border: 1px solid rgba(255,255,255,0.6);
}
.section h2 {
    margin-top: 0;
    font-size: 22px;
    color: var(--primary);
    position: relative;
    padding-left: 12px;
}
.section h2::before {
    content: "";
    position: absolute;
    width: 4px;
    height: 24px;
    left: 0;
    top: 4px;
    border-radius: 4px;
    background: var(--secondary);
}
.metric-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 18px;
    margin: 12px 0 4px;
}
.metric-card {
    padding: 18px;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(41,92,219,0.08), rgba(255,255,255,0.8));
}
.metric-card .label {
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 26px;
    font-weight: 600;
    margin-top: 8px;
    color: var(--text);
}
.alert {
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 14px;
    margin: 12px 0;
}
.alert-info { background: rgba(41,92,219,0.1); color: var(--primary); }
.alert-warn { background: rgba(231,76,60,0.09); color: #d54a34; }
.table-wrapper {
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
}
  .table {
      width: 100%;
      min-width: unset;
      border-collapse: collapse;
      font-size: 13px;
      table-layout: fixed;
  }
  .table-compact {
      font-size: 12px;
  }
.table thead {
    background: linear-gradient(120deg, #244bdc, #5c7dff);
    color: #fff;
}
  .table th {
      padding: 8px 10px;
      border: 1px solid var(--border);
      text-align: left;
      white-space: normal;
      word-break: break-word;
  }
  .table-compact th {
      padding: 6px 8px;
  }
  .table td {
      padding: 8px 10px;
      border: 1px solid var(--border);
      text-align: left;
      word-break: break-word;
  }
  .table-compact td {
      padding: 6px 8px;
  }
.table tbody tr:nth-child(even) {
    background: #f6f7fd;
}
.table tbody tr:hover {
    background: rgba(41,92,219,0.08);
}
.highlight-alert {
    background: #ffd6d6 !important;
}
.highlight-blue {
    background: #d7efff !important;
}
.highlight-yellow {
    background: #fff4c2 !important;
}
.highlight-manual {
    background: #fff4c2 !important;
}
.score-emphasis {
    color: #0b5ed7;
    font-weight: 600;
}
.factor-emphasis {
    color: #0b5ed7;
}
.top-list {
    list-style: none;
    padding-left: 0;
}
.top-list li {
    padding: 12px 0;
    border-bottom: 1px dashed var(--border);
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    color: #fff;
    margin-left: 8px;
}
.badge-positive { background: linear-gradient(120deg,#12c2a5,#0ba982); }
.badge-negative { background: linear-gradient(120deg,#ff6b6b,#f06543); }
.factor-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 20px;
    background: #fbfcff;
}
.factor-card h3 { margin-top: 0; }
.factor-card ul {
    padding-left: 18px;
    margin: 8px 0;
}
.text-block {
    font-size: 14px;
    line-height: 1.85;
    color: var(--text);
    white-space: pre-line;
}
.sub-block {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
}
.sub-block h4 {
    margin: 0 0 6px 0;
    color: var(--muted);
    font-size: 13px;
}
.notes-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 6px;
}
.notes-table th, .notes-table td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
    word-break: break-word;
}
.notes-table thead {
    background: #f0f4ff;
    color: #1e2430;
    font-weight: 600;
}
.notes-list {
    margin: 6px 0 0;
    padding-left: 18px;
    font-size: 13px;
    line-height: 1.6;
}
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-right: 6px;
    background: rgba(0,0,0,0.05);
}
.tag-positive { background: rgba(26,188,156,0.18); color: #0d8b72; }
.tag-negative { background: rgba(231,76,60,0.15); color: #c23c2b; }
.grid-two {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px;
}
.sub-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(244,247,255,0.8));
}
.sub-card h3 {
    margin-top: 0;
}
.muted { color: var(--muted); }
.factor-section .section-desc {
    color: var(--muted);
    margin-bottom: 12px;
}
.highlight-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    background: linear-gradient(135deg, rgba(41,92,219,0.07), rgba(245,247,255,0.95));
}
.highlight-card h4 {
    margin: 0 0 8px;
}
.chip {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 12px;
    background: rgba(0,0,0,0.06);
    margin-right: 6px;
}
.details-block details {
    background: rgba(41,92,219,0.05);
    border-radius: 12px;
    padding: 10px 16px;
}
.details-block summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--primary);
}
.details-block summary::-webkit-details-marker { display: none; }
.details-block summary::after {
    content: "鈻?;
    font-size: 12px;
    margin-left: 6px;
    color: var(--muted);
}
.details-block details[open] summary::after {
    content: "鈻?;
}
.details-block .text-block {
    margin-top: 10px;
}
.years-cell {
    width: 110px;
    white-space: normal;
}
.years-multiline {
    line-height: 1.4;
    white-space: normal;
}
"""


class HTMLReportBuilder:
    """Helper to assemble structured HTML reports."""

    def __init__(self, title: str, subtitle: Optional[str] = None):
        self.title = title
        self.subtitle = subtitle
        self.meta_items: List[Tuple[str, str]] = []
        self.sections: List[Tuple[str, str]] = []

    def set_meta(self, items: Sequence[Tuple[str, str]]) -> None:
        self.meta_items = list(items)

    def add_section(self, title: str, body_html: str) -> None:
        self.sections.append((title, body_html))

    def render(self) -> str:
        meta_html = ""
        if self.meta_items:
            entries = "".join(
                f"<div class='meta-item'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
                for label, value in self.meta_items
            )
            meta_html = f"<div class='meta-info'>{entries}</div>"

        sections_html = ""
        for title, body in self.sections:
            sections_html += (
                f"<section class='section'><h2>{escape(title)}</h2>"
                f"<div class='section-body'>{body}</div></section>"
            )

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <title>{escape(self.title)}</title>
    <style>{HTML_BASE_STYLE}</style>
</head>
<body>
    <div class="container">
        <header class="hero">
            <h1>{escape(self.title)}</h1>
            {'<p>' + escape(self.subtitle) + '</p>' if self.subtitle else ''}
            {meta_html}
        </header>
        {sections_html}
    </div>
</body>
</html>"""


def render_metric_cards(items: Sequence[Tuple[str, str]]) -> str:
    if not items:
        return ""
    cards = "".join(
        f"<div class='metric-card'><div class='label'>{escape(label)}</div>"
        f"<div class='value'>{escape(value)}</div></div>"
        for label, value in items
    )
    return f"<div class='metric-cards'>{cards}</div>"


def render_alert(text: str, level: str = "info") -> str:
    level_class = "alert-info" if level != "warn" else "alert-warn"
    return f"<div class='alert {level_class}'>{escape(text)}</div>"


def _fmt_percent_safe(value, decimals: int = 2) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "--"
        return f"{float(value):.{decimals}%}"
    except Exception:
        return "--"


def _fmt_str(value, default: str = "--") -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except Exception:
        return default
    return text if text.strip() else default


def render_report_notes(context: Dict[str, Any] = None) -> str:
    """
    渲染“报告说明”板块（静态模板，不依赖运行时数据）。
    若算法/阈值/权重变动，需要人工同步修改此模板文案。
    """
    run_rows = [
        ("日志文件", "最新运行日志位于 baogao/jianyan/，名称 factor_analysis_log_*.txt"),
        ("调试模式", "可在运行时通过参数/环境变量开启，默认未开启"),
        ("分档与样本", "分档/样本下限请参考配置文件 fa_config.py"),
    ]
    run_table = (
        '<table class="notes-table"><thead><tr><th>项目</th><th>信息</th></tr></thead><tbody>'
        + "".join(f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>" for k, v in run_rows)
        + "</tbody></table>"
    )

    formulas = [
        "年化收益率：日均收益 × 252（线性年化，不做复利）",
        "IC 口径：Spearman（秩相关，默认）",
        "夏普比率：日均收益 ÷ 日标准差 × sqrt(252)",
        "最大回撤：基于日收益序列峰谷计算",
        "平均每笔收益率：单笔收益均值",
    ]

    rule_items = [
        "样本下限、收益/协同增益等阈值：参见 fa_config.py 的对应模块设置",
        "用户区间：若有自定义区间，默认强制展示；若无则按样本下限过滤",
        "展示条数：榜单一般以 Top N 展示（部分报告会倍增显示）",
    ]

    scoring_items = [
        "评分权重：收益/夏普/回撤/样本量等指标按配置权重计算（详见 fa_config.py 或模块内说明）",
        "排序字段：综合得分降序；若有平滑得分，会辅以提示",
        "高亮规则：通常前 3 项蓝色，用户区间黄色，如有告警会另行标注",
    ]

    def _render_list(items):
        return "<ul class=\"notes-list\">" + "".join(f"<li>{escape(str(it))}</li>" for it in items) + "</ul>"

    sections = [
        f"<h3>运行与调试信息</h3>{run_table}",
        f"<h3>口径与公式</h3>{_render_list(formulas)}",
        f"<h3>榜单/入榜规则</h3>{_render_list(rule_items)}",
        f"<h3>评分与排名方式</h3>{_render_list(scoring_items)}",
    ]
    return "".join(f'<div class="sub-block">{block}</div>' for block in sections)


def _format_cell(value) -> str:
    if value is None:
        return "--"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "--"
    except TypeError:
        pass
    if pd.isna(value):
        return "--"
    return str(value)


def render_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    headers: Optional[Sequence[str]] = None,
    formatters: Optional[dict] = None,
    empty_text: str = "暂无可展示的数据",
    cell_classes: Optional[dict] = None,
    table_class: Optional[str] = None,
    row_classes: Optional[dict] = None,
    column_classes: Optional[dict] = None,
    html_columns: Optional[Sequence[str]] = None,
) -> str:
    if df is None or df.empty:
        return render_alert(empty_text, level="warn")
    headers = headers or columns
    formatters = formatters or {}
    html_columns = set(html_columns or [])
    table_cls = "table"
    if table_class:
        table_cls = f"{table_cls} {table_class}"

    header_cells: List[str] = []
    for idx, header in enumerate(headers):
        column_name = columns[idx] if idx < len(columns) else None
        header_class = ""
        if column_name and column_classes and column_name in column_classes:
            header_class = f" class='{column_classes[column_name]}'"
        header_cells.append(f"<th{header_class}>{escape(str(header))}</th>")
    thead = "".join(header_cells)

    rows_html: List[str] = []
    for _, row in df.iterrows():
        row_idx = row.name
        row_class_attr = ""
        if row_classes and row_idx in row_classes:
            row_class_attr = f" class='{row_classes[row_idx]}'"
        cells: List[str] = []
        for col in columns:
            value = row[col] if col in row else None
            formatter: Optional[Callable] = formatters.get(col)
            if formatter:
                try:
                    cell_value = formatter(value)
                except Exception:
                    cell_value = _format_cell(value)
            else:
                cell_value = _format_cell(value)
            display_value = "--" if cell_value is None else str(cell_value)

            class_parts: List[str] = []
            if column_classes and col in column_classes:
                class_parts.append(column_classes[col])
            if cell_classes and row_idx in cell_classes and col in cell_classes[row_idx]:
                class_parts.append(cell_classes[row_idx][col])
            cell_class_attr = f" class='{' '.join(class_parts)}'" if class_parts else ""

            if col in html_columns:
                cell_content = display_value
            else:
                cell_content = escape(display_value)
            cells.append(f"<td{cell_class_attr}>{cell_content}</td>")
        rows_html.append(f"<tr{row_class_attr}>" + "".join(cells) + "</tr>")
    tbody = "".join(rows_html)
    return f"<div class='table-wrapper'><table class='{table_cls}'><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>"



def render_list(items: Iterable[str]) -> str:
    items = [escape(item) for item in items if item]
    if not items:
        return ""
    inner = "".join(f"<li>{item}</li>" for item in items)
    return f"<ul>{inner}</ul>"


def render_text_block(text: str, empty_text: str = "鏆傛棤鍐呭") -> str:
    stripped = (text or "").strip()
    if not stripped:
        return render_alert(empty_text, level="warn")
    return f"<div class='text-block'>{escape(stripped)}</div>"


def build_run_metadata_section(
    report_options: Optional[Dict[str, Any]],
    result_count: int,
) -> Tuple[List[Tuple[str, str]], str]:
    """返回用于 hero meta 和“运行信息”表格的 HTML。"""
    options = report_options or {}
    log_file = options.get("log_file") or "未记录"
    debug_mode = "已开启" if options.get("debug_enabled") else "未开启"
    debug_dump = options.get("debug_dump_path") or "未生成"
    rows = [
        ("分析结果数量", str(result_count)),
        ("日志文件", log_file),
        ("调试模式", debug_mode),
        ("调试原始日志", debug_dump),
    ]
    df = pd.DataFrame(rows, columns=["item", "value"])
    table_html = render_table(
        df,
        columns=["item", "value"],
        headers=["项目", "信息"],
        empty_text="暂无运行信息",
    )
    meta_items: List[Tuple[str, str]] = [
        ("结果数量", str(result_count)),
        ("日志文件", log_file),
        ("Debug", debug_mode),
    ]
    if debug_dump != "未生成":
        meta_items.append(("调试原始日志", debug_dump))
    return meta_items, table_html
