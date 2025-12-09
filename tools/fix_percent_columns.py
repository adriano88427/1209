#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测原始 Excel 百分比列快速修正工具。

执行方式：在项目根目录运行 `python tools/fix_percent_columns.py`
脚本会自动备份原文件，并在检测到 >100% 的列上统一除以 100。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd


def _decode(code: str) -> str:
    return code.encode("ascii").decode("unicode_escape")


def _build_columns(codes: Iterable[str]) -> List[str]:
    return [_decode(code) for code in codes]


PERCENT_COLUMNS_FORCE = _build_columns(
    [
        r"\u8d22\u52a1\u6295\u8d44\u673a\u6784\u5408\u8ba1\uff08\u6295\u8d44\u516c\u53f8+\u79c1\u52df+\u96c6\u5408"
        r"\u7406\u8d22+\u5176\u4ed6\u7406\u8d22+\u5458\u5de5\u6301\u80a1+\u4fe1\u6258+QFII+\u5238\u5546+\u57fa\u91d1\uff09",
        r"\u670b\u53cb\u5408\u8ba1\uff08\u4f01\u4e1a\u5927\u80a1\u4e1c\uff08\u5927\u975e\uff09+\u793e\u4fdd+\u4fdd\u9669\uff09",
        r"\u5341\u5927\u6d41\u901a\u4e2a\u4eba\u6301\u80a1\u5408\u8ba1",
        r"\u9ad8\u7ba1/\u5927\u80a1\u4e1c\u6301\u80a1\u6bd4\u4f8b\u5927\u975e",
        r"\u9ad8\u7ba1/\u5927\u80a1\u4e1c\u6301\u80a1\u6bd4\u4f8b\uff08\u5c0f\u975e\uff09",
        r"\u666e\u901a\u6563\u6237\u6301\u80a1\u6bd4\u4f8b",
        r"\u9999\u6e2f\u4e2d\u592e\u7ed3\u7b97",
        r"\u4f01\u4e1a\u5927\u80a1\u4e1c\u5927\u975e\uff08\u5305\u542b\u56fd\u8d44\uff09",
        r"\u4f01\u4e1a\u5927\u80a1\u4e1c\uff08\u5305\u542b\u56fd\u8d44\uff09\uff08\u5c0f\u975e\uff09",
        r"\u5176\u5b83",
        r"\u6295\u8d44\u516c\u53f8",
        r"\u79c1\u52df\u57fa\u91d1",
        r"\u96c6\u5408\u7406\u8d22\u8ba1\u5212",
        r"\u5176\u4ed6\u7406\u8d22\u4ea7\u54c1",
        r"\u5458\u5de5\u6301\u80a1\u8ba1\u5212",
        r"\u4fe1\u6258\u6301\u4ed3\u5360\u6bd4",
        r"\u793e\u4fdd\u6301\u4ed3\u5360\u6bd4",
        r"QFII\u6301\u4ed3\u5360\u6bd4",
        r"\u4fdd\u9669\u6301\u4ed3\u5360\u6bd4",
        r"\u57fa\u91d1\u6301\u4ed3\u5360\u6bd4",
        r"\u5238\u5546\u6301\u4ed3\u5360\u6bd4",
        r"\u6237\u5747\u6301\u80a1\u6bd4\u4f8b",
        r"\u524d\u5341\u5927\u6d41\u901a\u80a1\u4e1c\u6301\u80a1\u6bd4\u4f8b\u5408\u8ba1(\u62a5\u544a\u671f)(%)",
        r"\u673a\u6784\u6301\u80a1\u6bd4\u4f8b(%)",
        r"\u524d10\u5927\u6d41\u901a\u80a1\u4e1c\u6301\u80a1\u6bd4\u4f8b\u5408\u8ba1",
        r"\u5341\u5927\u6d41\u901a\u80a1\u4e1c\u5c0f\u975e\u5408\u8ba1",
        r"\u5341\u5927\u6d41\u901a\u80a1\u4e1c\u5927\u975e\u5408\u8ba1",
        r"\u5341\u5927\u6d41\u901a\u673a\u6784\u5927\u975e",
        r"\u5341\u5927\u6d41\u901a\u673a\u6784\u5c0f\u975e",
    ]
)

RETURN_COLUMNS_FORCE = _build_columns(
    [
        r"\u5f53\u65e5\u6700\u9ad8\u6da8\u5e45",
        r"\u5f53\u65e5\u6536\u76d8\u6da8\u8dcc\u5e45",
        r"\u5f53\u65e5\u56de\u8c03",
    ]
)


def _scale_columns(
    df: pd.DataFrame,
    columns: List[str],
    report_lines: List[str],
    condition: Callable[[pd.Series], bool],
) -> None:
    for column in columns:
        if column not in df.columns:
            report_lines.append(f"[MISS] 列 {column} 不存在")
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            report_lines.append(f"[SKIP] 列 {column} 无有效数据，跳过")
            continue

        if condition(valid):
            df[column] = numeric / 100.0
            report_lines.append(f"[FIX] {column}: 检测到 >100% 样本，已除以 100")
        else:
            df[column] = numeric
            report_lines.append(f"[KEEP] {column}: 全部 ≤100%，保持原值")


def main() -> None:
    data_path = Path("shuju/biaoge")
    target_file = None
    for candidate in data_path.glob("*2025*.xlsx"):
        if "回测详细数据结合股本分析" in candidate.stem or "创业" in candidate.stem:
            target_file = candidate
            break
    if target_file is None:
        raise FileNotFoundError("未找到 2025 年的回测数据文件")

    backup_file = target_file.with_name(target_file.stem + "_backup" + target_file.suffix)
    if not backup_file.exists():
        shutil.copy2(target_file, backup_file)
        print(f"[INFO] 已创建备份: {backup_file}")

    df = pd.read_excel(target_file)
    report_lines: List[str] = []

    over_100 = lambda series: (series.abs() > 1).any()

    _scale_columns(df, PERCENT_COLUMNS_FORCE, report_lines, over_100)
    _scale_columns(df, RETURN_COLUMNS_FORCE, report_lines, over_100)

    df.to_excel(target_file, index=False)
    print(f"[DONE] 文件已更新: {target_file}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
