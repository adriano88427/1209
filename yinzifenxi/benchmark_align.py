# -*- coding: utf-8 -*-
"""
外部基准数据对齐脚本（不改主流程/源数据）：
- 读取并预处理数据（复用 FactorAnalysis 预处理）
- 标准化日期/资产代码
- 导出对齐后的基准输入（宽表 + 长表）供 Alphalens/Qlib 等使用
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from yinzifenxi.fa_config import RETURN_COLUMN, REPORT_OUTPUT_DIR
from yinzifenxi.fa_nonparam_analysis import FactorAnalysis


def _normalize_code(code: str) -> str:
    if code is None:
        return ""
    text = str(code).strip().upper().replace(".", "")
    # 保留前导0，数字部分补零到6位
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        digits = digits.zfill(6)
    return digits or text


def prepare_data(view: str = "neutralized") -> pd.DataFrame:
    fa = FactorAnalysis()
    ok = fa.preprocess_data()
    if not ok:
        raise RuntimeError("预处理失败，无法生成基准输入")

    df = fa.processed_data if view != "raw" else fa.pre_neutral_data
    if df is None or df.empty:
        raise RuntimeError(f"{view} 视图数据为空")

    df = df.copy()
    if "信号日期" not in df.columns:
        raise RuntimeError("缺少列 信号日期")

    code_col = None
    for cand in ("股票代码", "证券代码"):
        if cand in df.columns:
            code_col = cand
            break
    if not code_col:
        raise RuntimeError("缺少股票代码列")

    df["信号日期"] = pd.to_datetime(df["信号日期"], errors="coerce")
    df = df.dropna(subset=["信号日期"])
    df["code_std"] = df[code_col].apply(_normalize_code)
    df = df[df["code_std"] != ""]

    # 去重：日期+代码
    before = len(df)
    df = df.drop_duplicates(subset=["信号日期", "code_std"])
    after = len(df)
    print(f"[ALIGN] 去重 {before-after}/{before} (按日期+代码)")

    factor_cols = [c for c in fa.factors if c in df.columns]
    missing_factors = [c for c in fa.factors if c not in df.columns]
    if missing_factors:
        print(f"[ALIGN][WARN] 因子列缺失: {missing_factors}")
    cols_keep: List[str] = ["信号日期", "code_std", RETURN_COLUMN] + factor_cols
    df = df[cols_keep].copy()
    df = df.rename(columns={"信号日期": "date", "code_std": "asset", RETURN_COLUMN: "forward_return"})
    df = df.sort_values(["date", "asset"])
    return df


def export_data(df: pd.DataFrame, output_dir: str, prefix: str = "基准对齐") -> None:
    os.makedirs(output_dir, exist_ok=True)
    wide_path = os.path.join(output_dir, f"{prefix}_宽表.csv")
    long_path = os.path.join(output_dir, f"{prefix}_长表.csv")

    df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    print(f"[ALIGN] 宽表输出: {wide_path} (rows={len(df)}, factors={len(df.columns)-3})")

    factor_cols = [c for c in df.columns if c not in {"date", "asset", "forward_return"}]
    long_df = df.melt(id_vars=["date", "asset", "forward_return"], value_vars=factor_cols,
                      var_name="factor", value_name="factor_value")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"[ALIGN] 长表输出: {long_path} (rows={len(long_df)})")


def main():
    parser = argparse.ArgumentParser(description="外部基准数据对齐导出（宽表+长表）")
    parser.add_argument("--view", default="neutralized", choices=["neutralized", "raw"], help="视图选择，默认中性化")
    parser.add_argument("--output-dir", default=REPORT_OUTPUT_DIR, help="输出目录，默认 baogao/")
    parser.add_argument("--prefix", default="基准对齐", help="输出文件前缀")
    args = parser.parse_args()

    df = prepare_data(view=args.view)
    export_data(df, args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
