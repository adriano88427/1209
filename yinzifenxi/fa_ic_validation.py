# -*- coding: utf-8 -*-
"""
IC 验证辅助工具（轻量版，单文件）。

功能：
- 汇总差异表生成（等权/加权 IC、样本、缺样等）。
- 核心因子统计对比（IC、加权 IC、p 值、t 值、原始 IC 等）。
- 审计日志摘要（统计 [AUDIT] 行，提取因子级计数；包含零/负价过滤计数尝试）。
- 预留外部基准/CI 适配（可选开关，默认关闭）。

CLI 示例：
python -m yinzifenxi.fa_ic_validation \
  --baseline baogao/因子分析汇总_20251206_214717.csv \
  --target baogao/因子分析汇总_20251206_211530.csv \
  --output-dir baogao \
  --audit-log run.log \
  --core-factors 当日回调 机构持股比例(%) \
  --enable-benchmark \
  --enable-ci
"""

from __future__ import annotations

import argparse
import collections
import os
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd


SUMMARY_KEY = "因子名称"
DEFAULT_DIFF_COLS = [
    "IC均值",
    "加权IC均值",
    "IC有效点数",
    "IC总日数",
    "IC缺样比例",
]
DEFAULT_STATS_COLS = [
    "IC均值",
    "加权IC均值",
    "IC标准差",
    "加权IC标准差",
    "p值",
    "原始IC均值",
    "原始p值",
    "t统计量",
]


def _read_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"summary file not found: {path}")
    df = pd.read_csv(path)
    if SUMMARY_KEY not in df.columns:
        raise ValueError(f"缺少关键列 {SUMMARY_KEY} 于 {path}")
    return df


def build_diff(
    baseline_path: str,
    target_path: str,
    output_path: str,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = cols or DEFAULT_DIFF_COLS
    base = _read_summary(baseline_path)
    tgt = _read_summary(target_path)
    merged = base[[SUMMARY_KEY] + cols].merge(
        tgt[[SUMMARY_KEY] + cols],
        on=SUMMARY_KEY,
        suffixes=("_baseline", "_target"),
    )
    for c in cols:
        merged[f"{c}_diff"] = merged[f"{c}_baseline"] - merged[f"{c}_target"]
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def build_core_stats_compare(
    baseline_path: str,
    target_path: str,
    output_path: str,
    core_factors: Optional[Iterable[str]] = None,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = cols or DEFAULT_STATS_COLS
    base = _read_summary(baseline_path)
    tgt = _read_summary(target_path)
    if core_factors:
        core_set = set(core_factors)
        base = base[base[SUMMARY_KEY].isin(core_set)]
        tgt = tgt[tgt[SUMMARY_KEY].isin(core_set)]
    merged = base[[SUMMARY_KEY] + cols].merge(
        tgt[[SUMMARY_KEY] + cols],
        on=SUMMARY_KEY,
        suffixes=("_baseline", "_target"),
    )
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def summarize_audit_log(log_path: str, output_path: str) -> pd.DataFrame:
    """
    轻量审计摘要：统计 [AUDIT] 行数、按因子计数、按类型粗分，尝试解析零/负价过滤。
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"audit log not found: {log_path}")
    factor_counter = collections.Counter()
    kind_counter = collections.Counter()
    zero_neg_counter = 0
    total = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "[AUDIT]" not in line:
                # 额外：简单扫描零/负价字样（如日志未带 [AUDIT] 也可计数）
                if "zero" in line.lower() and "price" in line.lower():
                    zero_neg_counter += 1
                continue
            total += 1
            text = line.strip()
            # 提取因子名：假定格式 "[AUDIT] 因子名: ..."
            if "]" in text:
                try:
                    payload = text.split("]", 1)[1].strip()
                    if ":" in payload:
                        factor = payload.split(":", 1)[0].strip()
                        factor_counter[factor] += 1
                        # 粗分类
                        low = payload.lower()
                        if "winsor" in low:
                            kind_counter["winsor"] += 1
                        elif "跳过原因" in payload:
                            kind_counter["skip_reason"] += 1
                        elif "去重" in payload:
                            kind_counter["dedup"] += 1
                        elif "raw_vs_neutral" in low:
                            kind_counter["raw_vs_neutral"] += 1
                        elif "ic裁剪" in payload or "ic裁剪" in low:
                            kind_counter["ic_clip"] += 1
                        elif "zero" in low and "price" in low:
                            kind_counter["zero_neg_price"] += 1
                            zero_neg_counter += 1
                        else:
                            kind_counter["other"] += 1
                except Exception:
                    kind_counter["parse_error"] += 1
    summary_rows = [
        {"category": "total_audit_lines", "count": total},
        {"category": "zero_neg_price_hits", "count": zero_neg_counter},
    ]
    for k, v in kind_counter.most_common():
        summary_rows.append({"category": f"kind::{k}", "count": v})
    for k, v in factor_counter.most_common():
        summary_rows.append({"category": f"factor::{k}", "count": v})
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def compare_with_benchmark(
    baseline_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    factors: Optional[Iterable[str]],
    output_path: str,
    enabled: bool = False,
) -> Optional[pd.DataFrame]:
    """
    基准对照：对核心因子（可选）对比 baseline 与 benchmark 的 IC、加权 IC，并给出差异。
    benchmark_df 可为外部基准；如未提供则可传入 target_df 实现自比对。
    """
    if not enabled:
        return None
    core_set = set(factors) if factors else None
    base = baseline_df if core_set is None else baseline_df[baseline_df[SUMMARY_KEY].isin(core_set)]
    bench = benchmark_df if core_set is None else benchmark_df[benchmark_df[SUMMARY_KEY].isin(core_set)]
    cols = ["IC均值", "加权IC均值", "p值", "IC有效点数"]
    merged = base[[SUMMARY_KEY] + cols].merge(
        bench[[SUMMARY_KEY] + cols],
        on=SUMMARY_KEY,
        suffixes=("_baseline", "_benchmark"),
    )
    for c in ["IC均值", "加权IC均值"]:
        merged[f"{c}_diff"] = merged[f"{c}_baseline"] - merged[f"{c}_benchmark"]
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def compute_pandas_benchmark(
    df: pd.DataFrame,
    factors: Iterable[str],
    return_col: str,
    weighted: bool = False,
) -> pd.DataFrame:
    """
    纯 pandas 基准：按信号日期逐因子计算 Spearman IC，得到等权/可选加权均值。
    不修改主流程数据，仅用于验证。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[SUMMARY_KEY, "pandas基准IC", "pandas基准IC加权"])
    factors = list(factors or [])
    results: List[Dict[str, Any]] = []
    for factor in factors:
        if factor not in df.columns:
            continue
        sub = df[["信号日期", factor, return_col]].dropna()
        if sub.empty or sub["信号日期"].nunique() == 0:
            continue
        sub["信号日期"] = pd.to_datetime(sub["信号日期"], errors="coerce")
        sub = sub.dropna(subset=["信号日期"])
        if sub.empty:
            continue
        ics = []
        weights = []
        for _, g in sub.groupby("信号日期"):
            if len(g) < 2:
                continue
            corr = g[factor].corr(g[return_col], method="spearman")
            if pd.isna(corr):
                continue
            ics.append(float(corr))
            weights.append(len(g))
        if not ics:
            continue
        ic_mean = float(np.mean(ics))
        ic_weighted = float(np.average(ics, weights=weights)) if (weighted and weights) else np.nan
        results.append(
            {
                SUMMARY_KEY: factor,
                "pandas基准IC": ic_mean,
                "pandas基准IC加权": ic_weighted,
                "pandas基准IC日数": len(ics),
                "pandas基准样本总量": int(sum(weights)),
            }
        )
    return pd.DataFrame(results)




def compute_ci(
    data: pd.DataFrame,
    factors: Optional[Iterable[str]],
    output_path: str,
    enabled: bool = False,
    method: str = "normal",
    bootstrap_iters: int = 2000,
    bootstrap_seed: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    核心因子 CI 估计：
    - normal：均值 ± 1.96 * std / sqrt(n)（与先前一致）
    - bootstrap：基于正态近似生成样本后做重采样（无日度明细时的近似 Bootstrap）
    """
    if not enabled:
        return None
    core_set = set(factors) if factors else None
    df = data if core_set is None else data[data[SUMMARY_KEY].isin(core_set)]
    df = df.copy()
    rng = np.random.default_rng(bootstrap_seed)
    method_norm = (method or "normal").strip().lower()
    method_norm = method_norm if method_norm in {"normal", "bootstrap"} else "normal"
    z = 1.96
    iters = max(int(bootstrap_iters), 1)

    def _calc_ci(mean_val, std_val, n_val):
        if std_val is None or pd.isna(std_val) or n_val is None or pd.isna(n_val) or n_val <= 1:
            return mean_val, mean_val
        n_int = int(max(1, n_val))
        if method_norm == "bootstrap":
            samples = rng.normal(loc=mean_val, scale=std_val, size=n_int)
            boot_means = []
            for _ in range(iters):
                resample = rng.choice(samples, size=n_int, replace=True)
                boot_means.append(resample.mean())
            lower, upper = np.percentile(boot_means, [2.5, 97.5])
            return float(lower), float(upper)
        else:
            se = std_val / (n_val ** 0.5)
            lower = mean_val - z * se
            upper = mean_val + z * se
            return float(lower), float(upper)

    ic_std = df.get("IC标准差")
    ic_n = df.get("IC有效点数")
    w_ic_std = df.get("加权IC标准差")

    ci_lower = []
    ci_upper = []
    w_ci_lower = []
    w_ci_upper = []
    for idx, row in df.iterrows():
        lower, upper = _calc_ci(row["IC均值"], row.get("IC标准差"), row.get("IC有效点数"))
        ci_lower.append(lower)
        ci_upper.append(upper)
        if w_ic_std is not None:
            w_lower, w_upper = _calc_ci(row["加权IC均值"], row.get("加权IC标准差"), row.get("IC有效点数"))
            w_ci_lower.append(w_lower)
            w_ci_upper.append(w_upper)
    df["ic_ci_lower"] = ci_lower
    df["ic_ci_upper"] = ci_upper
    if w_ic_std is not None:
        df["w_ic_ci_lower"] = w_ci_lower
        df["w_ic_ci_upper"] = w_ci_upper
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def run_ic_validation(
    baseline: str,
    target: str,
    output_dir: str,
    audit_log: Optional[str] = None,
    core_factors: Optional[Iterable[str]] = None,
    enable_benchmark: bool = False,
    enable_ci: bool = False,
    benchmark_path: Optional[str] = None,
    ci_method: str = "normal",
    bootstrap_iters: int = 2000,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一入口：生成差异表、核心对比、审计摘要，可选基准/CI。
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, Any] = {}

    diff_path = os.path.join(output_dir, "IC差异对照表.csv")
    stats_path = os.path.join(output_dir, "IC统计对比表.csv")
    outputs["diff"] = build_diff(baseline, target, diff_path)
    outputs["stats"] = build_core_stats_compare(
        baseline, target, stats_path, core_factors=core_factors
    )

    if audit_log:
        audit_out = os.path.join(output_dir, "审计摘要_最新.csv")
        outputs["audit"] = summarize_audit_log(audit_log, audit_out)

    # 加载基线汇总用于后续可选处理
    baseline_df = _read_summary(baseline)
    target_df = _read_summary(target)
    benchmark_df = target_df
    if benchmark_path:
        try:
            benchmark_df = _read_summary(benchmark_path)
        except Exception:
            benchmark_df = target_df
    if enable_benchmark:
        bench_out = os.path.join(output_dir, "IC基准对照表.csv")
        outputs["benchmark"] = compare_with_benchmark(
            baseline_df, benchmark_df, core_factors, bench_out, enabled=True
        )
    if enable_ci:
        ci_out = os.path.join(output_dir, "IC置信区间_核心因子.csv")
        outputs["ci"] = compute_ci(
            baseline_df,
            core_factors,
            ci_out,
            enabled=True,
            method=ci_method,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed,
        )
    return outputs


def run_pandas_benchmark_only(
    data: pd.DataFrame,
    factors: Iterable[str],
    return_col: str,
    output_dir: str,
    weighted: bool = False,
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    仅运行纯 pandas 基准并生成对照表，不触发主流程或修改主报告。
    """
    os.makedirs(output_dir, exist_ok=True)
    bench_df = compute_pandas_benchmark(data, factors, return_col, weighted=weighted)
    bench_path = os.path.join(output_dir, "Pandas基准IC.csv")
    bench_df.to_csv(bench_path, index=False, encoding="utf-8-sig")

    compare_df = None
    compare_path = None
    if summary_path and os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path)
            cols = ["IC均值", "加权IC均值"]
            merged = summary_df[[SUMMARY_KEY] + cols].merge(
                bench_df[[SUMMARY_KEY, "pandas基准IC", "pandas基准IC加权"]],
                on=SUMMARY_KEY,
                how="left",
            )
            merged["pandas基准IC差值"] = merged["IC均值"] - merged["pandas基准IC"]
            merged["pandas基准IC加权差值"] = merged["加权IC均值"] - merged["pandas基准IC加权"]
            compare_df = merged
            compare_path = os.path.join(output_dir, "Pandas基准IC对照.csv")
            merged.to_csv(compare_path, index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] 对照基准时出错（可忽略）: {exc}")

    return {
        "benchmark": bench_df,
        "benchmark_path": bench_path,
        "compare": compare_df,
        "compare_path": compare_path,
    }


def _parse_factor_list(raw: Optional[str]) -> Optional[List[str]]:
    """从字符串解析因子列表，支持逗号/分号分隔。"""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts or None


def main():
    parser = argparse.ArgumentParser(description="IC 验证辅助工具")
    parser.add_argument("--baseline", required=True, help="基线汇总 CSV 路径")
    parser.add_argument("--target", required=True, help="对照汇总 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--audit-log", help="审计日志（含 [AUDIT] 的 log）")
    parser.add_argument("--enable-benchmark", action="store_true", help="开启外部基准占位输出（核心因子，默认关闭）")
    parser.add_argument("--enable-ci", action="store_true", help="开启核心因子 CI/Bootstrap 占位输出（默认关闭）")
    parser.add_argument("--benchmark-path", help="外部基准汇总 CSV（可选，不填则使用 target 自比对）")
    parser.add_argument("--ci-method", default="normal", choices=["normal", "bootstrap"], help="CI 计算方式")
    parser.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap 迭代次数（ci-method=bootstrap 时生效）")
    parser.add_argument("--bootstrap-seed", type=int, help="Bootstrap 随机种子（可选）")
    parser.add_argument(
        "--core-factors",
        nargs="*",
        help="核心因子列表（可选）；为空则对全部因子对比",
    )
    parser.add_argument(
        "--pandas-benchmark-only",
        action="store_true",
        help="仅运行纯 pandas 基准对照（不修改主报告；需要 --data / --return-col）",
    )
    parser.add_argument(
        "--data",
        help="预处理后的数据 CSV（含信号日期、代码、因子、收益列）",
    )
    parser.add_argument(
        "--return-col",
        help="收益列名（与 --data 对应）",
    )
    parser.add_argument(
        "--factors",
        help="因子列表，逗号或分号分隔；空则默认使用数据中的全部因子列（除信号日期/代码/收益）",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="pandas 基准是否按日样本加权计算均值",
    )
    args = parser.parse_args()
    if args.pandas_benchmark_only:
        if not args.data or not args.return_col:
            raise SystemExit("pandas基准模式需要同时提供 --data 与 --return-col")
        df = pd.read_csv(args.data)
        factor_list = _parse_factor_list(args.factors)
        # 自动推断因子列：排除信号日期/代码/收益列
        if not factor_list:
            drop_cols = {"信号日期", "股票代码", "证券代码", args.return_col}
            factor_list = [c for c in df.columns if c not in drop_cols]
        outputs = run_pandas_benchmark_only(
            data=df,
            factors=factor_list,
            return_col=args.return_col,
            output_dir=args.output_dir,
            weighted=args.weighted,
            summary_path=args.target,
        )
        for key, val in outputs.items():
            if val is not None:
                print(f"[IC-PANDAS-BENCH] {key} -> generated ({outputs.get('benchmark_path') or outputs.get('compare_path')})")
        print("[IC-PANDAS-BENCH] done.")
    else:
        outputs = run_ic_validation(
            baseline=args.baseline,
            target=args.target,
            output_dir=args.output_dir,
            audit_log=args.audit_log,
            core_factors=args.core_factors,
            enable_benchmark=args.enable_benchmark,
            enable_ci=args.enable_ci,
            benchmark_path=args.benchmark_path,
            ci_method=args.ci_method,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_seed=args.bootstrap_seed,
        )
        for key, val in outputs.items():
            print(f"[IC-VALID] {key} -> generated")
        print("[IC-VALID] done.")


if __name__ == "__main__":
    main()
