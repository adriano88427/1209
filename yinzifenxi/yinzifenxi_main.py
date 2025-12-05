# -*- coding: utf-8 -*-
"""
因子分析代码 - 修复版本
严格按照代码修改方案和实施计划进行修改

修改内容：
1. 删除重复函数定义 - 保留类内实现，删除外部辅助函数
2. 修复空return语句问题 - 为所有return语句添加返回值
3. 修复变量类型检查问题 - 确保numpy.isnan类型安全
4. 添加数组形状兼容性检查 - 防止broadcast错误
5. 改进异常处理机制 - 为关键函数添加try-catch
6. 验证语法正确性 - 确保代码可以正常编译运行

日期: 2025-11-21
版本: 修复版本
"""
import sys
import os
import warnings
import argparse
import csv
import time
import copy
from typing import Any, Dict
from datetime import datetime

# 兼容直接运行当前文件或通过包导入的场景
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from yinzifenxi.fa_config import (
    DEFAULT_DATA_FILE,
    DEFAULT_DATA_FILES,
    ANALYSIS_SWITCHES,
    DUAL_FACTOR_SETTINGS,
    DUAL_REPORT_OPTIONS,
    AUX_ANALYSIS_OPTIONS,
    validate_dual_config,
)
from yinzifenxi.fa_logging import Logger as ExternalLogger, detect_debug_enabled
from yinzifenxi.fa_nonparam_analysis import (
    FactorAnalysis,
    DEFAULT_PROCESS_FACTORS,
    DEFAULT_FACTOR_METHOD,
    DEFAULT_WINSORIZE,
    DEFAULT_USE_PEARSON,
    DEFAULT_GROUP_COUNT,
)
from yinzifenxi.fa_param_analysis import ParameterizedFactorAnalyzer
from yinzifenxi.fa_nonparam_helpers import (
    _fa_classify_factors_by_ic,
    _fa_generate_factor_classification_overview,
    _fa_get_suggested_weight,
    _fa_get_scoring_standards,
)
from yinzifenxi.fa_nonparam_report import (
    _fa_generate_summary_report,
    _fa_generate_factor_analysis_report,
    _fa_generate_positive_factors_analysis,
    _fa_generate_negative_factors_analysis,
)
try:
    from yinzifenxi.fa_dual_nonparam_analysis import run_dual_nonparam_pipeline
except ImportError:  # pragma: no cover
    run_dual_nonparam_pipeline = None

try:
    from yinzifenxi.fa_dual_param_analysis import run_dual_param_pipeline
except ImportError:  # pragma: no cover
    run_dual_param_pipeline = None

LOG_METRICS_FILE = os.path.join("baogao", "log_metrics.csv")

# 读取完整因子数据文件
configured_files = list(DEFAULT_DATA_FILES) if DEFAULT_DATA_FILES else []
if not configured_files and DEFAULT_DATA_FILE:
    configured_files = [DEFAULT_DATA_FILE]

print("[INFO] Using configured data files:")
for path in configured_files:
    print(f"  - {path}")

# 检查文件是否存在
available_files = []
for path in configured_files:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        print(f"[INFO] Data file path: {abs_path}")
        available_files.append(path)
    else:
        print(f"[WARN] Data file missing, skipped: {abs_path}")

if not available_files:
    print("[ERROR] No usable data file. Please verify configuration and retry.")
    sys.exit(1)

# 尝试导入scipy.stats，如果不可用则设置标志
HAS_SCIPY = False
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    print("[WARN] scipy is unavailable. Some statistics will be simplified but the analysis will continue.")

# 尝试导入matplotlib和seaborn，如果不可用则设置标志
HAS_PLOT = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    HAS_PLOT = True
except ImportError:
    print("[WARN] matplotlib/seaborn are unavailable. Visualization features are disabled but core analysis continues.")

# 稳健性统计与敏感性分析函数现由 yinzifenxi.fa_stat_utils 提供，主脚本不再重复定义。

# FactorAnalysis 已迁移至 yinzifenxi.fa_nonparam_analysis 模块

# ParameterizedFactorAnalyzer 已迁移至 yinzifenxi.fa_param_analysis 模块
# 使用时请从 yinzifenxi.fa_param_analysis 导入


# 主函数示例
def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="????? + ?????????")
    parser.add_argument(
        "--summary-report",
        action="store_true",
        help="????????????????????????",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="???????????????? FA_DEBUG=1?",
    )
    parser.add_argument(
        "--dump-debug-log",
        help="??????????????????????????? baogao?",
    )
    return parser.parse_args(argv)


def _summarize_extra_stat(value, max_preview=5):
    """
    将calculate_ic返回的extra_stats复杂数据压缩成可读摘要，避免一次性打印超长内容导致Windows控制台抛出
    [Errno 22] Invalid argument错误，也便于用户快速理解核心指标。
    """
    import numpy as _np

    if value is None:
        return "None"
    if isinstance(value, (int, float, _np.number)):
        return f"{float(value):.4f}"
    if isinstance(value, (list, tuple, _np.ndarray)):
        try:
            arr = _np.asarray(value)
        except Exception:
            arr = _np.array(value, dtype=object)
        length = arr.size
        if length == 0:
            return "空列表"
        if _np.issubdtype(arr.dtype, _np.number):
            return (
                f"{length}条 | 均值 {arr.mean():.4f} | 中位 {float(_np.median(arr)):.4f} "
                f"| 最小 {arr.min():.4f} | 最大 {arr.max():.4f}"
            )
        preview = ", ".join(str(item) for item in list(arr[:max_preview]))
        suffix = "..." if length > max_preview else ""
        return f"{length}条: [{preview}{suffix}]"
    if isinstance(value, dict):
        items = list(value.items())
        if not items:
            return "空字典"
        preview_items = ", ".join(f"{k}: {v}" for k, v in items[:max_preview])
        suffix = "..." if len(items) > max_preview else ""
        return f"{len(items)}项: {{{preview_items}{suffix}}}"
    text = str(value)
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _safe_ir(ic_mean_value, ic_std_value):
    if ic_std_value is None:
        return np.nan
    if isinstance(ic_std_value, (int, float, np.number)) and not np.isnan(ic_std_value) and ic_std_value != 0:
        return ic_mean_value / ic_std_value
    return np.nan


def main(argv=None):
    """主函数"""
    args = parse_cli_args(argv)
    if args.debug:
        os.environ["FA_DEBUG"] = "1"
    if args.dump_debug_log:
        os.environ["FA_DEBUG_DUMP"] = args.dump_debug_log

    debug_enabled = detect_debug_enabled(args.debug)
    dump_debug_log = args.dump_debug_log or os.environ.get("FA_DEBUG_DUMP")
    overall_start = time.perf_counter()
    # Redirect stdout to the structured logger so every message enters the log file.
    logger = ExternalLogger(
        log_all=True if debug_enabled else None,
        include_warnings=True if debug_enabled else None,
        debug_enabled=debug_enabled,
        debug_dump_path=dump_debug_log,
    )
    sys.stdout = logger

    print("[INFO] Factor analysis pipeline started")
    print(f"[INFO] Log file: {logger.log_file}")
    if debug_enabled:
        print("[DEBUG] Debug mode enabled via --debug or FA_DEBUG=1")
        if logger.debug_dump_path:
            print(f"[DEBUG] Raw debug log: {logger.debug_dump_path}")

    # 创建因子分析对象
    analyzer = FactorAnalysis(file_path=available_files)

    # 加载数据
    load_start = time.perf_counter()
    print("[INFO] Loading data...")
    load_success = analyzer.load_data()
    load_elapsed = time.perf_counter() - load_start
    if not load_success:
        print(f"[ERROR] Failed to load data after {load_elapsed:.2f}s. Exiting.")
        logger.close()
        return
    else:
        print(f"[INFO] Data loaded in {load_elapsed:.2f}s. Raw sample size: {len(analyzer.data)} rows")

    # 数据完整性校验
    if not analyzer.validate_data_sources():
        print("[ERROR] Data validation failed. Please inspect your input files and retry.")
        logger.close()
        return

    # 预处理数据
    print("\n[INFO] === Data Preprocessing ===")

    use_pearson = DEFAULT_USE_PEARSON
    process_factors = DEFAULT_PROCESS_FACTORS
    factor_method = DEFAULT_FACTOR_METHOD
    winsorize = DEFAULT_WINSORIZE
    group_count = DEFAULT_GROUP_COUNT

    preprocess_start = time.perf_counter()
    preprocess_ok = analyzer.preprocess_data()
    preprocess_elapsed = time.perf_counter() - preprocess_start
    if not preprocess_ok:
        print(f"[ERROR] Data preprocessing failed after {preprocess_elapsed:.2f}s. Exiting.")
        logger.close()
        return
    else:
        processed_rows = len(getattr(analyzer, 'processed_data', []))
        print(f"[INFO] Data preprocessing complete in {preprocess_elapsed:.2f}s. Remaining samples: {processed_rows} rows")

    coverage_start = time.perf_counter()
    if not analyzer.validate_processed_coverage():
        coverage_elapsed = time.perf_counter() - coverage_start
        print(f"[ERROR] Sample coverage is insufficient after {coverage_elapsed:.2f}s. Analysis terminated.")
        logger.close()
        return
    coverage_elapsed = time.perf_counter() - coverage_start
    print(f"[INFO] Sample coverage validation complete in {coverage_elapsed:.2f}s.")

    print("\n[INFO] === Factor Analysis Options ===")
    print("\n[INFO] Available factors:")
    for i, factor in enumerate(analyzer.factors, 1):
        print(f"{i}. {factor}")

    print("\n[INFO] Running full factor analysis...")
    factor_stage_start = time.perf_counter()
    try:
        fa_compute_start = time.perf_counter()
        analyzer.run_factor_analysis(use_pearson=use_pearson)
        fa_compute_elapsed = time.perf_counter() - fa_compute_start
        print(f"[INFO] Factor analysis core computation took {fa_compute_elapsed:.2f}s with {len(analyzer.analysis_results)} factor results")

        has_results = hasattr(analyzer, 'analysis_results') and analyzer.analysis_results
        if has_results:
            print("\n[INFO] Integrating auxiliary robustness statistics...")
            aux_options = copy.deepcopy(AUX_ANALYSIS_OPTIONS or {})
            aux_options["debug_enabled"] = bool(aux_options.get("debug_enabled")) or bool(debug_enabled)
            if debug_enabled:
                aux_options["log_details"] = True
            aux_mode = str(aux_options.get("mode") or "full").strip().lower()
            aux_enabled = bool(aux_options.get("enabled", True))
            aux_skip = (not aux_enabled) or aux_mode in ("skip", "off", "disabled")
            aux_window_sizes = tuple(aux_options.get("window_sizes") or (30, 60))
            aux_sample_sizes = tuple(aux_options.get("sample_sizes") or (0.8, 0.9, 1.0))
            aux_iterations = int(aux_options.get("n_iterations") or 100)
            aux_stats = None
            if aux_skip:
                print(f"[INFO] Auxiliary analysis skipped by configuration (enabled={aux_enabled}, mode={aux_mode}).")
            else:
                if aux_mode == "fast":
                    print("[INFO] Auxiliary analysis running in FAST mode; results may differ from full mode.")
                try:
                    aux_start = time.perf_counter()
                    aux_stats = analyzer.generate_auxiliary_analysis_report(
                        window_sizes=aux_window_sizes,
                        sample_sizes=aux_sample_sizes,
                        n_iterations=aux_iterations,
                        options=aux_options,
                    )
                    aux_elapsed = time.perf_counter() - aux_start
                    if not aux_stats:
                        print(f"[WARN] No auxiliary statistics available; skipped after {aux_elapsed:.2f}s. The report will contain baseline metrics only.")
                    else:
                        print(f"[INFO] Auxiliary robustness statistics merged successfully in {aux_elapsed:.2f}s")
                except Exception as aux_err:
                    print(f"[ERROR] Failed to merge auxiliary statistics: {str(aux_err)}")

            summary_start = time.perf_counter()
            summary_df = _fa_generate_summary_report(analyzer)
            summary_elapsed = time.perf_counter() - summary_start
            print(f"[INFO] Summary dataframe generated in {summary_elapsed:.2f}s")

            report_start = time.perf_counter()
            _fa_generate_factor_analysis_report(
                analyzer,
                summary_df,
                process_factors=process_factors,
                factor_method=factor_method,
                winsorize=winsorize,
                summary_mode=args.summary_report,
            )
            report_elapsed = time.perf_counter() - report_start
            print(f"[INFO] Main single-factor report generated in {report_elapsed:.2f}s")
        else:
            print("[WARN] No factor analysis results were produced; report skipped")
    except Exception as e:
        print(f"[ERROR] Encountered an exception during factor analysis: {str(e)}")
    finally:
        factor_elapsed = time.perf_counter() - factor_stage_start
        print(f"[INFO] Factor analysis stage ended in {factor_elapsed:.2f}s.")

    grouped_stage_start = time.perf_counter()
    print(f"\n[INFO] Running {group_count}-bucket grouped analysis for every factor...")

    unavailable = getattr(analyzer, 'unavailable_factors', set())
    selected_factors = [f for f in analyzer.factors if f not in unavailable]
    if unavailable:
        print("[WARN] Skipped factors due to parsing issues: " + ", ".join(sorted(unavailable)))
    for factor_name in selected_factors:
        print(f"\n[STEP] Analyzing factor: {factor_name}")
        try:
            cached_result = analyzer.analysis_results.get(factor_name)
            if cached_result:
                ic_mean = cached_result.get('ic_mean', np.nan)
                ic_std = cached_result.get('ic_std', np.nan)
                t_stat = cached_result.get('t_stat', np.nan)
                p_value = cached_result.get('p_value', np.nan)
                ir = cached_result.get('ir', np.nan)
                extra_stats = cached_result.get('extra_stats')
            else:
                ic_mean, ic_std, t_stat, p_value, extra_stats = analyzer.calculate_ic(
                    factor_name,
                    use_pearson=use_pearson,
                )
                ir = _safe_ir(ic_mean, ic_std)
                cached_result = {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'ir': ir,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'extra_stats': extra_stats,
                }
                analyzer.analysis_results[factor_name] = cached_result

            group_results = cached_result.get('group_results')
            if not group_results:
                group_results = analyzer.calculate_group_returns(factor_name, n_groups=group_count)
                if group_results:
                    cached_result['group_results'] = group_results

            if not group_results:
                print(f"[WARN] Unable to compute grouped returns for factor '{factor_name}'")
                continue

            avg_returns = group_results['avg_returns']
            long_short_return = (
                group_results['long_short_return']
                if not np.isnan(group_results['long_short_return'])
                else 0
            )
            ir = ir if not np.isnan(ir) else _safe_ir(ic_mean, ic_std)

            print(f"IC mean: {ic_mean:.4f}")
            print(f"IC std: {ic_std:.4f}")
            print(f"Information ratio: {ir:.4f}" if not np.isnan(ir) else "Information ratio: N/A")
            print(f"Long-short return: {long_short_return:.4f}%")
            print(f"\n{group_count}-bucket grouped returns:")
            print(avg_returns.to_string(index=False, float_format='%.3f'))
            print(f"  Detailed grouped returns for factor {factor_name} will be included in the parameterized analyzer output")

            extra_stats = cached_result.get('extra_stats')
            if extra_stats:
                print("  Additional robustness statistics (summary):")
                for key, value in extra_stats.items():
                    print(f"    {key}: {_summarize_extra_stat(value)}")
        except Exception as e:
            print(f"[ERROR] Failed to analyze factor '{factor_name}': {str(e)}")
    grouped_elapsed = time.perf_counter() - grouped_stage_start
    print(f"[INFO] Grouped factor walkthrough completed in {grouped_elapsed:.2f}s.")

    print("\n[INFO] === Factor analysis results saved ===")

    print("\n[INFO] Generating parameterized factor summary report...")
    parameterized_analyzer = None
    param_stage_start = time.perf_counter()
    try:
        parameterized_analyzer = ParameterizedFactorAnalyzer(analyzer.data.copy())
        if parameterized_analyzer.preprocess_data():
            report_filename = parameterized_analyzer.generate_parameterized_report()
            if report_filename:
                print(f"[OK] Parameterized factor HTML report generated: {report_filename}")
                print("The report includes:")
                print("  -  Factor rankings (positive and negative factors listed separately)")
                print("  -  Top factor recommendations (three best long and three best short factors)")
                print("  -  Detailed factor breakdowns with all metrics")
                print(f"  -  Group-level performance for each factor with {group_count} buckets")
                print("  -  Portfolio construction suggestions based on factor strength")
                print("  -  Risk disclosures and usage notes")
                print("\nSynchronized Excel export:")
                print("  -  Parameterized factor scores_[timestamp].xlsx (composite scores with highlights)")
            else:
                print("[ERROR] Failed to generate the parameterized factor HTML report")
        else:
            print("[ERROR] Parameterized factor preprocessing failed")
    except Exception as e:
        print(f"[ERROR] Exception while building parameterized factor report: {str(e)}")
    finally:
        param_elapsed = time.perf_counter() - param_stage_start
        print(f"[INFO] Parameterized factor stage finished in {param_elapsed:.2f}s.")

    dual_config = validate_dual_config()
    _run_dual_nonparam_workflow(analyzer, dual_config, logger)
    if parameterized_analyzer is not None:
        _run_dual_param_workflow(parameterized_analyzer, dual_config, logger)
    else:
        print("[WARN] Parameterized factor data unavailable; skipping dual-factor parametric analysis")

    total_elapsed = time.perf_counter() - overall_start
    print("\n[INFO] Factor analysis program completed")
    print(f"[INFO] Total runtime: {total_elapsed:.2f}s")
    logger.close()


def _with_logger_metadata(base_options, logger):
    """复制报表配置并附带日志调试信息。"""
    options = dict(base_options)
    options["debug_enabled"] = detect_debug_enabled()
    log_file = getattr(logger, "log_file", None) if logger else None
    if log_file:
        options["log_file"] = log_file
    dump_path = getattr(logger, "debug_dump_path", None) if logger else None
    if not dump_path:
        dump_path = os.environ.get("FA_DEBUG_DUMP")
    if dump_path:
        options["debug_dump_path"] = dump_path
    return options


def _record_run_metrics(pipeline_name: str, outputs: Dict[str, Any], report_options: Dict[str, Any]):
    """将双因子运行结果追加记录到 baogao/log_metrics.csv。"""
    if not outputs:
        return
    try:
        result_count = int(outputs.get("result_count", 0))
    except (TypeError, ValueError):
        result_count = 0
    duration = outputs.get("duration_sec")
    if duration is not None:
        try:
            duration_str = f"{float(duration):.4f}"
        except (TypeError, ValueError):
            duration_str = str(duration)
    else:
        duration_str = ""
    csv_path = outputs.get("csv_path") or ""
    html_path = outputs.get("html_path") or ""
    excel_path = outputs.get("excel_path") or ""
    log_file = report_options.get("log_file") or ""
    debug_dump = report_options.get("debug_dump_path") or ""
    debug_flag = "1" if report_options.get("debug_enabled") else "0"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(LOG_METRICS_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_METRICS_FILE)
    headers = [
        "timestamp",
        "pipeline",
        "result_count",
        "duration_sec",
        "csv_path",
        "html_path",
        "excel_path",
        "log_file",
        "debug_dump_path",
        "debug_enabled",
    ]
    row = [
        timestamp,
        pipeline_name,
        str(result_count),
        duration_str,
        csv_path,
        html_path,
        excel_path,
        log_file,
        debug_dump,
        debug_flag,
    ]
    with open(LOG_METRICS_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


def _run_dual_nonparam_workflow(analyzer, dual_config, logger):
    """执行非参数双因子分析（可选）。"""
    if not ANALYSIS_SWITCHES.get("dual_nonparam"):
        return
    if not run_dual_nonparam_pipeline:
        print("[WARN] Dual-factor non-parametric module is unavailable; skipping")
        return
    print("\n[INFO] === Dual-factor non-parametric analysis ===")
    report_options = _with_logger_metadata(DUAL_REPORT_OPTIONS, logger)
    try:
        outputs = run_dual_nonparam_pipeline(
            analyzer,
            dual_config,
            report_options,
            logger=logger,
        )
        if outputs:
            csv_path = outputs.get("csv_path")
            html_path = outputs.get("html_path")
            if csv_path:
                print(f"[OK] Dual-factor non-parametric CSV generated: {csv_path}")
            if html_path:
                print(f"[OK] Dual-factor non-parametric HTML generated: {html_path}")
            _record_run_metrics("dual_nonparam", outputs, report_options)
    except Exception as exc:
        print(f"[ERROR] Dual-factor non-parametric analysis failed: {exc}")


def _run_dual_param_workflow(parameterized_analyzer, dual_config, logger):
    """执行带参数双因子分析（可选）。"""
    if not ANALYSIS_SWITCHES.get("dual_param"):
        return
    if not run_dual_param_pipeline:
        print("[WARN] Dual-factor parametric module is unavailable; skipping")
        return
    print("\n[INFO] === Dual-factor parametric analysis ===")
    report_options = _with_logger_metadata(DUAL_REPORT_OPTIONS, logger)
    report_options["param_factor_pairs"] = dual_config.get("param_factor_pairs")
    try:
        outputs = run_dual_param_pipeline(
            parameterized_analyzer,
            dual_config,
            report_options,
            logger=logger,
        )
        if outputs:
            csv_path = outputs.get("csv_path")
            html_path = outputs.get("html_path")
            excel_path = outputs.get("excel_path")
            if csv_path:
                print(f"[OK] Dual-factor parametric CSV generated: {csv_path}")
            if excel_path:
                print(f"[OK] Dual-factor parametric Excel generated: {excel_path}")
            if html_path:
                print(f"[OK] Dual-factor parametric HTML generated: {html_path}")
            _record_run_metrics("dual_param", outputs, report_options)
    except Exception as exc:
        print(f"[ERROR] Dual-factor parametric analysis failed: {exc}")


if __name__ == "__main__":
    main()
