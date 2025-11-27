# -*- coding: utf-8 -*-
# 通用统计与工具函数模块（fa_stat_utils）
#
# 当前阶段：仅抽取并复刻 yinzifenxi1119.py 中的通用函数实现，不修改原脚本调用逻辑。

import numpy as np
import scipy


# ========== 类型检查相关工具函数（与原脚本保持一致） ==========

def ensure_list(obj, obj_name="object"):
    """
    确保对象转成 list。
    - list -> 原样
    - tuple/np.ndarray/pd.Series -> list(...)
    - 标量 -> [value]
    仅在无法转换时返回空列表。
    """
    if isinstance(obj, list):
        return obj
    try:
        if isinstance(obj, (tuple, np.ndarray)):
            return [elem for elem in obj]
        # pandas Series 支持 iter
        if hasattr(obj, "tolist"):
            converted = obj.tolist()
            if isinstance(converted, list):
                return converted
            return [converted]
        if isinstance(obj, (int, float, np.number)):
            return [float(obj)]
        if obj is None:
            return []
        return list(obj)
    except Exception as exc:
        print(f"  警告: {obj_name} 转换列表失败 ({exc})，返回空列表")
        return []


def safe_len(obj, obj_name="object"):
    """安全获取对象长度，防止len() of unsized object错误"""
    try:
        if isinstance(obj, (list, tuple, np.ndarray)):
            return len(obj)
        else:
            print(f"  警告: {obj_name} 类型 {type(obj)} 不支持len()操作，重置为0")
            return 0
    except Exception as e:
        print(f"  错误: 获取{obj_name}长度时出错 {e}，重置为0")
        return 0


def safe_ensure_list(obj, obj_name="object"):
    """安全版本的 ensure_list，内部直接复用 ensure_list"""
    return ensure_list(obj, obj_name=obj_name)


def normalize_sequence(obj, obj_name="object", allow_scalar=True):
    """
    统一处理 numpy scalar / list-like / None:
    - 如果是列表或可迭代，返回 list
    - 如果是单个数值且 allow_scalar，则返回 [value]
    - 其余情况返回 []
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if allow_scalar and isinstance(obj, (int, float, np.number)):
        return [float(obj)]
    try:
        return ensure_list(obj, obj_name=obj_name)
    except Exception as exc:
        print(f"  警告: {obj_name} normalize 失败 ({exc})，返回空列表")
        return []


def bounded_normalize(weight_map, min_ratio=0.0, max_ratio=1.0):
    """
    对权重字典进行归一化，并在需要时限制上下界。
    """
    if not weight_map:
        return {}
    weights = {}
    for key, value in weight_map.items():
        try:
            weights[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            weights[key] = 0.0
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}
    weights = {key: value / total for key, value in weights.items()}
    min_ratio = max(0.0, min_ratio or 0.0)
    max_ratio = max(min_ratio, max_ratio or 1.0)

    for _ in range(8):
        total = sum(weights.values())
        if total <= 0:
            break
        weights = {key: value / total for key, value in weights.items()}
        adjusted = False

        low_keys = [k for k, v in weights.items() if v < min_ratio - 1e-9]
        if low_keys:
            deficit = sum(min_ratio - weights[k] for k in low_keys)
            for k in low_keys:
                weights[k] = min_ratio
            redistribute = [k for k in weights if k not in low_keys]
            if redistribute:
                share = deficit / len(redistribute)
                for k in redistribute:
                    weights[k] = max(min_ratio, weights[k] - share)
            adjusted = True

        high_keys = [k for k, v in weights.items() if v > max_ratio + 1e-9]
        if high_keys:
            surplus = sum(weights[k] - max_ratio for k in high_keys)
            for k in high_keys:
                weights[k] = max_ratio
            redistribute = [k for k in weights if k not in high_keys]
            if redistribute:
                share = surplus / len(redistribute)
                for k in redistribute:
                    weights[k] = min(max_ratio, weights[k] + share)
            adjusted = True

        if not adjusted:
            break

    total = sum(weights.values())
    if total > 0:
        weights = {key: value / total for key, value in weights.items()}
    return weights


# ========== 稳健性统计方法辅助函数（导出实现版本） ==========

def kendall_tau_corr(x, y):
    """对外暴露的 Kendall Tau 计算接口"""
    return kendall_tau_corr_impl(x, y)


def robust_correlation(x, y, method='median'):
    """稳健相关系数计算"""
    return robust_correlation_impl(x, y, method=method)


def mann_whitney_u_test(x, y):
    """Mann-Whitney U 检验"""
    return mann_whitney_u_test_impl(x, y)


def bootstrap_confidence_interval(x, y=None, statistic='correlation', n_bootstrap=1000, confidence_level=0.95):
    """Bootstrap 置信区间（y可选，默认对单样本求均值区间）"""
    return bootstrap_confidence_interval_impl(
        x,
        y,
        statistic=statistic,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def detect_outliers(x, method='iqr'):
    """异常值检测"""
    return detect_outliers_impl(x, method=method)


def sensitivity_analysis(x, y, outlier_methods=('iqr', 'zscore'), include_outliers=True):
    """敏感性分析"""
    return sensitivity_analysis_impl(x, y, outlier_methods=outlier_methods, include_outliers=include_outliers)


def false_discovery_control(p_values, method='bh', alpha=0.05):
    """多重检验校正"""
    return false_discovery_control_impl(p_values, method=method, alpha=alpha)


def rolling_window_analysis(df, factor_col, return_col, window_sizes=(30, 60),
                            compute_ic_decay=True, save_plots=False):
    """滚动窗口分析"""
    return rolling_window_analysis_impl(
        df,
        factor_col,
        return_col,
        window_sizes=window_sizes,
        compute_ic_decay=compute_ic_decay,
        save_plots=save_plots,
    )


def temporal_stability_analysis(factor_results):
    """时序稳定性分析"""
    return temporal_stability_analysis_impl(factor_results)


def sample_sensitivity_analysis(df, factor_col, return_col,
                                sample_sizes=(0.8, 0.9, 1.0), n_iterations=100):
    """样本敏感性分析"""
    return sample_sensitivity_analysis_impl(
        df,
        factor_col,
        return_col,
        sample_sizes=sample_sizes,
        n_iterations=n_iterations,
    )


# 下面是真正带实现的版本，从拆分脚本原样复制过来（当前不在主流程中直接调用）

def kendall_tau_corr_impl(x, y):
    """
    计算Kendall's Tau相关系数（不依赖scipy）
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.nan

    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan

    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1

    tau = (concordant - discordant) / (n * (n - 1) / 2)
    return tau


def robust_correlation_impl(x, y, method='median'):
    """
    计算稳健相关系数
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.nan

    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan

    if method == 'median':
        def median_abs_deviation(data):
            return np.median(np.abs(data - np.median(data)))

        n = len(x)
        x_centered = x - np.median(x)
        y_centered = y - np.median(y)

        x_mad = median_abs_deviation(x)
        y_mad = median_abs_deviation(y)

        if x_mad == 0 or y_mad == 0:
            return np.nan

        robust_x = np.sign(x_centered) * np.minimum(np.abs(x_centered), 3 * x_mad)
        robust_y = np.sign(y_centered) * np.minimum(np.abs(y_centered), 3 * y_mad)

        return np.corrcoef(robust_x, robust_y)[0, 1]

    elif method == 'trimmed_mean':
        n = len(x)
        trim_pct = 0.1
        trim_count = int(n * trim_pct)

        x_sorted = np.sort(x)[trim_count:-trim_count]
        y_sorted = np.sort(y)[trim_count:-trim_count]

        if len(x_sorted) < 2 or len(y_sorted) < 2:
            return np.corrcoef(x, y)[0, 1]

        return np.corrcoef(x_sorted, y_sorted)[0, 1]

    else:
        return np.corrcoef(x, y)[0, 1]


def mann_whitney_u_test_impl(x, y):
    """
    进行Mann-Whitney U检验（非参数检验）
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if len(x) < 1 or len(y) < 1:
            return np.nan, np.nan

        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1

        R1 = np.sum(ranks[:len(x)])
        n1, n2 = len(x), len(y)
        U1 = R1 - n1 * (n1 + 1) / 2
        U2 = n1 * n2 - U1
        U = min(U1, U2)

        n = n1 + n2
        mean_U = n1 * n2 / 2
        var_U = n1 * n2 * (n + 1) / 12

        if var_U == 0:
            return U, 1.0

        z = (U - mean_U) / np.sqrt(var_U)

        if 'stats' in dir(scipy):
            p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        else:
            import math
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

        return U, p_value

    except Exception:
        return np.nan, np.nan


def bootstrap_confidence_interval_impl(x, y=None, statistic='correlation', n_bootstrap=1000, confidence_level=0.95):
    """
    计算Bootstrap置信区间
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        if len(x) != len(y):
            return np.nan, np.nan, []

    n = len(x)
    if n < 2:
        return np.nan, np.nan, []

    bootstrap_stats = []

    def correlation_stat(data1, data2):
        return np.corrcoef(data1, data2)[0, 1]

    def mean_diff_stat(data1, data2):
        return np.mean(data1) - np.mean(data2)

    def mean_stat(data1, _data2=None):
        return np.mean(data1)

    if statistic == 'correlation':
        if y is None or len(y) < 2:
            return np.nan, np.nan, []
        stat_func = correlation_stat
    elif statistic == 'mean_diff':
        if y is None or len(y) < 2:
            return np.nan, np.nan, []
        stat_func = mean_diff_stat
    else:
        stat_func = mean_stat

    bootstrap_stats = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        boot_x = x[indices]
        boot_y = y[indices] if y is not None and statistic != 'mean' else None

        try:
            stat_value = stat_func(boot_x, boot_y)
            if not np.isnan(stat_value) and np.isfinite(stat_value):
                bootstrap_stats.append(stat_value)
        except Exception:
            continue

    if len(bootstrap_stats) < 10:
        return np.nan, np.nan, bootstrap_stats

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)

    return ci_lower, ci_upper, bootstrap_stats


def detect_outliers_impl(x, method='iqr'):
    """
    异常值诊断实现
    """
    x = np.asarray(x)

    if len(x) < 3:
        return {'outlier_mask': np.zeros(len(x), dtype=bool), 'method': method, 'threshold': np.nan}

    result = {'method': method, 'threshold': np.nan}

    if method == 'iqr':
        Q1 = np.percentile(x, 25)
        Q3 = np.percentile(x, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        result['threshold'] = (lower_bound, upper_bound)
        result['outlier_mask'] = (x < lower_bound) | (x > upper_bound)
        result['mild_outlier_mask'] = (x < lower_bound) | (x > upper_bound)

        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        result['extreme_outlier_mask'] = (x < extreme_lower) | (x > extreme_upper)

    elif method == 'zscore':
        z_scores = np.abs((x - np.mean(x)) / np.std(x))
        threshold = 3
        result['threshold'] = threshold
        result['outlier_mask'] = z_scores > threshold

    elif method == 'modified_zscore':
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        modified_z_scores = 0.6745 * (x - median) / mad
        threshold = 3.5
        result['threshold'] = threshold
        result['outlier_mask'] = np.abs(modified_z_scores) > threshold

    return result


def sensitivity_analysis_impl(x, y, outlier_methods=('iqr', 'zscore'), include_outliers=True):
    """
    敏感性分析实现：包含 vs 剔除异常值的对比
    """
    if len(x) != len(y) or len(x) < 3:
        return {'error': '数据长度不足或不一致'}

    x = np.asarray(x)
    y = np.asarray(y)
    results = {}

    original_corr = np.corrcoef(x, y)[0, 1]
    results['original'] = {
        'correlation': original_corr,
        'sample_size': len(x)
    }

    for method in outlier_methods:
        outlier_info = detect_outliers_impl(x, method)
        outlier_mask = outlier_info['outlier_mask']

        if include_outliers:
            x_with_outliers = x[outlier_mask]
            y_with_outliers = y[outlier_mask]
            if len(x_with_outliers) >= 2:
                corr_with_outliers = np.corrcoef(x_with_outliers, y_with_outliers)[0, 1]
                results[f'{method}_with_outliers'] = {
                    'correlation': corr_with_outliers,
                    'sample_size': len(x_with_outliers)
                }

        clean_x = x[~outlier_mask]
        clean_y = y[~outlier_mask]
        if len(clean_x) >= 2:
            corr_without_outliers = np.corrcoef(clean_x, clean_y)[0, 1]
            results[f'{method}_without_outliers'] = {
                'correlation': corr_without_outliers,
                'sample_size': len(clean_x),
                'outliers_removed': np.sum(outlier_mask)
            }

    clean_results = {k: v for k, v in results.items() if 'without_outliers' in k}
    if clean_results:
        correlations = [v['correlation'] for v in clean_results.values() if not np.isnan(v['correlation'])]
        if correlations:
            results['sensitivity'] = {
                'correlation_std': np.std(correlations),
                'correlation_range': np.max(correlations) - np.min(correlations),
                'max_difference_from_original': max(abs(corr - original_corr) for corr in correlations)
            }

    return results


def false_discovery_control_impl(p_values, method='bh', alpha=0.05):
    """
    多重检验校正实现（False Discovery Rate 控制）
    """
    p_values = np.asarray(p_values)

    if len(p_values) == 0:
        return np.array([]), np.array([]), 0

    valid_mask = ~np.isnan(p_values)
    if not np.any(valid_mask):
        return p_values, np.zeros(len(p_values), dtype=bool), 0

    valid_p = p_values[valid_mask]

    if method == 'bh':
        sorted_indices = np.argsort(valid_p)
        sorted_p = valid_p[sorted_indices]
        m = len(valid_p)

        thresholds = (np.arange(1, m + 1) / m) * alpha
        significant_indices = np.where(sorted_p <= thresholds)[0]

        if len(significant_indices) > 0:
            last_significant = significant_indices[-1]
            corrected_p = np.copy(valid_p)
            for i in range(last_significant + 1):
                corrected_p[sorted_indices[i]] = sorted_p[i] * m / (i + 1)
        else:
            corrected_p = valid_p.copy()

    elif method == 'by':
        sorted_indices = np.argsort(valid_p)
        sorted_p = valid_p[sorted_indices]
        m = len(valid_p)

        c_m = np.sum(1.0 / np.arange(1, m + 1))
        thresholds = (np.arange(1, m + 1) / (m * c_m)) * alpha
        significant_indices = np.where(sorted_p <= thresholds)[0]

        if len(significant_indices) > 0:
            last_significant = significant_indices[-1]
            corrected_p = np.copy(valid_p)
            for i in range(last_significant + 1):
                corrected_p[sorted_indices[i]] = sorted_p[i] * c_m * m / (i + 1)
        else:
            corrected_p = valid_p.copy()

    else:
        corrected_p = valid_p.copy()

    full_corrected_p = np.full(len(p_values), np.nan)
    full_corrected_p[valid_mask] = corrected_p

    reject_null = (full_corrected_p <= alpha) & ~np.isnan(full_corrected_p)
    n_significant = np.sum(reject_null)

    return full_corrected_p, reject_null, n_significant


def rolling_window_analysis_impl(df, factor_col, return_col, window_sizes=(30, 60),
                                 compute_ic_decay=True, save_plots=False):
    """
    滚动窗口分析：优化滚动窗口机制
    """
    results = {
        'window_sizes': window_sizes,
        'rolling_ic': {},
        'ic_decay': {},
        'stability_metrics': {}
    }

    df = df.sort_values('信号日期')
    unique_dates = sorted(df['信号日期'].unique())

    for window_size in window_sizes:
        print(f"\n分析窗口大小: {window_size} 个交易日")

        rolling_ics = []
        rolling_dates = []

        for i in range(len(unique_dates) - window_size + 1):
            window_dates = unique_dates[i:i + window_size]
            window_data = df[df['信号日期'].isin(window_dates)]

            if len(window_data) >= window_size * 2:
                valid_data = window_data.dropna(subset=[factor_col, return_col])

                if len(valid_data) >= window_size:
                    try:
                        ic = custom_spearman_corr(valid_data[factor_col], valid_data[return_col])
                        if not np.isnan(ic) and np.isfinite(ic):
                            rolling_ics.append(ic)
                            rolling_dates.append(window_dates[-1])
                    except Exception:
                        continue

        results['rolling_ic'][window_size] = {
            'dates': rolling_dates,
            'ic_values': rolling_ics,
            'mean_ic': np.mean(rolling_ics) if rolling_ics else np.nan,
            'ic_std': np.std(rolling_ics) if rolling_ics else np.nan
        }

        print(f"  有效窗口数: {len(rolling_ics)}")
        if rolling_ics:
            print(f"  平均IC值: {np.mean(rolling_ics):.4f}")
            print(f"  IC标准差: {np.std(rolling_ics):.4f}")

    if compute_ic_decay:
        print("\n计算IC衰减分析...")

        for window_size in window_sizes:
            if window_size in results['rolling_ic'] and results['rolling_ic'][window_size]['ic_values']:
                ic_series = results['rolling_ic'][window_size]['ic_values']

                abs_ics = np.abs(ic_series)
                initial_ic = abs_ics[0] if len(abs_ics) > 0 else 0

                if initial_ic > 0:
                    half_life = None
                    for i, ic_val in enumerate(abs_ics):
                        if ic_val <= initial_ic / 2:
                            half_life = i + 1
                            break

                    results['ic_decay'][window_size] = {
                        'half_life': half_life,
                        'initial_ic': initial_ic,
                        'final_ic': abs_ics[-1] if len(abs_ics) > 0 else np.nan,
                        'decay_rate': (
                            (initial_ic - (abs_ics[-1] if len(abs_ics) > 0 else 0)) / len(abs_ics)
                        ) if len(abs_ics) > 0 else np.nan,
                    }

        for window_size in window_sizes:
            if window_size in results['rolling_ic']:
                ic_values = results['rolling_ic'][window_size]['ic_values']
                if ic_values:
                    results['stability_metrics'][window_size] = {
                        'coefficient_of_variation': (
                            np.std(ic_values) / abs(np.mean(ic_values))
                        ) if np.mean(ic_values) != 0 else np.inf,
                        'persistence': (
                            np.corrcoef(range(len(ic_values)), ic_values)[0, 1]
                            if len(ic_values) > 1 else np.nan
                        ),
                        'mean_abs_ic': np.mean(np.abs(ic_values))
                    }

    return results


def temporal_stability_analysis_impl(factor_results):
    """
    结果稳健性检验：时序稳定性
    """
    stability_results = {
        'ic_stability': {},
        'rank_stability': {},
        'temporal_trends': {}
    }

    if 'ic_values' in factor_results:
        ic_values = factor_results['ic_values']
        if len(ic_values) > 2:
            ic_series = np.array(ic_values)
            lag1_corr = np.corrcoef(ic_series[:-1], ic_series[1:])[0, 1] if len(ic_series) > 2 else np.nan
            x = np.arange(len(ic_series))
            trend_corr = np.corrcoef(x, ic_series)[0, 1]

            stability_results['ic_stability'] = {
                'autocorr_lag1': lag1_corr,
                'trend_correlation': trend_corr,
                'is_stationary': abs(trend_corr) < 0.3,
                'ic_volatility': np.std(ic_series) / abs(np.mean(ic_series)) if np.mean(ic_series) != 0 else np.inf
            }

            trend = np.polyfit(x, ic_series, 1)[0]
            stability_results['temporal_trends'] = {
                'linear_trend': trend,
                'trend_pvalue': np.nan,
                'sign_changes': np.sum(np.diff(np.sign(ic_series)) != 0),
                'mean_reversion_strength': 1 - abs(lag1_corr) if not np.isnan(lag1_corr) else np.nan
            }

    if 'factor_rankings' in factor_results:
        rankings = factor_results['factor_rankings']
        if len(rankings) > 1:
            ranking_changes = []
            for i in range(1, len(rankings)):
                change = np.mean(np.abs(np.array(rankings[i]) - np.array(rankings[i-1])))
                ranking_changes.append(change)

            stability_results['rank_stability'] = {
                'mean_ranking_change': np.mean(ranking_changes) if ranking_changes else np.nan,
                'ranking_volatility': np.std(ranking_changes) if ranking_changes else np.nan,
                'ranking_consistency': 1 - (
                    np.std(ranking_changes) / np.mean(ranking_changes)
                    if ranking_changes and np.mean(ranking_changes) > 0 else np.inf
                ),
            }

    return stability_results


def sample_sensitivity_analysis_impl(df, factor_col, return_col,
                                     sample_sizes=(0.8, 0.9, 1.0), n_iterations=100):
    """
    结果稳健性检验：样本敏感性分析
    """
    sensitivity_results = {
        'sample_size_effects': {},
        'robustness_metrics': {}
    }

    valid_data = df.dropna(subset=[factor_col, return_col])
    total_samples = len(valid_data)

    print(f"\n开始样本敏感性分析（总样本数: {total_samples}）")

    for sample_size in sample_sizes:
        print(f"  分析样本大小: {sample_size*100:.0f}%")
        sample_ics = []

        for iteration in range(n_iterations):
            n_samples = int(total_samples * sample_size)
            sampled_data = valid_data.sample(n=n_samples, random_state=iteration)

            try:
                ic = custom_spearman_corr(sampled_data[factor_col], sampled_data[return_col])
                if not np.isnan(ic) and np.isfinite(ic):
                    sample_ics.append(ic)
            except Exception:
                continue

        if sample_ics:
            sensitivity_results['sample_size_effects'][sample_size] = {
                'ic_mean': np.mean(sample_ics),
                'ic_std': np.std(sample_ics),
                'ic_median': np.median(sample_ics),
                'ic_q25': np.percentile(sample_ics, 25),
                'ic_q75': np.percentile(sample_ics, 75),
                'n_successful_iterations': len(sample_ics),
                'success_rate': len(sample_ics) / n_iterations
            }

            print(f"    成功迭代: {len(sample_ics)}/{n_iterations}")
            print(f"    平均IC: {np.mean(sample_ics):.4f} ± {np.std(sample_ics):.4f}")

    if len(sensitivity_results['sample_size_effects']) > 1:
        ic_means = [stats['ic_mean'] for stats in sensitivity_results['sample_size_effects'].values()]
        ic_stds = [stats['ic_std'] for stats in sensitivity_results['sample_size_effects'].values()]

        sensitivity_results['robustness_metrics'] = {
            'ic_stability_across_samples': (
                np.std(ic_means) / abs(np.mean(ic_means))
            ) if np.mean(ic_means) != 0 else np.inf,
            'mean_variance_across_samples': np.mean(ic_stds),
            'best_sample_size': max(
                sensitivity_results['sample_size_effects'].keys(),
                key=lambda x: abs(sensitivity_results['sample_size_effects'][x]['ic_mean'])
            ),
            'most_stable_sample_size': min(
                sensitivity_results['sample_size_effects'].keys(),
                key=lambda x: sensitivity_results['sample_size_effects'][x]['ic_std']
            )
        }

    return sensitivity_results


def custom_spearman_corr(x, y):
    """
    计算Spearman相关系数，确保数学上的准确性，不添加任何人为限制或修正
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.nan

    if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
        return np.nan

    def rank_with_ties(data):
        arr = np.asarray(data)
        sorted_indices = np.argsort(arr)
        ranks = np.zeros_like(sorted_indices, dtype=float)

        i = 0
        n = len(arr)

        while i < n:
            current_value = arr[sorted_indices[i]]
            j = i

            while j < n and arr[sorted_indices[j]] == current_value:
                j += 1

            rank = (i + 1 + j) / 2

            for k in range(i, j):
                ranks[sorted_indices[k]] = rank

            i = j

        return ranks

    rank_x = rank_with_ties(x)
    rank_y = rank_with_ties(y)

    n = len(rank_x)
    sum_xy = np.sum(rank_x * rank_y)
    sum_x = np.sum(rank_x)
    sum_y = np.sum(rank_y)
    sum_x2 = np.sum(rank_x**2)
    sum_y2 = np.sum(rank_y**2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

    if denominator == 0:
        return np.nan

    corr = numerator / denominator
    corr = min(max(corr, -1.0), 1.0)

    return corr


def calculate_standard_annual_return(total_return_rate, observation_years, method='standard_compound'):
    """
    标准复利/对数方法计算年化收益率，附带数值稳定性检查与验证信息。
    """
    try:
        if total_return_rate is None or observation_years is None:
            return np.nan, {'error': '输入参数包含None值'}
        if not np.isfinite(total_return_rate) or not np.isfinite(observation_years):
            return np.nan, {'error': '输入参数包含无穷大或NaN值'}
        if total_return_rate <= -1:
            return np.nan, {'error': f'总收益率不能小于-100%，实际值: {total_return_rate}'}
        if observation_years <= 0:
            return np.nan, {'error': f'观测期年数必须大于0，实际值: {observation_years}'}
        if observation_years > 100:
            return np.nan, {'error': f'观测期年数过大，可能导致数值不稳定: {observation_years}'}

        final_value = total_return_rate + 1
        if final_value <= 0:
            return np.nan, {'error': f'最终价值倍数必须为正数，实际值: {final_value}'}

        if method == 'log_based':
            log_final_value = np.log(final_value)
            if not np.isfinite(log_final_value):
                return np.nan, {'error': '对数计算溢出，数值不稳定'}
            annual_log_return = log_final_value / observation_years
            annual_return_rate = np.exp(annual_log_return) - 1
            details = {
                'method': 'log_based',
                'final_value': final_value,
                'log_final_value': log_final_value,
                'annual_log_return': annual_log_return,
                'observation_years': observation_years,
            }
        else:
            annual_return_rate = final_value ** (1 / observation_years) - 1
            details = {
                'method': 'standard_compound',
                'final_value': final_value,
                'observation_years': observation_years,
                'calculation': f'{final_value}^(1/{observation_years}) - 1',
            }

        if not np.isfinite(annual_return_rate):
            return np.nan, {'error': '年化收益率计算结果数值不稳定'}
        if abs(annual_return_rate) > 10:
            return np.nan, {'error': f'年化收益率过于极端: {annual_return_rate:.2%}'}

        details.update({
            'annual_return_rate': annual_return_rate,
            'annual_return_percent': annual_return_rate * 100,
            'quality_assessment': {
                'numerical_stability': 'stable' if abs(annual_return_rate) < 5 else ('moderate' if abs(annual_return_rate) < 10 else 'unstable'),
                'return_magnitude': (
                    'extreme_high' if annual_return_rate > 1 else
                    'high' if annual_return_rate > 0.5 else
                    'positive' if annual_return_rate > 0 else
                    'negative'
                ),
                'calculation_reliable': abs(annual_return_rate) < 5,
            }
        })

        reconstructed = (1 + annual_return_rate) ** observation_years - 1
        verification_error = abs(reconstructed - total_return_rate)
        details['verification'] = {
            'reconstructed_total_return': reconstructed,
            'original_total_return': total_return_rate,
            'verification_error': verification_error,
            'verification_passed': verification_error < 1e-6,
        }

        return annual_return_rate, details
    except Exception as exc:
        return np.nan, {
            'error': f'标准复利年化计算错误: {exc}',
            'total_return_rate': total_return_rate,
            'observation_years': observation_years,
        }


def safe_calculate_annual_return(total_return, years, method='standard_compound'):
    """
    包装版年化收益率计算，负责类型转换与异常兜底。
    """
    try:
        if not isinstance(total_return, (int, float, np.number)):
            if isinstance(total_return, str):
                total_return = float(total_return)
            else:
                total_return = 0.0
        if not isinstance(years, (int, float, np.number)):
            if isinstance(years, str):
                years = float(years)
            else:
                years = 1.0
        return calculate_standard_annual_return(total_return, years, method)
    except Exception as exc:
        print(f"  年化计算出错: {exc}")
        return np.nan, {'error': f'年化计算失败: {exc}'}


def validate_annual_return_calculation(annual_return, observation_years, original_total_return, tolerance=0.01):
    """
    通过反向计算验证年化收益率的正确性。
    """
    try:
        if any(val is None for val in (annual_return, observation_years, original_total_return)):
            return {'valid': False, 'error': '输入参数包含None值'}
        if not all(np.isfinite(val) for val in (annual_return, observation_years, original_total_return)):
            return {'valid': False, 'error': '输入参数包含无穷大或NaN值'}
        if observation_years <= 0 or observation_years > 100:
            return {'valid': False, 'error': f'观测期年数无效: {observation_years}'}

        reconstructed = (1 + annual_return) ** observation_years - 1
        absolute_error = abs(reconstructed - original_total_return)
        relative_error = absolute_error / max(1e-9, abs(original_total_return))

        is_valid = relative_error <= tolerance
        validation_result = {
            'valid': is_valid,
            'annual_return': annual_return,
            'observation_years': observation_years,
            'original_total_return': original_total_return,
            'reconstructed_total_return': reconstructed,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'tolerance': tolerance,
        }

        if not is_valid:
            validation_result['warning'] = f'相对误差 {relative_error:.4f} 超过容忍度 {tolerance}'
        return validation_result
    except Exception as exc:
        return {'valid': False, 'error': f'验证过程出错: {exc}'}
