# -*- coding: utf-8 -*-
"""
非参数因子分析报告模块，当前版本从 yinzifenxi1119_split.py 中拆分而来。

本模块**已**在 yinzifenxi1119_split.py 中被调用，负责承载所有报告生成逻辑，
以便 FactorAnalysis 主类可以逐步瘦身。所有函数保持与原脚本一致的行为。
"""

import numpy as np
import pandas as pd
from datetime import datetime
from daima.fa_config import DEFAULT_DATA_FILE, build_report_path


def _fa_generate_summary_report(self):
    """
    生成分析汇总报告
    """
    if not self.analysis_results:
        print("错误：请先运行因子分析")
        return

    # 创建汇总表
    summary_data = []
    missing_data_count = 0

    for factor, results in self.analysis_results.items():
        row = {
            '因子名称': factor,
            'IC均值': results['ic_mean'],
            'IC标准差': results['ic_std'],
            'IR值': results['ir'],
            't统计量': results['t_stat'],
            'p值': results['p_value']
        }

        if 'group_results' in results and results['group_results'] is not None:
            row['多空收益'] = results['group_results']['long_short_return']

        extra_stats = results.get('extra_stats') or {}
        if extra_stats:
            row['IC有效点数'] = extra_stats.get('daily_points')
            row['IC总日数'] = extra_stats.get('total_dates')
            row['IC缺样比例'] = extra_stats.get('skip_ratio')
            row['IC模式'] = extra_stats.get('ic_mode')
            row['筛选模式'] = extra_stats.get('screening_mode')
            row['日均样本'] = extra_stats.get('avg_daily_samples')
            row['样本中位数'] = extra_stats.get('median_daily_samples')
            row['样本P25'] = extra_stats.get('p25_daily_samples')
            row['窗口天数'] = extra_stats.get('ic_window_days')
            row['日最小样本'] = extra_stats.get('min_samples_per_day')
            row['整体样本量'] = extra_stats.get('overall_sample_size')
            row['整体回退说明'] = extra_stats.get('fallback_reason')
            row['主要板块'] = extra_stats.get('segment_primary')
            row['主要板块占比'] = extra_stats.get('segment_primary_ratio')
            row['次要板块'] = extra_stats.get('segment_secondary')
            row['次要板块占比'] = extra_stats.get('segment_secondary_ratio')
            row['主要板块IC'] = extra_stats.get('segment_primary_ic')
            row['次要板块IC'] = extra_stats.get('segment_secondary_ic')
            segment_warnings = extra_stats.get('segment_warnings') or []
            row['板块警告'] = "；".join(segment_warnings) if segment_warnings else None
            row['板块建议'] = extra_stats.get('segment_recommendation')

        # 检查缺失值
        if np.isnan(results['ic_std']) or np.isnan(results['ir']):
            missing_data_count += 1
            print(f"警告: 因子 '{factor}' 存在缺失数据 - IC标准差: {results['ic_std']}, IR值: {results['ir']}")

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # 创建显示用的数据框，将缺失值替换为'N/A'
    display_df = summary_df.copy()
    float_cols = ['IC均值', 'IC标准差', 'IR值', 't统计量', 'p值', '多空收益',
                  '日均样本', '样本中位数', '样本P25', '主要板块IC', '次要板块IC']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: 'N/A' if pd.isna(x) else f"{x:.3f}")

    percent_cols = ['IC缺样比例', '主要板块占比', '次要板块占比']
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: 'N/A' if pd.isna(x) else f"{float(x)*100:.1f}%"
            )

    int_cols = ['IC有效点数', 'IC总日数', '窗口天数', '日最小样本', '整体样本量']
    for col in int_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: 'N/A' if pd.isna(x) else f"{int(round(float(x)))}"
            )

    print("\n=== 因子分析汇总报告 ===")
    print(display_df.to_string(index=False))

    # 如果存在缺失数据，显示警告
    if missing_data_count > 0:
        print(f"\n警告: 共有 {missing_data_count} 个因子存在缺失数据，请检查详细信息")

    # 保存汇总报告，设置小数位数为3位
    summary_df_rounded = summary_df.round(3)
    # 添加时间戳到文件名，但避免使用可能导致特定格式报告的命名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'因子分析汇总_{timestamp}.csv'  # 修改文件名格式，避免生成'因子分析报告_当日回调_时间戳.csv'
    file_path = build_report_path(filename)
    summary_df_rounded.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总报告已保存到 '{file_path}'")

    return summary_df


def _fa_generate_factor_analysis_report(self, summary_df, process_factors=False, factor_method='standardize', winsorize=False, summary_mode=False):
    """
    生成精简的因子分析报告

    Args:
        summary_df: 因子分析汇总数据框
        process_factors: 是否对因子进行了处理
        factor_method: 因子处理方法，'standardize'（标准化）或 'normalize'（归一化）
        winsorize: 是否进行了缩尾处理
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'因子分析详情_精简版_{timestamp}.txt'  # 修改文件名格式，生成精简版
    report_path = build_report_path(report_filename)

    # 使用新的分类函数对因子进行分类
    positive_factors, negative_factors = self.classify_factors_by_ic()

    # 生成因子分类概览
    classification_overview = self.generate_factor_classification_overview()

    # 生成各部分内容
    positive_analysis = self.generate_positive_factors_analysis(summary_mode=summary_mode)
    negative_analysis = self.generate_negative_factors_analysis(summary_mode=summary_mode)
    scoring_standards = self._get_scoring_standards()

    # 使用try-except块捕获可能的异常
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            # 报告标题
            f.write("=" * 80 + "\n")
            f.write("                    因子分析详细报告                   \n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {DEFAULT_DATA_FILE}\n")
            if summary_mode:
                f.write("（提示：已启用摘要模式，仅展示评分融合与稳健性摘要信息）\n")
                f.write("\n")
            else:
                f.write("\n")

            # 1. 因子分类概览
            f.write("1. 因子分类概览\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_overview)
            f.write("\n")

            # 2. 正向因子详细分析
            f.write("2. 正向因子详细分析\n")
            f.write("=" * 50 + "\n\n")
            f.write(positive_analysis)
            f.write("\n")

            # 3. 负向因子详细分析
            f.write("3. 负向因子详细分析\n")
            f.write("=" * 50 + "\n\n")
            f.write(negative_analysis)
            f.write("\n")

            # 4. 评分标准说明
            f.write("4. 评分标准说明\n")
            f.write("=" * 50 + "\n\n")
            f.write(scoring_standards)
            f.write("\n")

            # 显式刷新缓冲区
            f.flush()

        print(f"详细分析报告已生成: {report_path}")
        return report_path

    except Exception as e:
        print(f"生成报告时发生错误: {str(e)}")
        # 尝试重新生成一个简化版本
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("                    因子分析详细报告                   \n")
                f.write("=" * 80 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据文件: {DEFAULT_DATA_FILE}\n")
                f.write("\n")

                # 1. 因子分类概览
                f.write("1. 因子分类概览\n")
                f.write("=" * 50 + "\n\n")
                f.write(classification_overview)
                f.write("\n")

                # 2. 正向因子详细分析
                f.write("2. 正向因子详细分析\n")
                f.write("=" * 50 + "\n\n")
                f.write(positive_analysis)
                f.write("\n")

                # 3. 负向因子详细分析 - 简化版本
                f.write("3. 负向因子详细分析\n")
                f.write("=" * 50 + "\n\n")
                f.write("注意：由于内容过多，此处显示简化版本\n")
                f.write(f"负向因子总数: {len(negative_factors)}个\n")
                f.write("\n")

                # 4. 评分标准说明
                f.write("4. 评分标准说明\n")
                f.write("=" * 50 + "\n\n")
                f.write(scoring_standards)
                f.write("\n")

                # 显式刷新缓冲区
                f.flush()

            print(f"简化版详细分析报告已生成: {report_path}")
            return report_path
        except Exception as e2:
            print(f"生成简化报告时也发生错误: {str(e2)}")
            return None


def _fa_format_number(value, digits=3):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fa_format_percentage(value, digits=1):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        fmt = "{:." + str(digits) + "%}"
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return "N/A"


def _fa_format_count(value):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


def _fa_format_window_label(window):
    if isinstance(window, (int, np.integer)):
        return f"{int(window)}日"
    if isinstance(window, (float, np.floating)):
        if float(window).is_integer():
            return f"{int(window)}日"
        return f"{float(window):.0f}期"
    return str(window)


def _fa_format_sample_label(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        if 0 < value <= 1:
            return f"{int(round(value * 100))}%"
        return f"{float(value):.0f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _fa_sort_keys(keys):
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=lambda x: (isinstance(x, str), str(x)))


def _fa_get_auxiliary_entry(self, factor_name):
    aux_stats = getattr(self, 'auxiliary_stats', {}) or {}
    if factor_name in aux_stats:
        return aux_stats[factor_name]
    analysis_entry = getattr(self, 'analysis_results', {}).get(factor_name, {})
    return analysis_entry.get('auxiliary_stats')


def _fa_get_factor_extra_stats(self, factor_name):
    analysis_entry = getattr(self, 'analysis_results', {}).get(factor_name, {})
    extra = analysis_entry.get('extra_stats') or {}
    return extra


def _fa_format_ic_summary(self, factor_name):
    extra = _fa_get_factor_extra_stats(self, factor_name)
    if not extra:
        return "IC计算信息：无"
    daily_points = extra.get('daily_points')
    ic_mode = extra.get('ic_mode')
    avg_samples = extra.get('avg_daily_samples')
    total_dates = extra.get('total_dates')
    skip_ratio = extra.get('skip_ratio')
    min_samples = extra.get('min_samples_per_day')
    window_days = extra.get('ic_window_days')
    screen_mode = extra.get('screening_mode')
    median_samples = extra.get('median_daily_samples')
    fallback_reason = extra.get('fallback_reason') if ic_mode == "overall" else None
    sample_p25 = extra.get('p25_daily_samples')
    parts = []
    if daily_points is not None:
        if total_dates:
            parts.append(f"有效点数 {daily_points}/{total_dates}")
        else:
            parts.append(f"有效点数 {daily_points}")
    if ic_mode:
        if ic_mode == "overall":
            parts.append("模式 整体IC")
        else:
            parts.append(f"模式 {ic_mode}")
    if screen_mode:
        parts.append(screen_mode)
    if avg_samples is not None:
        parts.append(f"平均样本 {avg_samples:.1f}")
    if median_samples is not None:
        parts.append(f"中位 {median_samples:.1f}")
    if sample_p25 is not None:
        parts.append(f"P25 {sample_p25:.1f}")
    if min_samples is not None:
        parts.append(f"min样本 {min_samples}")
    if window_days is not None:
        parts.append(f"窗口 {window_days}日")
    if skip_ratio is not None:
        parts.append(f"缺样 {skip_ratio * 100:.1f}%")
    if fallback_reason:
        parts.append(fallback_reason)
    if not parts:
        return "IC计算信息：无"
    return "IC计算信息：" + "，".join(parts)


def _fa_format_segment_overview(self, factor_name):
    extra = _fa_get_factor_extra_stats(self, factor_name)
    if not extra:
        return None
    counts = extra.get('segment_counts') or {}
    if not counts:
        return None
    metrics = extra.get('segment_metrics') or {}
    segment_summary = extra.get('segment_summary') or {}
    recommendation = extra.get('segment_recommendation')
    segments = []
    for seg, info in counts.items():
        ratio = info.get('ratio')
        ratio_text = f"{ratio * 100:.1f}%" if ratio is not None else "N/A"
        count_text = info.get('count', 0)
        ic_value = metrics.get(seg, {}).get('overall_ic')
        ic_text = f"IC {ic_value:.3f}" if ic_value is not None and not pd.isna(ic_value) else "IC N/A"
        seg_parts = [f"{seg} {count_text}条({ratio_text}, {ic_text})"]
        summary_info = segment_summary.get(seg)
        if summary_info:
            ret = summary_info.get('avg_return')
            win_rate = summary_info.get('win_rate')
            summary_parts = []
            if ret is not None and not pd.isna(ret):
                summary_parts.append(f"收益 {ret:.3f}")
            if win_rate is not None and not pd.isna(win_rate):
                summary_parts.append(f"胜率 {win_rate * 100:.1f}%")
            if summary_parts:
                seg_parts.append("，".join(summary_parts))
        segments.append("；".join(seg_parts))
    if not segments:
        return None
    line = "板块覆盖：" + "，".join(segments)
    warnings = extra.get('segment_warnings') or []
    if warnings:
        line += "；提示：" + "；".join(warnings)
    if recommendation:
        line += f"；建议：{recommendation}"
    return line


_COMPONENT_DISPLAY_NAMES = {
    'base': "基础表现",
    'overall': "整体样本表现",
    'rolling': "滚动稳定度",
    'temporal': "时序一致性",
    'sample': "样本稳健性",
}


def _fa_get_factor_reliability(self, factor_name):
    analysis_entry = getattr(self, 'analysis_results', {}).get(factor_name, {})
    return analysis_entry.get('reliability') or {}


def _fa_format_reliability_summary(self, factor_name):
    reliability_entry = _fa_get_factor_reliability(self, factor_name)
    if not reliability_entry:
        return None
    labels = reliability_entry.get('labels') or {}
    weights = reliability_entry.get('weights') or {}
    parts = []
    for key, title in _COMPONENT_DISPLAY_NAMES.items():
        weight = weights.get(key)
        label = labels.get(key)
        if weight is None and not label:
            continue
        segment = []
        if weight is not None:
            segment.append(f"权重{weight * 100:.1f}%")
        if label:
            segment.append(f"可靠性{label}")
        if segment:
            parts.append(f"{title}: " + "，".join(segment))
    if not parts:
        return None
    note = reliability_entry.get('notes')
    summary = "；".join(parts)
    if note:
        summary += f"。{note}"
    return "评分权重调整：" + summary


def _fa_format_reliability_details(self, factor_name):
    extra = _fa_get_factor_extra_stats(self, factor_name)
    if not extra:
        return None
    parts = []
    avg_samples = extra.get('avg_daily_samples')
    if avg_samples is not None:
        parts.append(f"日均样本 {avg_samples:.1f}")
    sample_cv = extra.get('daily_sample_cv')
    if sample_cv is not None and not pd.isna(sample_cv):
        parts.append(f"样本CV {sample_cv:.2f}")
    skip_ratio = extra.get('skip_ratio')
    if skip_ratio is not None:
        parts.append(f"缺样 {skip_ratio * 100:.1f}%")
    qualified_ratio = extra.get('qualified_day_ratio')
    if qualified_ratio is not None:
        parts.append(f"有效日占比 {qualified_ratio * 100:.1f}%")
    top5_share = extra.get('daily_top5_share')
    if top5_share is not None and top5_share > 0:
        parts.append(f"Top5日期占比 {top5_share * 100:.1f}%")
    window_note = extra.get('ic_window_note') or extra.get('screening_mode')
    if window_note:
        parts.append(window_note)
    if not parts:
        return None
    return "数据可靠性评估：" + "，".join(parts)


def _fa_format_overall_reference(self, factor_name):
    extra = _fa_get_factor_extra_stats(self, factor_name)
    if not extra:
        return None
    metrics = extra.get('overall_metrics') or {}
    if not metrics:
        metrics = {
            key: extra.get(key)
            for key in extra.keys()
            if key.startswith('overall_')
        }
    if not metrics:
        return None
    ic_value = metrics.get('overall_ic')
    ir_value = metrics.get('overall_ir')
    p_value = metrics.get('overall_p_value')
    sample_size = metrics.get('overall_sample_size')
    mode = metrics.get('overall_mode')
    parts = []
    if ic_value is not None and not pd.isna(ic_value):
        parts.append(f"IC {ic_value:.4f}")
    if ir_value is not None and not pd.isna(ir_value):
        parts.append(f"IR {ir_value:.4f}")
    if p_value is not None and not pd.isna(p_value):
        parts.append(f"p值 {p_value:.4f}")
    if sample_size is not None and not pd.isna(sample_size):
        parts.append(f"样本 {int(round(float(sample_size)))}")
    if mode:
        parts.append(f"方法 {mode}")
    if not parts:
        return None
    return "整体表现参考：" + "，".join(parts)


def _fa_format_rolling_section(rolling_data):
    lines = []
    if not rolling_data:
        lines.append("  • 滚动IC表现：暂无数据")
        return lines

    lines.append("  • 滚动IC表现：")
    for window in _fa_sort_keys(rolling_data.keys()):
        payload = rolling_data.get(window) or {}
        stats = payload.get('stats') or {}
        stability = payload.get('stability') or {}
        decay = payload.get('decay') or {}
        mean_ic = _fa_format_number(stats.get('mean_ic'))
        cv = _fa_format_number(stability.get('coefficient_of_variation'), digits=2)
        maic = _fa_format_number(stability.get('mean_abs_ic'))
        half_life = _fa_format_number((decay or {}).get('half_life'), digits=1)
        sample_count = payload.get('sample_count')
        sample_text = f"{_fa_format_count(sample_count)}期" if sample_count else "N/A"
        lines.append(
            f"    · {_fa_format_window_label(window)}滚动: 均值{mean_ic} "
            f"(样本{sample_text}, CV {cv}, 半衰期 {half_life}, MAIC {maic})"
        )
    return lines


def _fa_format_temporal_section(temporal_data):
    lines = []
    if not temporal_data:
        lines.append("  • 时序一致性：暂无数据")
        return lines

    ic_stability = temporal_data.get('ic_stability') or {}
    trends = temporal_data.get('temporal_trends') or {}
    rank_stats = temporal_data.get('rank_stability') or {}
    lag1 = _fa_format_number(ic_stability.get('autocorr_lag1'), digits=2)
    trend_source = ic_stability.get('trend_correlation')
    if trend_source is None:
        trend_source = trends.get('trend_correlation')
    trend_corr = _fa_format_number(trend_source, digits=2)
    sign_changes = _fa_format_count(trends.get('sign_changes'))
    mean_reversion = _fa_format_number(trends.get('mean_reversion_strength'), digits=2)
    rank_vol = _fa_format_number(rank_stats.get('ranking_volatility'), digits=2)
    rank_change = _fa_format_number(rank_stats.get('mean_ranking_change'), digits=2)
    lines.append(
        "  • 时序一致性："
        f"Lag1自相关 {lag1}，趋势相关 {trend_corr}，换向次数 {sign_changes}，"
        f"均值回归度 {mean_reversion}，排名波动 {rank_vol}，平均名次变动 {rank_change}"
    )
    return lines


def _fa_format_sample_section(sample_data):
    lines = []
    effects = (sample_data or {}).get('sample_size_effects') or {}

    if not effects:
        lines.append("  • 样本敏感性：暂无数据")
    else:
        lines.append("  • 样本敏感性：")
        for size in _fa_sort_keys(effects.keys()):
            stats = effects.get(size) or {}
            mean_ic = _fa_format_number(stats.get('ic_mean'))
            std_ic = _fa_format_number(stats.get('ic_std'))
            q25 = stats.get('ic_q25')
            q75 = stats.get('ic_q75')
            iqr = None
            if q25 is not None and q75 is not None and not (pd.isna(q25) or pd.isna(q75)):
                iqr = q75 - q25
            iqr_text = _fa_format_number(iqr) if iqr is not None else "N/A"
            success_rate = _fa_format_percentage(stats.get('success_rate'))
            ratio_label = _fa_format_sample_label(size)
            lines.append(
                f"    · {ratio_label}样本: IC {mean_ic} ± {std_ic}, IQR {iqr_text}, 成功率 {success_rate}"
            )

    robustness = (sample_data or {}).get('robustness_metrics') or {}
    if robustness:
        variance = _fa_format_number(robustness.get('mean_variance_across_samples'), digits=4)
        best_ratio = _fa_format_sample_label(robustness.get('best_sample_size'))
        most_stable = _fa_format_sample_label(robustness.get('most_stable_sample_size'))
        lines.append(
            f"    · 跨样本稳健性: 方差 {variance}, 最佳样本 {best_ratio}, 稳定样本 {most_stable}"
        )
    return lines


def _fa_format_stability_details(self, factor_name):
    aux_entry = _fa_get_auxiliary_entry(self, factor_name)
    if not aux_entry:
        return ["稳健性分析：暂无辅助统计数据，建议重新运行辅助分析模块。"]

    lines = ["稳健性分析："]
    lines.extend(_fa_format_rolling_section(aux_entry.get('rolling')))
    lines.extend(_fa_format_temporal_section(aux_entry.get('temporal')))
    lines.extend(_fa_format_sample_section(aux_entry.get('sample')))
    return lines


def _fa_format_score_fusion_line(factor_row):
    if factor_row is None:
        return None
    _component_labels = [
        ('base', "基础表现（日度IC/IR）", '基础得分'),
        ('overall', "整体样本表现（整体IC/IR/P值）", '整体得分'),
        ('rolling', "滚动稳定度（CV/半衰期/绝对IC）", '滚动得分'),
        ('temporal', "时序一致性（Lag1/趋势/换向）", '时序得分'),
        ('sample', "样本稳健性（跨样本方差/成功率）", '样本得分'),
    ]
    final_score = factor_row.get('综合得分')
    rating = factor_row.get('评级')

    parts = []
    for key, label, field in _component_labels:
        value = factor_row.get(field)
        if value is None:
            continue
        try:
            if pd.notna(value):
                weight = factor_row.get(f'weight_{key}')
                reliability_label = factor_row.get(f'reliability_label_{key}')
                if weight is not None:
                    parts.append(
                        f"{label}（权重{weight * 100:.1f}%"
                        f"{'，可靠性'+reliability_label if reliability_label else ''}） {float(value):.1f}"
                    )
                else:
                    parts.append(f"{label} {float(value):.1f}")
        except Exception:
            continue
    if not parts:
        return None
    return f"评分融合：" + " / ".join(parts) + f" → 综合 {final_score:.1f} ({rating})"


def _fa_average_values(values):
    cleaned = []
    for val in values:
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        try:
            cleaned.append(float(val))
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _fa_format_stability_summary(self, factor_name):
    aux_entry = _fa_get_auxiliary_entry(self, factor_name)
    if not aux_entry:
        return "CV N/A，半衰期 N/A，Lag1 N/A，样本Std N/A，成功率 N/A"

    rolling = aux_entry.get('rolling') or {}
    cv_vals, half_vals, maic_vals = [], [], []
    for payload in rolling.values():
        stability = payload.get('stability') or {}
        decay = payload.get('decay') or {}
        cv_vals.append(stability.get('coefficient_of_variation'))
        maic_vals.append(stability.get('mean_abs_ic'))
        half_vals.append(decay.get('half_life'))
    cv_avg = _fa_average_values(cv_vals)
    half_avg = _fa_average_values(half_vals)
    maic_avg = _fa_average_values(maic_vals)

    temporal = aux_entry.get('temporal') or {}
    ic_stability = temporal.get('ic_stability') or {}
    trends = temporal.get('temporal_trends') or {}
    lag1 = ic_stability.get('autocorr_lag1')
    trend_corr = ic_stability.get('trend_correlation')
    if trend_corr is None:
        trend_corr = trends.get('trend_correlation')
    sign_changes = trends.get('sign_changes')

    sample_data = aux_entry.get('sample') or {}
    effects = sample_data.get('sample_size_effects') or {}
    sample_std_vals, sample_success_vals = [], []
    for stats in effects.values():
        if not stats:
            continue
        sample_std_vals.append(stats.get('ic_std'))
        sample_success_vals.append(stats.get('success_rate'))
    sample_std_avg = _fa_average_values(sample_std_vals)
    sample_success_avg = _fa_average_values(sample_success_vals)
    cross_variance = (sample_data.get('robustness_metrics') or {}).get('mean_variance_across_samples')

    parts = [
        f"CV { _fa_format_number(cv_avg, 2)}",
        f"半衰期 { _fa_format_number(half_avg, 1)}",
        f"MAIC { _fa_format_number(maic_avg, 3)}",
        f"Lag1 { _fa_format_number(lag1, 2)}",
        f"趋势 { _fa_format_number(trend_corr, 2)}",
        f"换向 { _fa_format_count(sign_changes)}",
        f"样本Std { _fa_format_number(sample_std_avg, 3)}",
        f"成功率 { _fa_format_percentage(sample_success_avg, 1)}",
        f"跨样本方差 { _fa_format_number(cross_variance, 4)}",
    ]
    return "，".join(parts)


def _fa_render_factor_summary(self, factors_df, is_positive=True):
    title = "正向因子摘要" if is_positive else "负向因子摘要（反向使用）"
    if len(factors_df) == 0:
        return "未发现可用于生成摘要的因子。"

    lines = []
    lines.append("=" * 80)
    lines.append(f"                     {title}")
    lines.append("=" * 80)
    for idx, (_, factor) in enumerate(factors_df.iterrows(), 1):
        win_rate = _fa_format_percentage(factor['胜率']) if '胜率' in factor and pd.notna(factor['胜率']) else "N/A"
        weight = self._get_suggested_weight(factor['评级'], is_positive)
        base_line = (
            f"{idx}. {factor['因子名称']} | 综合{factor['综合得分']:.2f} ({factor['评级']}) | "
            f"IC {factor['IC均值']:.4f} | IR {factor['IR值']:.4f} | 多空 {factor['多空收益']:.4f}"
        )
        lines.append(base_line)
        fusion_line = _fa_format_score_fusion_line(factor)
        if fusion_line:
            lines.append("   " + fusion_line)
        overall_line = _fa_format_overall_reference(self, factor['因子名称'])
        if overall_line:
            lines.append(f"   {overall_line}")
        stability_summary = _fa_format_stability_summary(self, factor['因子名称'])
        lines.append(f"   稳健性摘要: {stability_summary}")
        ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
        if ic_summary_line:
            lines.append(f"   {ic_summary_line}")
        segment_line = _fa_format_segment_overview(self, factor['因子名称'])
        if segment_line:
            lines.append(f"   {segment_line}")
        reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
        if reliability_summary:
            lines.append(f"   {reliability_summary}")
        reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
        if reliability_detail:
            lines.append(f"   {reliability_detail}")
        if not is_positive:
            lines.append(f"   反向IC: {abs(factor['IC均值']):.4f}")
        lines.append(f"   建议权重: {weight}；胜率 {win_rate}")
    return "\n".join(lines)


def _fa_generate_positive_factors_analysis(self, summary_mode=False):
    """
    生成正向因子详细分析报告

    Returns:
        str: 正向因子详细分析报告
    """
    # 获取分类因子数据
    positive_factors, negative_factors = self.classify_factors_by_ic()

    if len(positive_factors) == 0:
        return "未发现正向因子，无法生成详细分析。"

    # 构建正向因子分析报告
    analysis_lines = []

    if summary_mode:
        return _fa_render_factor_summary(self, positive_factors, is_positive=True)

    # 添加标题
    analysis_lines.append("=" * 80)
    analysis_lines.append("                     正向因子详细分析")
    analysis_lines.append("=" * 80)

    # 添加总体概况
    analysis_lines.append(f"正向因子总数: {len(positive_factors)}个")

    # 计算整体统计指标
    avg_ic = positive_factors['IC均值'].mean()
    avg_ir = positive_factors['IR值'].mean()
    avg_long_short = positive_factors['多空收益'].mean()
    total_factors = len(positive_factors)

    analysis_lines.append(f"平均IC均值: {avg_ic:.4f}")
    analysis_lines.append(f"平均IR值: {avg_ir:.4f}")
    analysis_lines.append(f"平均多空收益: {avg_long_short:.4f}")

    # 添加评级分布统计
    analysis_lines.append("\n评级分布:")
    rating_dist = positive_factors['评级'].value_counts()
    total_rating = len(positive_factors)

    # 只检查实际存在的评级
    for rating in ['A+级', 'A级', 'A-级', 'B+级', 'B级', 'C+级', 'C级', 'D级']:
        if rating in rating_dist.index:
            count = rating_dist[rating]
            percentage = (count / total_rating) * 100
            analysis_lines.append(f"  {rating}: {count}个 ({percentage:.1f}%)")

    # 如果没有找到任何评级，输出调试信息
    if len(rating_dist) == 0:
        analysis_lines.append("  无评级数据")
    elif all(rating not in rating_dist.index for rating in ['A+级', 'A级', 'A-级', 'B+级', 'B级', 'C+级', 'C级', 'D级']):
        analysis_lines.append("  评级数据格式异常")
        for rating, count in rating_dist.items():
            percentage = (count / total_rating) * 100
            analysis_lines.append(f"  {rating}: {count}个 ({percentage:.1f}%)")

    # 添加正向因子逐一分析（按综合得分排序）
    if len(positive_factors) > 0:
        sorted_positive = positive_factors.sort_values(['综合得分', 'IC均值'], ascending=[False, False])
        analysis_lines.append("\n正向因子逐个分析（按综合得分排序）:")
        for idx, (_, factor) in enumerate(sorted_positive.iterrows(), 1):
            analysis_lines.append(f"\n--- 正向因子 #{idx}: {factor['因子名称']} ---")
            analysis_lines.append(f"综合得分: {factor['综合得分']:.2f}")
            if pd.notna(factor.get('基础得分')):
                analysis_lines.append(f"基础表现得分: {factor['基础得分']:.1f}")
            if pd.notna(factor.get('整体得分')):
                analysis_lines.append(f"整体表现得分: {factor['整体得分']:.1f}")
            if pd.notna(factor.get('稳健得分')):
                analysis_lines.append(f"动态稳健性得分: {factor['稳健得分']:.1f}")
            if pd.notna(factor.get('滚动得分')):
                analysis_lines.append(f"  - 滚动稳定度: {factor['滚动得分']:.1f}")
            if pd.notna(factor.get('时序得分')):
                analysis_lines.append(f"  - 时序一致性: {factor['时序得分']:.1f}")
            if pd.notna(factor.get('样本得分')):
                analysis_lines.append(f"  - 样本稳健性: {factor['样本得分']:.1f}")
            analysis_lines.append(f"评级: {factor['评级']}")
            analysis_lines.append(f"建议权重: {self._get_suggested_weight(factor['评级'], True)}")
            analysis_lines.append(f"IC均值: {factor['IC均值']:.4f}")
            analysis_lines.append(f"IC标准差: {factor['IC标准差']:.4f}")
            analysis_lines.append(f"IR值: {factor['IR值']:.4f}")
            analysis_lines.append(f"多空收益: {factor['多空收益']:.4f}")
            analysis_lines.append(f"统计显著性: p值={factor['p值']:.4f}")

            win_rate = "N/A"
            if 'group_results' in self.analysis_results.get(factor['因子名称'], {}) and \
               self.analysis_results[factor['因子名称']]['group_results'] is not None:
                avg_returns = self.analysis_results[factor['因子名称']]['group_results']['avg_returns']
                if '胜率' in avg_returns.columns:
                    win_rate = f"{avg_returns['胜率'].mean():.2%}"
            analysis_lines.append(f"胜率: {win_rate}")

            overall_line = _fa_format_overall_reference(self, factor['因子名称'])
            if overall_line:
                analysis_lines.append(overall_line)
            ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
            if ic_summary_line:
                analysis_lines.append(ic_summary_line)
            segment_line = _fa_format_segment_overview(self, factor['因子名称'])
            if segment_line:
                analysis_lines.append(segment_line)
            reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
            if reliability_summary:
                analysis_lines.append(reliability_summary)
            reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
            if reliability_detail:
                analysis_lines.append(reliability_detail)

            performance_analysis = []
            if factor['IC均值'] > 0.02:
                performance_analysis.append("IC均值优秀，预测能力强")
            elif factor['IC均值'] > 0.01:
                performance_analysis.append("IC均值良好，预测能力适中")
            else:
                performance_analysis.append("IC均值一般，预测能力有待提升")

            if factor['IR值'] > 0.5:
                performance_analysis.append("风险调整后收益表现优秀")
            elif factor['IR值'] > 0.3:
                performance_analysis.append("风险调整后收益表现良好")
            else:
                performance_analysis.append("风险调整后收益表现一般")

            if factor['胜率'] > 0.6:
                performance_analysis.append("策略胜率较高")
            elif factor['胜率'] > 0.5:
                performance_analysis.append("策略胜率适中")
            else:
                performance_analysis.append("策略胜率偏低，需要优化")

            analysis_lines.append("综合评价: " + "；".join(performance_analysis))
            fusion_line = _fa_format_score_fusion_line(factor)
            if fusion_line:
                analysis_lines.append(fusion_line)
            analysis_lines.extend(_fa_format_stability_details(self, factor['因子名称']))

    # 添加策略建议
    analysis_lines.append("\n\n投资策略建议:")

    if len(positive_factors) > 0:
        # 获取前5个最佳因子
        top_5 = positive_factors.head(5)

        analysis_lines.append("\n1. 单因子策略:")
        for idx, (_, factor) in enumerate(top_5.iterrows(), 1):
            weight_range = self._get_suggested_weight(factor['评级'], True)
            analysis_lines.append(f"   因子{idx}: {factor['因子名称']}")
            analysis_lines.append(f"   - 评级: {factor['评级']}")
            analysis_lines.append(f"   - 综合得分: {factor['综合得分']:.2f}")
            analysis_lines.append(f"   - IC均值: {factor['IC均值']:.4f}")
            analysis_lines.append(f"   - IC标准差: {factor['IC标准差']:.4f}")
            analysis_lines.append(f"   - IR值: {factor['IR值']:.4f}")
            analysis_lines.append(f"   - 多空收益: {factor['多空收益']:.4f}")
            analysis_lines.append(f"   - 建议配置权重: {weight_range}")
            analysis_lines.append(f"   - 预期年化收益: {factor['IC均值']*12:.1%} (基于IC均值估算)")
            analysis_lines.append(f"   - 风险水平: {'低' if factor['IR值'] > 0.5 else '中' if factor['IR值'] > 0.3 else '高'}")

        analysis_lines.append("\n2. 多因子组合策略:")
        analysis_lines.append("   - 选择3-5个不同评级的正向因子组合")
        analysis_lines.append("   - 核心配置: A级因子，占比50-70%")
        analysis_lines.append("   - 辅助配置: B级因子，占比20-40%")
        analysis_lines.append("   - 分散配置: C级因子，占比10-20%")

        analysis_lines.append("\n3. 风险控制措施:")
        analysis_lines.append("   - 单一因子权重不超过25%")
        analysis_lines.append("   - 设置止损点: 单因子IC连续低于-0.01时考虑剔除")
        analysis_lines.append("   - 定期重新评估: 建议每月重新计算IC值")
        analysis_lines.append("   - 市值中性: 建议对市值进行中性化处理")

    # 添加数据质量评估
    analysis_lines.append("\n数据质量评估:")
    valid_factors = positive_factors.dropna()
    missing_data_pct = ((len(positive_factors) - len(valid_factors)) / len(positive_factors)) * 100

    analysis_lines.append(f"数据完整性: {100-missing_data_pct:.1f}%")
    analysis_lines.append(f"有效因子数: {len(valid_factors)}个")

    if missing_data_pct > 10:
        analysis_lines.append("警告: 部分因子数据缺失较多，可能影响分析结果的可靠性")

    # 添加市场环境适应性
    analysis_lines.append("\n市场环境适应性:")
    if avg_ic > 0.015:
        analysis_lines.append("当前市场环境对正向因子较为有利")
    elif avg_ic > 0.01:
        analysis_lines.append("当前市场环境对正向因子较为中性")
    else:
        analysis_lines.append("当前市场环境对正向因子不利，建议调整因子选择标准")

    return "\n".join(analysis_lines)


def _fa_generate_negative_factors_analysis(self, summary_mode=False):
    """
    生成负向因子详细分析报告

    Returns:
        str: 负向因子详细分析报告
    """
    # 获取分类因子数据
    positive_factors, negative_factors = self.classify_factors_by_ic()

    if len(negative_factors) == 0:
        return "未发现负向因子，无法生成详细分析。"

    # 构建负向因子分析报告
    analysis_lines = []

    if summary_mode:
        return _fa_render_factor_summary(self, negative_factors, is_positive=False)

    # 添加标题
    analysis_lines.append("=" * 80)
    analysis_lines.append("                     负向因子详细分析")
    analysis_lines.append("=" * 80)

    # 添加总体概况
    analysis_lines.append(f"负向因子总数: {len(negative_factors)}个")

    # 计算整体统计指标
    avg_ic = negative_factors['IC均值'].mean()
    avg_ir = negative_factors['IR值'].mean()
    avg_long_short = negative_factors['多空收益'].mean()

    analysis_lines.append(f"平均IC均值: {avg_ic:.4f}")
    analysis_lines.append(f"平均IR值: {avg_ir:.4f}")
    analysis_lines.append(f"平均多空收益: {avg_long_short:.4f}")

    # 添加评级分布统计
    analysis_lines.append("\n评级分布:")
    rating_dist = negative_factors['评级'].value_counts()
    total_rating = len(negative_factors)

    for rating in ['A+', 'A级', 'A-', 'B+', 'B级', 'B-', 'C+', 'C级', 'C-', 'D级']:
        if rating in rating_dist.index:
            count = rating_dist[rating]
            percentage = (count / total_rating) * 100
            analysis_lines.append(f"  {rating}: {count}个 ({percentage:.1f}%)")

    # 添加负向因子逐一分析（按综合得分排序）
    analysis_lines.append(f"\n负向因子逐个分析（按综合得分排序）:")
    sorted_negative = negative_factors.sort_values(['综合得分', 'IC均值'], ascending=[False, True])
    for idx, (_, factor) in enumerate(sorted_negative.iterrows(), 1):
        analysis_lines.append(f"\n--- 负向因子 #{idx}: {factor['因子名称']} ---")
        analysis_lines.append(f"综合得分: {factor['综合得分']:.2f}")
        if pd.notna(factor.get('基础得分')):
            analysis_lines.append(f"基础表现得分: {factor['基础得分']:.1f}")
        if pd.notna(factor.get('整体得分')):
            analysis_lines.append(f"整体表现得分: {factor['整体得分']:.1f}")
        if pd.notna(factor.get('稳健得分')):
            analysis_lines.append(f"动态稳健性得分: {factor['稳健得分']:.1f}")
        if pd.notna(factor.get('滚动得分')):
            analysis_lines.append(f"  - 滚动稳定度: {factor['滚动得分']:.1f}")
        if pd.notna(factor.get('时序得分')):
            analysis_lines.append(f"  - 时序一致性: {factor['时序得分']:.1f}")
        if pd.notna(factor.get('样本得分')):
            analysis_lines.append(f"  - 样本稳健性: {factor['样本得分']:.1f}")
        analysis_lines.append(f"评级: {factor['评级']}")
        analysis_lines.append(f"建议权重 (反向): {self._get_suggested_weight(factor['评级'], False)}")
        analysis_lines.append(f"IC均值: {factor['IC均值']:.4f} (原值)")
        analysis_lines.append(f"反向IC均值: {-factor['IC均值']:.4f}")
        analysis_lines.append(f"IC标准差: {factor['IC标准差']:.4f}")
        analysis_lines.append(f"IR值: {factor['IR值']:.4f}")
        analysis_lines.append(f"多空收益: {factor['多空收益']:.4f}")
        analysis_lines.append(f"胜率: {factor['胜率']:.2%}")
        analysis_lines.append(f"统计显著性: p值={factor['p值']:.4f}")
        overall_line = _fa_format_overall_reference(self, factor['因子名称'])
        if overall_line:
            analysis_lines.append(overall_line)
        ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
        if ic_summary_line:
            analysis_lines.append(ic_summary_line)
        segment_line = _fa_format_segment_overview(self, factor['因子名称'])
        if segment_line:
            analysis_lines.append(segment_line)
        reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
        if reliability_summary:
            analysis_lines.append(reliability_summary)
        reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
        if reliability_detail:
            analysis_lines.append(reliability_detail)

        # 分析因子表现
        performance_analysis = []
        if abs(factor['IC均值']) > 0.02:
            performance_analysis.append("反向IC绝对值优秀，预测能力强")
        elif abs(factor['IC均值']) > 0.01:
            performance_analysis.append("反向IC绝对值良好，预测能力适中")
        else:
            performance_analysis.append("反向IC绝对值一般，预测能力有待提升")

        if factor['IR值'] > 0.5:
            performance_analysis.append("风险调整后收益表现优秀")
        elif factor['IR值'] > 0.3:
            performance_analysis.append("风险调整后收益表现良好")
        else:
            performance_analysis.append("风险调整后收益表现一般")

        if factor['胜率'] > 0.6:
            performance_analysis.append("反向策略胜率较高")
        elif factor['胜率'] > 0.5:
            performance_analysis.append("反向策略胜率适中")
        else:
            performance_analysis.append("反向策略胜率偏低，需要优化")

        analysis_lines.append("综合评价: " + "；".join(performance_analysis))
        fusion_line = _fa_format_score_fusion_line(factor)
        if fusion_line:
            analysis_lines.append(fusion_line)
        analysis_lines.extend(_fa_format_stability_details(self, factor['因子名称']))

    # 添加反向策略建议
    analysis_lines.append("\n\n反向投资策略建议:")

    if len(negative_factors) > 0:
        # 获取前5个最佳负向因子（最负的IC值）
        top_5_negative = negative_factors.head(5)

        analysis_lines.append("\n1. 单因子反向策略:")
        for idx, (_, factor) in enumerate(top_5_negative.iterrows(), 1):
            weight_range = self._get_suggested_weight(factor['评级'], False)
            analysis_lines.append(f"   因子{idx}: {factor['因子名称']} (反向)")
            analysis_lines.append(f"   - 评级: {factor['评级']}")
            analysis_lines.append(f"   - 综合得分: {factor['综合得分']:.2f}")
            analysis_lines.append(f"   - IC均值: {factor['IC均值']:.4f} (原值)")
            analysis_lines.append(f"   - IC标准差: {factor['IC标准差']:.4f}")
            analysis_lines.append(f"   - IR值: {factor['IR值']:.4f}")
            analysis_lines.append(f"   - 多空收益: {factor['多空收益']:.4f}")
            analysis_lines.append(f"   - 建议配置权重: {weight_range}")
            analysis_lines.append(f"   - 预期年化收益: {abs(factor['IC均值'])*12:.1%} (基于反向IC值估算)")
            analysis_lines.append(f"   - 风险水平: {'低' if factor['IR值'] > 0.5 else '中' if factor['IR值'] > 0.3 else '高'}")

        analysis_lines.append("\n2. 反向因子组合策略:")
        analysis_lines.append("   - 选择2-3个不同评级的负向因子组合")
        analysis_lines.append("   - 核心配置: A级负向因子，占比30-50%")
        analysis_lines.append("   - 辅助配置: B级负向因子，占比20-30%")
        analysis_lines.append("   - 风险分散: C级负向因子，占比10-20%")
        analysis_lines.append("   - 注意: 负向因子通常用于风险对冲，不宜配置过高权重")

        analysis_lines.append("\n3. 反向策略风险控制:")
        analysis_lines.append("   - 单一反向因子权重不超过20%")
        analysis_lines.append("   - 设置止损点: 反向IC连续高于0.01时考虑剔除")
        analysis_lines.append("   - 定期重新评估: 建议每两周重新计算IC值")
        analysis_lines.append("   - 组合使用: 与正向因子结合使用实现风险对冲")

    # 添加对冲策略建议
    analysis_lines.append("\n对冲策略建议:")
    analysis_lines.append("\n1. 市场中性对冲:")
    analysis_lines.append("   - 正向因子: 50-70%")
    analysis_lines.append("   - 负向因子: 30-50%")
    analysis_lines.append("   - 目标: 降低组合整体波动性")

    analysis_lines.append("\n2. 行业中性对冲:")
    analysis_lines.append("   - 正向因子: 40-60%")
    analysis_lines.append("   - 负向因子: 40-60%")
    analysis_lines.append("   - 目标: 消除行业偏好风险")

    analysis_lines.append("\n3. 时间中性对冲:")
    analysis_lines.append("   - 正向因子: 60-80%")
    analysis_lines.append("   - 负向因子: 20-40%")
    analysis_lines.append("   - 目标: 在特定时间窗口内捕捉机会")

    return "\n".join(analysis_lines)

