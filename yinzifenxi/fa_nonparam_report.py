# -*- coding: utf-8 -*-
"""
非参数因子分析报告模块，当前版本从 yinzifenxi1119_split.py 中拆分而来。

本模块**已**在 yinzifenxi1119_split.py 中被调用，负责承载所有报告生成逻辑，
以便 FactorAnalysis 主类可以逐步瘦身。所有函数保持与原脚本一致的行为。
"""

import numpy as np
import pandas as pd
from datetime import datetime
from html import escape

from .fa_config import DEFAULT_DATA_FILE, DEFAULT_DATA_FILES, build_report_path
from .fa_report_utils import (
    HTMLReportBuilder,
    render_metric_cards,
    render_alert,
    render_text_block,
    render_table,
    render_report_notes,
)

DATA_FILE_LABEL = (
    ", ".join(DEFAULT_DATA_FILES)
    if DEFAULT_DATA_FILES
    else (DEFAULT_DATA_FILE or "（未配置数据文件）")
)


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
            'p值': results['p_value'],
            '原始IC均值': results.get('raw_ic_mean'),
            '原始IC标准差': results.get('raw_ic_std'),
            '原始IR值': results.get('raw_ir'),
            '原始t统计量': results.get('raw_t_stat'),
            '原始p值': results.get('raw_p_value'),
        }

        if 'group_results' in results and results['group_results'] is not None:
            row['多空收益'] = results['group_results']['long_short_return']

        extra_stats = results.get('extra_stats') or {}
        raw_extra_stats = results.get('raw_extra_stats') or {}
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
        if raw_extra_stats:
            row['原始IC有效点数'] = raw_extra_stats.get('daily_points')
            row['原始IC总日数'] = raw_extra_stats.get('total_dates')
            row['原始IC缺样比例'] = raw_extra_stats.get('skip_ratio')
            row['原始IC模式'] = raw_extra_stats.get('ic_mode')
            row['原始筛选模式'] = raw_extra_stats.get('screening_mode')

        # 检查缺失值
        if np.isnan(results['ic_std']) or np.isnan(results['ir']):
            missing_data_count += 1
            print(f"警告: 因子 '{factor}' 存在缺失数据 - IC标准差: {results['ic_std']}, IR值: {results['ir']}")

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # 创建显示用的数据框，将缺失值替换为'N/A'
    display_df = summary_df.copy()
    float_cols = ['IC均值', 'IC标准差', 'IR值', 't统计量', 'p值', '多空收益',
                  '原始IC均值', '原始IC标准差', '原始IR值', '原始t统计量', '原始p值',
                  '日均样本', '样本中位数', '样本P25', '主要板块IC', '次要板块IC']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: 'N/A' if pd.isna(x) else f"{x:.3f}")

    percent_cols = ['IC缺样比例', '原始IC缺样比例', '主要板块占比', '次要板块占比']
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: 'N/A' if pd.isna(x) else f"{float(x)*100:.1f}%"
            )

    int_cols = ['IC有效点数', 'IC总日数', '窗口天数', '日最小样本', '整体样本量',
                '原始IC有效点数', '原始IC总日数']
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
    filename = f'单因子分析汇总_{timestamp}.csv'  # 修改文件名格式，避免生成'因子分析报告_当日回调_时间戳.csv'
    file_path = build_report_path(filename)
    summary_df_rounded.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总报告已保存到 '{file_path}'")

    return summary_df


def _fa_generate_factor_analysis_report(self, summary_df, process_factors=False, factor_method='standardize', winsorize=False, summary_mode=False):
    """
    生成精简版因子分析报告

    Args:
        summary_df: 因子分析汇总数据
        process_factors: 是否对因子做预处理
        factor_method: 预处理方式（standardize/normalize）
        winsorize: 是否进行缩尾
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'单因子分析_{timestamp}.html'
    report_path = build_report_path(report_filename)

    builder = HTMLReportBuilder("因子分析详细报告", "摘要模式" if summary_mode else "完整报告")
    builder.set_meta(
        [
            ('生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            ('数据文件', DATA_FILE_LABEL),
            ('分析模式', '摘要' if summary_mode else '完整'),
        ]
    )

    try:
        positive_factors, negative_factors = self.classify_factors_by_ic()
    except Exception as exc:
        print(f"[WARN] 因子分类失败: {exc}")
        positive_factors = negative_factors = pd.DataFrame(), pd.DataFrame()

    def _safe_section(generator, fallback, warn_level="warn"):
        try:
            return generator()
        except Exception as section_exc:
            print(f"[WARN] {fallback} 生成失败: {section_exc}")
            return f"{fallback}生成失败：{section_exc}"

    classification_overview = _safe_section(
        lambda: self.generate_factor_classification_overview(),
        "因子分类概览"
    )
    positive_analysis = _safe_section(
        lambda: self.generate_positive_factors_analysis(summary_mode=summary_mode),
        "正向因子分析"
    )
    negative_analysis = _safe_section(
        lambda: self.generate_negative_factors_analysis(summary_mode=summary_mode),
        "负向因子分析"
    )
    scoring_standards = _safe_section(self._get_scoring_standards, "评分标准说明")

    if summary_df is not None and not summary_df.empty:
        cards_html = render_metric_cards(
            [
                ("分析因子数", str(len(summary_df))),
                ("平均 IC", _fa_format_number(summary_df['IC均值'].mean(), 3)),
                ("平均原始 IC", _fa_format_number(summary_df['原始IC均值'].mean(), 3)),
                ("平均 IR", _fa_format_number(summary_df['IR值'].mean(), 3)),
                ("多空收益中位数", _fa_format_number(summary_df['多空收益'].median(), 3)),
            ]
        )
        builder.add_section("核心指标概览", cards_html)
    else:
        builder.add_section("核心指标概览", render_alert("暂无汇总数据，无法展示整体指标。", level="warn"))

    builder.add_section("因子分类概览", render_text_block(classification_overview))

    def _render_factor_section(factors_df, description, is_positive):
        if factors_df is None or factors_df.empty:
            return render_alert("暂无相关因子数据。", level="warn")
        avg_ic = _fa_format_number(factors_df['IC均值'].mean(), 3)
        avg_raw_ic = _fa_format_number(factors_df['原始IC均值'].mean(), 3)
        avg_ir = _fa_format_number(factors_df['IR值'].mean(), 3)
        avg_lr = _fa_format_number(factors_df['多空收益'].mean(), 3)
        cards = render_metric_cards(
            [
                ("因子数量", str(len(factors_df))),
                ("平均IC", avg_ic),
                ("平均原始IC", avg_raw_ic),
                ("平均IR", avg_ir),
                ("平均多空收益", avg_lr),
            ]
        )
        top_df = factors_df.sort_values(['综合得分', 'IC均值'], ascending=[False, False]).head(3)
        highlight_html = ""
        if not top_df.empty:
            items = []
            for idx, (_, row) in enumerate(top_df.iterrows(), 1):
                badge_class = "tag-positive" if is_positive else "tag-negative"
                items.append(
                    f"<div class='highlight-card'>"
                    f"<h4>TOP{idx} · {escape(row['因子名称'])}</h4>"
                    f"<p class='chip {badge_class}'>综合得分 { _fa_format_number(row['综合得分'], 2) }</p>"
                    f"<p class='muted'>IC { _fa_format_number(row['IC均值'], 3) } / 原始IC { _fa_format_number(row['原始IC均值'], 3) } · IR { _fa_format_number(row['IR值'],3) } · 多空收益 { _fa_format_number(row['多空收益'],3) }</p>"
                    f"</div>"
                )
            highlight_html = "<div class='grid-two'>" + "".join(items) + "</div>"
        else:
            highlight_html = render_alert("暂无排名信息。", level="warn")

        display_df = factors_df[['因子名称', '综合得分', 'IC均值', '原始IC均值', 'IR值', '多空收益', '评级']].copy().head(10)
        table_html = render_table(
            display_df,
            ['因子名称', '综合得分', 'IC均值', '原始IC均值', 'IR值', '多空收益', '评级'],
            headers=['因子', '综合得分', 'IC', '原始IC', 'IR', '多空收益', '评级'],
            formatters={
                '综合得分': lambda x: _fa_format_number(x, 2),
                'IC均值': lambda x: _fa_format_number(x, 3),
                '原始IC均值': lambda x: _fa_format_number(x, 3),
                'IR值': lambda x: _fa_format_number(x, 3),
                '多空收益': lambda x: _fa_format_number(x, 3),
            },
        )

        return (
            f"<div class='factor-section'>"
            f"<p class='section-desc'>{escape(description)}</p>"
            f"{cards}"
            f"{highlight_html}"
            f"{table_html}"
            f"</div>"
        )

    builder.add_section(
        "正向因子详细分析",
        _render_factor_section(positive_factors, "兼顾收益与稳定性的多头信号，重点关注高分因子的权重配置。", True)
        + f"<div class='details-block'><details><summary>查看文字版诊断</summary>{render_text_block(positive_analysis)}</details></div>"
    )
    builder.add_section(
        "负向因子详细分析",
        _render_factor_section(negative_factors, "用于对冲或反向交易的信号，关注反向胜率与风险暴露。", False)
        + f"<div class='details-block'><details><summary>查看文字版诊断</summary>{render_text_block(negative_analysis)}</details></div>"
    )
    builder.add_section("评分标准说明", render_text_block(scoring_standards))

    builder.add_section("报告说明", render_report_notes())

    html_content = builder.render()
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"详细分析报告已生成: {report_path}")
        return report_path
    except Exception as e:
        print(f"生成报告时发生错误: {str(e)}")
        try:
            fallback = f"""<!DOCTYPE html>
            <html lang="zh-CN"><head><meta charset="utf-8"><title>因子分析报告失败</title></head>
            <body><p>报告生成失败：{escape(str(e))}</p></body></html>"""
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(fallback)
            print(f"[WARN] 已写入降级版报告: {report_path}")
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
        lines.append("  -  滚动IC表现：暂无数据")
        return lines

    lines.append("  -  滚动IC表现：")
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
        lines.append("  -  时序一致性：暂无数据")
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
        "  -  时序一致性："
        f"Lag1自相关 {lag1}，趋势相关 {trend_corr}，换向次数 {sign_changes}，"
        f"均值回归度 {mean_reversion}，排名波动 {rank_vol}，平均名次变动 {rank_change}"
    )
    return lines


def _fa_format_sample_section(sample_data):
    lines = []
    effects = (sample_data or {}).get('sample_size_effects') or {}

    if not effects:
        lines.append("  -  样本敏感性：暂无数据")
    else:
        lines.append("  -  样本敏感性：")
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
        return "> 未发现可用于生成摘要的因子。\n"

    lines = [f"**{title}**", ""]
    for idx, (_, factor) in enumerate(factors_df.iterrows(), 1):
        win_rate = _fa_format_percentage(factor['胜率']) if '胜率' in factor and pd.notna(factor['胜率']) else "N/A"
        weight = self._get_suggested_weight(factor['评级'], is_positive)
        lines.append(f"{idx}. **{factor['因子名称']}** — 综合 {factor['综合得分']:.2f}（{factor['评级']}）")
        details = [
            f"IC {factor['IC均值']:.4f}",
            f"IR {factor['IR值']:.4f}",
            f"多空收益 {factor['多空收益']:.4f}",
            f"胜率 {win_rate}",
            f"建议权重 {weight}",
        ]
        for detail in details:
            lines.append(f"   - {detail}")

        fusion_line = _fa_format_score_fusion_line(factor)
        if fusion_line:
            lines.append(f"   - {fusion_line}")
        overall_line = _fa_format_overall_reference(self, factor['因子名称'])
        if overall_line:
            lines.append(f"   - {overall_line}")
        stability_summary = _fa_format_stability_summary(self, factor['因子名称'])
        lines.append(f"   - 稳健性摘要：{stability_summary}")
        ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
        if ic_summary_line:
            lines.append(f"   - {ic_summary_line}")
        segment_line = _fa_format_segment_overview(self, factor['因子名称'])
        if segment_line:
            lines.append(f"   - {segment_line}")
        reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
        if reliability_summary:
            lines.append(f"   - {reliability_summary}")
        reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
        if reliability_detail:
            lines.append(f"   - {reliability_detail}")
        if not is_positive:
            lines.append(f"   - 反向 IC：{abs(factor['IC均值']):.4f}")
        lines.append("")
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
        return "> 未发现正向因子，无法生成详细分析。\n"

    analysis_lines = []
    if summary_mode:
        return _fa_render_factor_summary(self, positive_factors, is_positive=True)

    avg_ic = positive_factors['IC均值'].mean()
    avg_ir = positive_factors['IR值'].mean()
    avg_long_short = positive_factors['多空收益'].mean()

    analysis_lines.append("**正向因子总体概况**")
    analysis_lines.append(f"- 正向因子总数：{len(positive_factors)} 个")
    analysis_lines.append(f"- 平均 IC 均值：{avg_ic:.4f}")
    analysis_lines.append(f"- 平均 IR：{avg_ir:.4f}")
    analysis_lines.append(f"- 平均多空收益：{avg_long_short:.4f}")

    rating_dist = positive_factors['评级'].value_counts()
    analysis_lines.append("\n**评级分布**")
    if rating_dist.empty:
        analysis_lines.append("- 暂无评级数据")
    else:
        total_rating = len(positive_factors)
        for rating in ['A+级', 'A级', 'A-级', 'B+级', 'B级', 'C+级', 'C级', 'D级']:
            if rating in rating_dist.index:
                count = rating_dist[rating]
                percentage = (count / total_rating) * 100
                analysis_lines.append(f"- {rating}: {count} 个 ({percentage:.1f}%)")

    analysis_lines.append("\n**逐因子分析（按综合得分排序）**")
    sorted_positive = positive_factors.sort_values(['综合得分', 'IC均值'], ascending=[False, False])
    for idx, (_, factor) in enumerate(sorted_positive.iterrows(), 1):
        analysis_lines.append(f"\n#### 正向因子 #{idx}: {factor['因子名称']}")
        analysis_lines.append(f"- 综合得分：{factor['综合得分']:.2f}（{factor['评级']}）")
        analysis_lines.append(f"- 建议权重：{self._get_suggested_weight(factor['评级'], True)}")
        score_details = []
        if pd.notna(factor.get('基础得分')):
            score_details.append(f"基础得分 {factor['基础得分']:.1f}")
        if pd.notna(factor.get('整体得分')):
            score_details.append(f"整体得分 {factor['整体得分']:.1f}")
        if pd.notna(factor.get('稳健得分')):
            score_details.append(f"稳健性 {factor['稳健得分']:.1f}")
        if score_details:
            analysis_lines.append(f"- 评分拆解：{' / '.join(score_details)}")
        raw_ic_val = factor.get('原始IC均值')
        raw_ic_text = f"，原始IC：{raw_ic_val:.4f}" if raw_ic_val is not None and not pd.isna(raw_ic_val) else ""
        analysis_lines.append(f"- IC 均值：{factor['IC均值']:.4f}，IC 标准差：{factor['IC标准差']:.4f}{raw_ic_text}")
        analysis_lines.append(f"- IR 值：{factor['IR值']:.4f}，多空收益：{factor['多空收益']:.4f}")
        analysis_lines.append(f"- p 值：{factor['p值']:.4f}")

        win_rate = "N/A"
        if 'group_results' in self.analysis_results.get(factor['因子名称'], {}) and \
           self.analysis_results[factor['因子名称']]['group_results'] is not None:
            avg_returns = self.analysis_results[factor['因子名称']]['group_results']['avg_returns']
            if '胜率' in avg_returns.columns:
                win_rate = f"{avg_returns['胜率'].mean():.2%}"
        analysis_lines.append(f"- 胜率：{win_rate}")

        overall_line = _fa_format_overall_reference(self, factor['因子名称'])
        if overall_line:
            analysis_lines.append(f"- {overall_line}")
        ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
        if ic_summary_line:
            analysis_lines.append(f"- {ic_summary_line}")
        segment_line = _fa_format_segment_overview(self, factor['因子名称'])
        if segment_line:
            analysis_lines.append(f"- {segment_line}")
        reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
        if reliability_summary:
            analysis_lines.append(f"- {reliability_summary}")
        reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
        if reliability_detail:
            analysis_lines.append(f"- {reliability_detail}")

        performance_analysis = []
        if factor['IC均值'] > 0.02:
            performance_analysis.append("IC 表现优秀")
        elif factor['IC均值'] > 0.01:
            performance_analysis.append("IC 表现良好")
        else:
            performance_analysis.append("IC 表现一般")

        if factor['IR值'] > 0.5:
            performance_analysis.append("风险调整收益优秀")
        elif factor['IR值'] > 0.3:
            performance_analysis.append("风险调整收益良好")
        else:
            performance_analysis.append("风险调整收益一般")

        if factor['胜率'] > 0.6:
            performance_analysis.append("胜率较高")
        elif factor['胜率'] > 0.5:
            performance_analysis.append("胜率适中")
        else:
            performance_analysis.append("胜率偏低")

        analysis_lines.append("**综合评价**：" + "；".join(performance_analysis))
        fusion_line = _fa_format_score_fusion_line(factor)
        if fusion_line:
            analysis_lines.append(f"- {fusion_line}")
        analysis_lines.extend(_fa_format_stability_details(self, factor['因子名称']))

    analysis_lines.append("\n**投资策略建议**")
    top_5 = positive_factors.head(5)
    if len(top_5) > 0:
        analysis_lines.append("1. **单因子策略**")
        for idx, (_, factor) in enumerate(top_5.iterrows(), 1):
            weight_range = self._get_suggested_weight(factor['评级'], True)
            analysis_lines.append(f"   - 因子{idx}: **{factor['因子名称']}** ({factor['评级']})，建议权重 {weight_range}；IC {factor['IC均值']:.4f}，IR {factor['IR值']:.4f}")
        analysis_lines.append("\n2. **多因子组合策略**")
        analysis_lines.append("   - 选择 3~5 个不同风格的正向因子组合，核心 A 级权重 50%~70%。")
        analysis_lines.append("   - 每月/季按综合得分与风险参数动态调权，避免单因子暴露。")
        analysis_lines.append("\n3. **风险控制**")
        analysis_lines.append("   - 单因子权重上限 25%，IC 连续走弱及时剔除。")
        analysis_lines.append("   - 建议结合成交量、市值等维度做中性化处理，提升稳健度。")

    valid_factors = positive_factors.dropna()
    missing_data_pct = ((len(positive_factors) - len(valid_factors)) / len(positive_factors)) * 100
    analysis_lines.append("\n**数据质量与市场环境**")
    analysis_lines.append(f"- 数据完整性：{100 - missing_data_pct:.1f}%（有效因子 {len(valid_factors)} 个）")
    if missing_data_pct > 10:
        analysis_lines.append("- 警告：部分因子缺失较多，分析结果可能存在偏差。")
    if avg_ic > 0.015:
        analysis_lines.append("- 当前市场对正向因子友好，可适度加权。")
    elif avg_ic > 0.01:
        analysis_lines.append("- 市场环境中性，建议保持均衡配置。")
    else:
        analysis_lines.append("- 市场环境偏弱，务必控制仓位并加强回测验证。")

    return "\n".join(analysis_lines)


def _fa_generate_negative_factors_analysis(self, summary_mode=False):
    """
    生成负向因子详细分析报告
    """
    positive_factors, negative_factors = self.classify_factors_by_ic()

    if len(negative_factors) == 0:
        return "> 未发现负向因子，无法生成详细分析。"

    if summary_mode:
        return _fa_render_factor_summary(self, negative_factors, is_positive=False)

    analysis_lines = []
    analysis_lines.append("**负向因子总体概况**")
    analysis_lines.append(f"- 负向因子总数：{len(negative_factors)} 个")
    analysis_lines.append(f"- 平均反向 IC：{abs(negative_factors['IC均值'].mean()):.4f}")
    analysis_lines.append(f"- 平均 IR：{negative_factors['IR值'].mean():.4f}")
    analysis_lines.append(f"- 平均多空收益：{negative_factors['多空收益'].mean():.4f}")

    analysis_lines.append("\n**评级分布**")
    rating_dist = negative_factors['评级'].value_counts()
    if rating_dist.empty:
        analysis_lines.append("- 暂无评级数据")
    else:
        total_rating = len(negative_factors)
        for rating in ['A+级', 'A级', 'A-级', 'B+级', 'B级', 'C+级', 'C级', 'D级']:
            if rating in rating_dist.index:
                count = rating_dist[rating]
                percentage = (count / total_rating) * 100
                analysis_lines.append(f"- {rating}: {count} 个 ({percentage:.1f}%)")

    analysis_lines.append("\n**逐因子分析（按综合得分排序）**")
    sorted_negative = negative_factors.sort_values(['综合得分', 'IC均值'], ascending=[False, True])
    for idx, (_, factor) in enumerate(sorted_negative.iterrows(), 1):
        analysis_lines.append(f"\n#### 负向因子 #{idx}: {factor['因子名称']}")
        analysis_lines.append(f"- 综合得分：{factor['综合得分']:.2f}（{factor['评级']}）")
        analysis_lines.append(f"- 建议反向权重：{self._get_suggested_weight(factor['评级'], False)}")
        score_details = []
        if pd.notna(factor.get('基础得分')):
            score_details.append(f"基础得分 {factor['基础得分']:.1f}")
        if pd.notna(factor.get('整体得分')):
            score_details.append(f"整体得分 {factor['整体得分']:.1f}")
        if pd.notna(factor.get('稳健得分')):
            score_details.append(f"稳健性 {factor['稳健得分']:.1f}")
        if score_details:
            analysis_lines.append(f"- 评分拆解：{' / '.join(score_details)}")
        raw_ic_val = factor.get('原始IC均值')
        raw_ic_text = f"，原始IC：{abs(raw_ic_val):.4f}" if raw_ic_val is not None and not pd.isna(raw_ic_val) else ""
        analysis_lines.append(f"- 反向 IC：{abs(factor['IC均值']):.4f}，IC 标准差：{factor['IC标准差']:.4f}{raw_ic_text}")
        analysis_lines.append(f"- IR 值：{factor['IR值']:.4f}，多空收益：{factor['多空收益']:.4f}")
        analysis_lines.append(f"- p 值：{factor['p值']:.4f}")

        win_rate = "N/A"
        factor_result = self.analysis_results.get(factor['因子名称'], {})
        group_results = factor_result.get('group_results')
        if group_results and 'avg_returns' in group_results:
            avg_returns = group_results['avg_returns']
            if '胜率' in avg_returns.columns:
                win_rate = f"{avg_returns['胜率'].mean():.2%}"
        analysis_lines.append(f"- 反向胜率：{win_rate}")

        overall_line = _fa_format_overall_reference(self, factor['因子名称'])
        if overall_line:
            analysis_lines.append(f"- {overall_line}")
        ic_summary_line = _fa_format_ic_summary(self, factor['因子名称'])
        if ic_summary_line:
            analysis_lines.append(f"- {ic_summary_line}")
        segment_line = _fa_format_segment_overview(self, factor['因子名称'])
        if segment_line:
            analysis_lines.append(f"- {segment_line}")
        reliability_summary = _fa_format_reliability_summary(self, factor['因子名称'])
        if reliability_summary:
            analysis_lines.append(f"- {reliability_summary}")
        reliability_detail = _fa_format_reliability_details(self, factor['因子名称'])
        if reliability_detail:
            analysis_lines.append(f"- {reliability_detail}")

        performance_analysis = []
        abs_ic = abs(factor['IC均值'])
        if abs_ic > 0.02:
            performance_analysis.append("反向 IC 绝对值优秀")
        elif abs_ic > 0.01:
            performance_analysis.append("反向 IC 绝对值良好")
        else:
            performance_analysis.append("反向 IC 表现一般")

        if factor['IR值'] < -0.5:
            performance_analysis.append("风险调整收益优秀（反向）")
        elif factor['IR值'] < -0.3:
            performance_analysis.append("风险调整收益良好（反向）")
        else:
            performance_analysis.append("风险调整收益一般")

        if factor['胜率'] < 0.4:
            performance_analysis.append("反向胜率较高")
        elif factor['胜率'] < 0.5:
            performance_analysis.append("反向胜率适中")
        else:
            performance_analysis.append("反向胜率偏低")

        analysis_lines.append("**综合评价**：" + "；".join(performance_analysis))
        fusion_line = _fa_format_score_fusion_line(factor)
        if fusion_line:
            analysis_lines.append(f"- {fusion_line}")
        analysis_lines.extend(_fa_format_stability_details(self, factor['因子名称']))

    analysis_lines.append("\n**策略建议**")
    top_5 = negative_factors.head(5)
    if len(top_5) > 0:
        analysis_lines.append("1. **反向对冲策略**")
        for idx, (_, factor) in enumerate(top_5.iterrows(), 1):
            weight_range = self._get_suggested_weight(factor['评级'], False)
            analysis_lines.append(
                f"   - 因子{idx}: **{factor['因子名称']}** ({factor['评级']})，建议权重 {weight_range}，反向 IC {abs(factor['IC均值']):.4f}"
            )
        analysis_lines.append("\n2. **对冲组合**")
        analysis_lines.append("   - 选择 2~3 个稳定的负向因子与正向策略搭配，形成 1:2 或 1:3 的风险对冲。")
        analysis_lines.append("   - 负向因子总权重建议控制在组合 30% 以内。")
        analysis_lines.append("\n3. **使用注意事项**")
        analysis_lines.append("   - 在市场走弱或单一风格过热时启用，平稳期谨慎使用。")
        analysis_lines.append("   - 反向 IC 绝对值跌破 0.01 或胜率>50% 视为失效信号，需暂停。")

    valid_factors = negative_factors.dropna()
    missing_data_pct = ((len(negative_factors) - len(valid_factors)) / len(negative_factors)) * 100
    analysis_lines.append("\n**数据质量与市场环境**")
    analysis_lines.append(f"- 数据完整性：{100 - missing_data_pct:.1f}%（有效因子 {len(valid_factors)} 个）")
    if missing_data_pct > 10:
        analysis_lines.append("- 警告：负向因子缺失较多，需谨慎解读。")
    avg_ic = negative_factors['IC均值'].mean()
    if avg_ic < -0.015:
        analysis_lines.append("- 当前市场对负向策略友好，可适度加仓。")
    elif avg_ic < -0.01:
        analysis_lines.append("- 市场环境中性，优先作为风险对冲。")
    else:
        analysis_lines.append("- 市场对负向策略不利，仅保留最有把握的信号。")

    return "\n".join(analysis_lines)

