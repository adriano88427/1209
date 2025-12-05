# -*- coding: utf-8 -*-
"""
Non-parametric FactorAnalysis helper functions.

These helpers encapsulate classification/权重建议等逻辑，以便在拆分过程中复用。
"""

import numpy as np
import pandas as pd

from .fa_config import RELIABILITY_CONFIG, SEGMENT_MIN_SAMPLES
from .fa_stat_utils import bounded_normalize


def _fa_get_suggested_weight(self, rating, is_positive):
    """
    根据因子评级获取建议权重
    """
    if is_positive:
        if "A+" in rating or "A级" in rating:
            return "15%-25%"
        if "A-" in rating or "B+" in rating:
            return "10%-15%"
        if "B级" in rating or "C+" in rating:
            return "5%-10%"
        if "C级" in rating:
            return "3%-5%"
        return "0%-3%"
    if "A-" in rating:
        return "-10%至-20%"
    if "B+" in rating or "B级" in rating:
        return "-5%至-10%"
    if "C+" in rating or "C级" in rating:
        return "-3%至-5%"
    return "0%至-3%"


def _fa_get_scoring_standards(self):
    """
    获取因子评分标准的详细说明
    """
    standards = []
    standards.append("本报告采用“分维度评分 + 动态可靠性加权”方案：")
    standards.append("")
    standards.append("1. **基础表现（日度 IC / IR / 显著性）**")
    standards.append("   - IC：|IC|≥0.05 → 100 分，0.03 → 90，0.02 → 75，0.015 → 60，0.01 → 45，其余 20；")
    standards.append("   - IR：|IR|≥1.5 → 100，1.0 → 90，0.7 → 80，0.5 → 70，0.3 → 55，0.1 → 40；")
    standards.append("   - 统计显著性：p<0.01 → 100，0.01~0.05 → 85，0.05~0.1 → 70，≥0.1 → 40；")
    standards.append("   - 多空收益 / 胜率：收益≥4% → 100，3% → 90，2% → 80；胜率≥65% → 100，50% → 65。最终基础得分 = 40%*IC + 20%*IR + 20%*p 值 + 20%*(收益/胜率均值)。")
    standards.append("")
    standards.append("2. **整体样本表现（全样本 IC / IR / p 值 / 样本量）**")
    standards.append("   - 指标来自 `extra_stats['overall_metrics']`；IC / IR 与基础层级一致；")
    standards.append("   - 样本量：≥800 → 100，600 → 95，400 → 90，300 → 85，200 → 75，120 → 65，80 → 55，40 → 45；")
    standards.append("   - 综合得分 = 45%*整体 IC + 20%*整体 IR + 15%*p 值 + 10%*多空收益 + 10%*样本量。")
    standards.append("")
    standards.append("3. **滚动稳定度（Rolling IC）**")
    standards.append("   - 统计每个滚动窗口的 CV / 半衰期 / MAIC，分别使用 `_score_cv/_score_half_life/_score_maic`；")
    standards.append("   - CV≤0.3 → 100，0.6 → 60，>0.8 → 20；半衰期≥20 → 100，≤5 → 40；MAIC≥0.05 → 100，0.02 → 65；")
    standards.append("   - 各窗口得分求平均得到滚动得分。")
    standards.append("")
    standards.append("4. **时序一致性**")
    standards.append("   - Lag1 自相关最佳区间 0.4~0.8；")
    standards.append("   - 趋势相关系数 |r|≤0.2 保持高分，|r|>0.5 降至 20；")
    standards.append("   - 换向次数 1~3 次最优，>6 次降至 45；结合均值回归强度、排名波动等指标加权。")
    standards.append("")
    standards.append("5. **样本稳健性（抽样敏感度）**")
    standards.append("   - 80%/90%/100% 抽样的 IC 均值、标准差、IQR、成功率分别评分：std≤0.003 → 100，0.01 → 65，>0.02 → 35；成功率≥95% → 100，<80% → 35；")
    standards.append("   - 跨样本方差≤0.01 → 95，0.02 → 60，>0.05 → 30；平均后得到样本稳健性。")
    standards.append("")
    standards.append("6. **动态权重与可靠性**")
    standards.append("   - 名义权重：基础 35% / 整体 25% / 滚动 15% / 时序 12.5% / 样本 12.5%；")
    standards.append("   - 每维度计算 0.3~1.0 的可靠性（依据样本量、缺样、集中度、滚动 CV 等）；")
    standards.append("   - `final_weight = base_weight * clamp(reliability, 0.4~1.6)`，若可靠性 <0.35 则剔除；归一化后限制在 5%~55%。")
    standards.append("")
    standards.append("7. **综合评分与评级**")
    standards.append("   - 最终得分 = Σ(维度得分 * 动态权重)，保留 2 位小数；")
    standards.append("   - 评级：A+ ≥90，A ≥85，A- ≥75，B+ ≥65，B ≥55，C+ ≥45，否则 C；")
    standards.append("   - 正/负向因子根据评级给出建议权重与权重调整说明。")
    standards.append("")
    standards.append("8. **低样本与风险提示**")
    standards.append("   - 若触发低样本模式（avg<4 或缺样>20%），报告将提示“低样本量模式，窗口=3 日，min=2”，并上调稳健性权重；")
    standards.append("   - 滚动/时序可靠性为“低”时，报告内会标注“稳健性权重不足，建议谨慎使用”。")
    standards.append("")
    standards.append("9. **投资建议提示**")
    standards.append("   - A± 因子建议 10%~25% 权重，B 级 3%~10%，C 级主要用于辅助或对冲；")
    standards.append("   - 负向因子按反向 IC 评分，报告会给出反向胜率与权重提示。")

    return "\n".join(standards)


def _fa_classify_factors_by_ic(self):
    """
    根据IC均值将因子分为正向和负向两类
    """
    if not self.analysis_results:
        print("错误：请先运行因子分析")
        return pd.DataFrame(), pd.DataFrame()

    factors_data = []
    for factor, results in self.analysis_results.items():
        ic_mean = results.get('ic_mean', np.nan)
        ic_std = results.get('ic_std', np.nan)
        ir = results.get('ir', np.nan)
        raw_ic_mean = results.get('raw_ic_mean', np.nan)
        p_value = results.get('p_value', np.nan)
        long_short_return = 0
        if 'group_results' in results and results['group_results'] is not None:
            long_short_return = results['group_results'].get('long_short_return', np.nan)

        positive_score = 0
        if not np.isnan(ic_mean):
            if ic_mean >= 0.12:
                positive_score += 4
            elif ic_mean >= 0.08:
                positive_score += 3.5
            elif ic_mean >= 0.05:
                positive_score += 3
            elif ic_mean >= 0.02:
                positive_score += 2
            elif ic_mean >= 0.01:
                positive_score += 1
            else:
                positive_score += 0.5

        if not np.isnan(p_value):
            if p_value < 0.01:
                positive_score += 1
            elif p_value < 0.05:
                positive_score += 0.8
            elif p_value < 0.1:
                positive_score += 0.6
            else:
                positive_score += 0.3

        if not np.isnan(ir):
            if ir >= 1.5:
                positive_score += 2.5
            elif ir >= 1.0:
                positive_score += 2
            elif ir >= 0.5:
                positive_score += 1.5
            elif ir >= 0.3:
                positive_score += 1
            elif ir >= 0.15:
                positive_score += 0.8
            else:
                positive_score += 0.5

        if not np.isnan(long_short_return):
            if long_short_return >= 4:
                positive_score += 2
            elif long_short_return >= 3:
                positive_score += 1.8
            elif long_short_return >= 2:
                positive_score += 1.5
            elif long_short_return >= 1:
                positive_score += 1
            else:
                positive_score += 0.5

        negative_score = 0
        if not np.isnan(ic_mean) and ic_mean < 0:
            abs_ic = abs(ic_mean)
            if abs_ic >= 0.1:
                negative_score += 4
            elif abs_ic >= 0.07:
                negative_score += 3.5
            elif abs_ic >= 0.05:
                negative_score += 3
            elif abs_ic >= 0.03:
                negative_score += 2
            else:
                negative_score += 1

        if not np.isnan(p_value):
            if p_value < 0.01:
                negative_score += 1
            elif p_value < 0.05:
                negative_score += 0.8
            elif p_value < 0.1:
                negative_score += 0.6
            else:
                negative_score += 0.3

        if not np.isnan(ir):
            abs_ir = abs(ir)
            if abs_ir >= 1.5:
                negative_score += 2
            elif abs_ir >= 1.0:
                negative_score += 1.5
            elif abs_ir >= 0.5:
                negative_score += 1
            elif abs_ir >= 0.3:
                negative_score += 0.8
            elif abs_ir >= 0.15:
                negative_score += 0.5
            else:
                negative_score += 0.5

        if not np.isnan(long_short_return):
            if long_short_return < -2:
                negative_score += 1
            elif long_short_return < -1:
                negative_score += 0.8
            elif long_short_return < 0:
                negative_score += 0.6
            else:
                negative_score += 0.3

        factor_type = "正向因子" if ic_mean > 0 else "负向因子" if ic_mean < 0 else "中性因子"
        analysis_entry = getattr(self, 'analysis_results', {}).get(factor, {})
        integrated = analysis_entry.get('integrated_scores') if analysis_entry else None

        if integrated:
            rating = integrated.get('rating', "C")
            if rating and not str(rating).endswith('级'):
                rating_label = f"{rating}级"
            else:
                rating_label = rating or "C级"
            total_score = integrated.get('final_score', 0)
            base_component = integrated.get('base_score')
            overall_component = integrated.get('overall_score')
            stability_component = integrated.get('stability_score')
            rolling_component = integrated.get('rolling_score')
            temporal_component = integrated.get('temporal_score')
            sample_component = integrated.get('sample_score')
            component_weights = integrated.get('component_weights') or {}
            reliability_scores = integrated.get('reliability_scores') or {}
            reliability_labels = integrated.get('reliability_labels') or {}
        else:
            rating_label = None
            if ic_mean > 0:
                if ic_mean >= 0.08 and ir >= 0.3:
                    rating_label = "A+级" if positive_score >= 3.5 else "A级"
                elif ic_mean >= 0.08:
                    rating_label = "A-级"
                elif ic_mean >= 0.05 and ir >= 0.2:
                    rating_label = "B+级" if positive_score >= 2.5 else "B级"
                elif ic_mean >= 0.02:
                    rating_label = "C+级" if positive_score >= 1.5 else "C级"
                else:
                    rating_label = "D级"
            else:
                abs_ic = abs(ic_mean)
                if abs_ic >= 0.08 and abs(ir) >= 0.3:
                    rating_label = "A-级"
                elif abs_ic >= 0.05 and abs(ir) >= 0.2:
                    rating_label = "B+级"
                elif abs_ic >= 0.03:
                    rating_label = "B级"
                elif abs_ic >= 0.02:
                    rating_label = "C+级"
                else:
                    rating_label = "D级"
            total_score = positive_score if ic_mean > 0 else negative_score
            base_component = None
            overall_component = None
            stability_component = None
            rolling_component = None
            temporal_component = None
            sample_component = None

        factor_entry = {
            '因子名称': factor,
            'IC均值': ic_mean,
            'IC标准差': ic_std,
            '原始IC均值': raw_ic_mean,
            'IR值': ir,
            'p值': p_value,
            '多空收益': long_short_return,
            '胜率': results.get('win_rate', np.nan),
            '综合得分': total_score,
            '基础得分': base_component,
            '整体得分': overall_component,
            '稳健得分': stability_component,
            '滚动得分': rolling_component,
            '时序得分': temporal_component,
            '样本得分': sample_component,
            '因子类型': factor_type,
            '评级': rating_label,
        }
        if integrated:
            factor_entry.update({
                'weight_base': component_weights.get('base'),
                'weight_overall': component_weights.get('overall'),
                'weight_rolling': component_weights.get('rolling'),
                'weight_temporal': component_weights.get('temporal'),
                'weight_sample': component_weights.get('sample'),
                'reliability_score_base': reliability_scores.get('base'),
                'reliability_score_overall': reliability_scores.get('overall'),
                'reliability_score_rolling': reliability_scores.get('rolling'),
                'reliability_score_temporal': reliability_scores.get('temporal'),
                'reliability_score_sample': reliability_scores.get('sample'),
                'reliability_label_base': reliability_labels.get('base'),
                'reliability_label_overall': reliability_labels.get('overall'),
                'reliability_label_rolling': reliability_labels.get('rolling'),
                'reliability_label_temporal': reliability_labels.get('temporal'),
                'reliability_label_sample': reliability_labels.get('sample'),
                'weight_notes': integrated.get('weight_notes'),
            })
        factors_data.append(factor_entry)

    factors_df = pd.DataFrame(factors_data)
    positive_factors = factors_df[factors_df['IC均值'] > 0].sort_values('IC均值', ascending=False)
    negative_factors = factors_df[factors_df['IC均值'] < 0].sort_values('IC均值', ascending=True)
    return positive_factors, negative_factors


def _fa_generate_factor_classification_overview(self):
    """
    生成因子分类概览（Markdown）
    """
    positive_factors, negative_factors = _fa_classify_factors_by_ic(self)
    overview_lines = []

    total_factors = len(positive_factors) + len(negative_factors)
    overview_lines.append(f"- **因子总数**：{total_factors} 个")

    if len(positive_factors) > 0:
        avg_positive_ic = positive_factors['IC均值'].mean()
        best_positive_factor = positive_factors.iloc[0]
        overview_lines.append(
            f"- **正向因子**：{len(positive_factors)} 个，平均 IC {avg_positive_ic:.3f}，"
            f"最佳为 **{best_positive_factor['因子名称']}**（IC {best_positive_factor['IC均值']:.3f}，评级 {best_positive_factor['评级']}）"
        )
    else:
        overview_lines.append("- **正向因子**：暂无可用样本")

    if len(negative_factors) > 0:
        avg_negative_ic = negative_factors['IC均值'].mean()
        best_negative_factor = negative_factors.iloc[0]
        overview_lines.append(
            f"- **负向因子**：{len(negative_factors)} 个，平均 IC {avg_negative_ic:.3f}，"
            f"最佳为 **{best_negative_factor['因子名称']}**（反向 IC {abs(best_negative_factor['IC均值']):.3f}，评级 {best_negative_factor['评级']}）"
        )
    else:
        overview_lines.append("- **负向因子**：暂无可用样本")

    if len(positive_factors) > 0:
        overview_lines.append("\n**正向因子评级分布**")
        positive_ratings = positive_factors['评级'].value_counts()
        for rating, count in positive_ratings.items():
            overview_lines.append(f"- {rating}: {count} 个")

    if len(negative_factors) > 0:
        overview_lines.append("\n**负向因子评级分布**")
        negative_ratings = negative_factors['评级'].value_counts()
        for rating, count in negative_ratings.items():
            overview_lines.append(f"- {rating}: {count} 个")

    strategy_lines = []
    if len(positive_factors) > 0:
        strategy_lines.append("正向因子（做多）建议：")
        top_positive = positive_factors.head(3)
        for factor in top_positive.to_dict('records'):
            strategy_lines.append(
                f"  - {factor['因子名称']}：权重 { _fa_get_suggested_weight(self, factor['评级'], True)}，"
                f"IC {factor['IC均值']:.3f}，IR {factor['IR值']:.3f}"
            )
    if len(negative_factors) > 0:
        strategy_lines.append("负向因子（对冲/反向）建议：")
        top_negative = negative_factors.head(3)
        for factor in top_negative.to_dict('records'):
            strategy_lines.append(
                f"  - {factor['因子名称']}：权重 { _fa_get_suggested_weight(self, factor['评级'], False)}，"
                f"反向 IC {abs(factor['IC均值']):.3f}，IR {factor['IR值']:.3f}"
            )
    if strategy_lines:
        overview_lines.append("\n**策略要点**")
        overview_lines.extend(strategy_lines)

    overview_lines.append("\n**组合配置提示**")
    if len(positive_factors) > 0 and len(negative_factors) > 0:
        overview_lines.append("- 正向因子 + 负向因子对冲：3-5 个正向因子搭配 2-3 个负向因子。")
        overview_lines.append("- 保持风格多元，避免集中在单一行业或市值区间。")
    elif len(positive_factors) > 0:
        overview_lines.append("- 仅多头组合：挑选不同评级的正向因子分散风险。")
        overview_lines.append("- 定期复核 IC 稳定性，避免单因子权重过大。")
    elif len(negative_factors) > 0:
        overview_lines.append("- 仅反向/对冲组合：控制单个负因子的敞口，关注成交量限制。")
        overview_lines.append("- 配合核心正向策略或市场指数使用，避免孤立部署。")
    else:
        overview_lines.append("- 暂无组合建议，请确认分析结果。")

    return "\n".join(overview_lines)


# ======== 综合评分与稳健性融合工具 ========

def _clamp_score(value, low=0.0, high=100.0):
    if np.isnan(value):
        return low
    return max(low, min(high, value))


def _score_piecewise(value, thresholds):
    """
    thresholds: list of (threshold, score) sorted descending by threshold.
    """
    if value is None or np.isnan(value):
        return 50.0
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return thresholds[-1][1] if thresholds else 50.0


def _score_ic_component(ic_value):
    return _score_piecewise(abs(ic_value or 0.0), [
        (0.05, 100),
        (0.03, 90),
        (0.02, 75),
        (0.015, 60),
        (0.01, 45),
        (0.005, 30),
        (0.0, 20),
    ])


def _score_ir_component(ir_value):
    return _score_piecewise(abs(ir_value or 0.0), [
        (1.5, 100),
        (1.0, 90),
        (0.7, 80),
        (0.5, 70),
        (0.3, 55),
        (0.1, 40),
        (0.0, 25),
    ])


def _score_p_value_component(p_value):
    if p_value is None or np.isnan(p_value):
        return 60.0
    if p_value <= 0.01:
        return 100.0
    if p_value <= 0.05:
        return 85.0
    if p_value <= 0.1:
        return 70.0
    return 40.0


def _score_return_component(ret):
    if ret is None or np.isnan(ret):
        return 55.0
    value = abs(ret)
    return _score_piecewise(value, [
        (0.05, 100),
        (0.04, 90),
        (0.03, 80),
        (0.02, 65),
        (0.01, 50),
        (0.005, 35),
        (0.0, 20),
    ])


def _score_win_rate_component(win_rate):
    if win_rate is None or np.isnan(win_rate):
        return 60.0
    return _score_piecewise(win_rate, [
        (0.65, 100),
        (0.6, 90),
        (0.55, 80),
        (0.5, 65),
        (0.45, 50),
        (0.4, 35),
        (0.0, 20),
    ])


def _score_sample_size_component(sample_size):
    if sample_size is None or np.isnan(sample_size):
        return 60.0
    return _score_piecewise(sample_size, [
        (800, 100),
        (600, 95),
        (400, 90),
        (300, 85),
        (200, 75),
        (120, 65),
        (80, 55),
        (40, 45),
        (0, 35),
    ])


def _compute_win_rate_from_group_results(group_results):
    avg_returns = group_results.get('avg_returns') if group_results else None
    if avg_returns is None or '平均收益' not in avg_returns.columns:
        return None
    total = len(avg_returns)
    if total == 0:
        return None
    positive = (avg_returns['平均收益'] > 0).sum()
    return positive / total


def _score_cv(cv):
    if cv is None or np.isnan(cv):
        return 60.0
    if cv <= 0.3:
        return 100.0
    if cv <= 0.6:
        return 100.0 - (cv - 0.3) / 0.3 * 40.0  # 100 -> 60
    if cv <= 0.8:
        return 60.0 - (cv - 0.6) / 0.2 * 40.0   # 60 -> 20
    return 20.0


def _score_half_life(half_life):
    if half_life is None or np.isnan(half_life):
        return 55.0
    if half_life >= 20:
        return 100.0
    if half_life <= 5:
        return 40.0
    return 40.0 + (half_life - 5) / 15.0 * 60.0


def _score_maic(maic):
    if maic is None or np.isnan(maic):
        return 60.0
    return _score_piecewise(maic, [
        (0.05, 100),
        (0.04, 90),
        (0.03, 80),
        (0.02, 65),
        (0.01, 45),
        (0.0, 30),
    ])


def _score_lag_corr(lag_value):
    if lag_value is None or np.isnan(lag_value):
        return 60.0
    value = abs(lag_value)
    if value > 0.95:
        return 40.0
    if 0.4 <= value <= 0.8:
        # best around 0.6
        return 100.0 - abs(value - 0.6) / 0.2 * 20.0
    return 70.0 - min(abs(value - 0.4), abs(value - 0.8)) * 50.0


def _score_trend_corr(trend_corr):
    if trend_corr is None or np.isnan(trend_corr):
        return 65.0
    return max(20.0, 100.0 - abs(trend_corr) / 0.5 * 80.0)


def _score_sign_changes(sign_changes):
    if sign_changes is None or np.isnan(sign_changes):
        return 60.0
    if 1 <= sign_changes <= 3:
        return 95.0
    if 0 <= sign_changes < 1:
        return 75.0
    if 3 < sign_changes <= 6:
        return 70.0
    return 45.0


def _score_mean_reversion(value):
    if value is None or np.isnan(value):
        return 60.0
    value = max(0.0, min(1.0, value))
    return 40.0 + value * 60.0


def _score_rank_volatility(volatility):
    if volatility is None or np.isnan(volatility):
        return 60.0
    if volatility <= 0.3:
        return 95.0
    if volatility <= 0.6:
        return 95.0 - (volatility - 0.3) / 0.3 * 35.0
    if volatility <= 1.0:
        return 60.0 - (volatility - 0.6) / 0.4 * 30.0
    return 30.0


def _score_sample_std(std):
    if std is None or np.isnan(std):
        return 60.0
    if std <= 0.003:
        return 100.0
    if std <= 0.006:
        return 100.0 - (std - 0.003) / 0.003 * 15.0  # 100 -> 85
    if std <= 0.01:
        return 85.0 - (std - 0.006) / 0.004 * 20.0   # 85 -> 65
    if std <= 0.02:
        return 65.0 - (std - 0.02) / 0.01 * 30.0   # 65 -> 35
    if std <= 0.04:
        return 35.0 - (std - 0.04) / 0.02 * 10.0   # 35 -> 25
    return 25.0


def _score_iqr_range(iqr_value):
    if iqr_value is None or np.isnan(iqr_value):
        return 60.0
    if iqr_value <= 0.02:
        return 95.0
    if iqr_value <= 0.05:
        return 95.0 - (iqr_value - 0.02) / 0.03 * 45.0
    return 35.0


def _score_success_rate(rate):
    if rate is None or np.isnan(rate):
        return 60.0
    if rate >= 0.95:
        return 100.0
    if rate >= 0.9:
        return 85.0
    if rate >= 0.8:
        return 70.0
    if rate >= 0.7:
        return 55.0
    return 35.0


def _score_cross_sample_variance(value):
    if value is None or np.isnan(value):
        return 60.0
    if value <= 0.01:
        return 95.0
    if value <= 0.02:
        return 95.0 - (value - 0.01) / 0.01 * 35.0
    if value <= 0.05:
        return 60.0 - (value - 0.02) / 0.03 * 30.0
    return 30.0


def _average_scores(scores):
    valid = [s for s in scores if s is not None and not np.isnan(s)]
    if not valid:
        return 50.0
    return float(np.mean(valid))


def _score_reliability_metric(value, high, mid, low, higher_is_better=True, default=0.6):
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except Exception:
        pass
    val = float(value)
    if higher_is_better:
        if val >= high:
            return 1.0
        if val >= mid:
            return 0.8
        if val >= low:
            return 0.55
        return 0.35
    if val <= high:
        return 1.0
    if val <= mid:
        return 0.8
    if val <= low:
        return 0.55
    return 0.35


def _weighted_reliability(pairs, default=0.6):
    total = 0.0
    weight_sum = 0.0
    for score, weight in pairs:
        if score is None:
            continue
        total += score * weight
        weight_sum += weight
    if weight_sum <= 0:
        return default
    return max(0.3, min(1.0, total / weight_sum))


def _score_to_reliability(score_value, default=0.6):
    if score_value is None:
        return default
    try:
        if np.isnan(score_value):
            return default
    except Exception:
        pass
    scaled = float(score_value) / 100.0
    return max(0.3, min(1.0, scaled))


def _compute_base_reliability(extra_stats):
    extra_stats = extra_stats or {}
    segment_counts = extra_stats.get('segment_counts') or {}
    segment_ratios = [info.get('ratio', 0.0) for info in segment_counts.values()]
    concentration = max(segment_ratios) if segment_ratios else 0.0
    coverage_count = sum(1 for info in segment_counts.values() if info.get('count', 0) >= SEGMENT_MIN_SAMPLES)
    coverage_ratio = (coverage_count / max(len(segment_counts), 1)) if segment_counts else 1.0
    pairs = [
        (_score_reliability_metric(extra_stats.get('avg_daily_samples'), 8, 4, 2, True), 0.3),
        (_score_reliability_metric(extra_stats.get('daily_sample_cv'), 0.3, 0.6, 0.9, False), 0.2),
        (_score_reliability_metric(1 - (extra_stats.get('skip_ratio') or 0.0), 0.9, 0.7, 0.5, True), 0.2),
        (_score_reliability_metric(extra_stats.get('qualified_day_ratio'), 0.8, 0.5, 0.3, True), 0.15),
        (_score_reliability_metric(1 - concentration if segment_ratios else 1.0, 0.7, 0.5, 0.2, True), 0.1),
        (_score_reliability_metric(coverage_ratio, 0.8, 0.6, 0.3, True), 0.05),
    ]
    return _weighted_reliability(pairs)


def _compute_overall_reliability(extra_stats):
    extra_stats = extra_stats or {}
    overall_metrics = extra_stats.get('overall_metrics') or {}
    def _pick(field):
        if overall_metrics and field in overall_metrics:
            return overall_metrics.get(field)
        return extra_stats.get(field)
    pairs = [
        (_score_reliability_metric(_pick('overall_sample_size'), 800, 400, 200, True), 0.4),
        (_score_reliability_metric(_pick('overall_ci_width'), 0.02, 0.05, 0.08, False), 0.25),
        (_score_reliability_metric(1 - (_pick('overall_p_value') or 0.0), 0.99, 0.95, 0.9, True), 0.15),
        (_score_reliability_metric(_pick('overall_factor_unique'), 15, 8, 4, True), 0.1),
        (_score_reliability_metric(_pick('overall_return_unique'), 15, 8, 4, True), 0.1),
    ]
    return _weighted_reliability(pairs)


def _compute_rolling_reliability(aux_entry):
    summary = (aux_entry or {}).get('metric_summary') or {}
    pairs = []
    if 'rolling_cv_avg' in summary:
        pairs.append((_score_to_reliability(_score_cv(summary.get('rolling_cv_avg'))), 0.4))
    if 'rolling_half_life_avg' in summary:
        pairs.append((_score_to_reliability(_score_half_life(summary.get('rolling_half_life_avg'))), 0.3))
    if 'rolling_maic_avg' in summary:
        pairs.append((_score_to_reliability(_score_maic(summary.get('rolling_maic_avg'))), 0.3))
    return _weighted_reliability(pairs)


def _compute_temporal_reliability(aux_entry):
    summary = (aux_entry or {}).get('metric_summary') or {}
    pairs = []
    if 'temporal_autocorr_lag1' in summary:
        pairs.append((_score_to_reliability(_score_lag_corr(summary.get('temporal_autocorr_lag1'))), 0.35))
    if 'temporal_trend_corr' in summary:
        pairs.append((_score_to_reliability(_score_trend_corr(summary.get('temporal_trend_corr'))), 0.25))
    if 'temporal_sign_changes' in summary:
        pairs.append((_score_to_reliability(_score_sign_changes(summary.get('temporal_sign_changes'))), 0.2))
    if 'temporal_rank_volatility' in summary:
        pairs.append((_score_to_reliability(_score_rank_volatility(summary.get('temporal_rank_volatility'))), 0.2))
    return _weighted_reliability(pairs)


def _compute_sample_reliability(aux_entry):
    summary = (aux_entry or {}).get('metric_summary') or {}
    pairs = []
    if 'sample_success_rate_avg' in summary:
        pairs.append((_score_to_reliability(_score_success_rate(summary.get('sample_success_rate_avg'))), 0.35))
    if 'sample_ic_std_avg' in summary:
        pairs.append((_score_to_reliability(_score_sample_std(summary.get('sample_ic_std_avg'))), 0.35))
    if 'sample_cross_variance' in summary:
        pairs.append((_score_to_reliability(_score_cross_sample_variance(summary.get('sample_cross_variance'))), 0.3))
    return _weighted_reliability(pairs)


def _derive_reliability_scores(base_result, auxiliary_stats_entry, config=None):
    extra_stats = (base_result or {}).get('extra_stats') or {}
    aux_entry = auxiliary_stats_entry or {}
    scores = {
        'base': _compute_base_reliability(extra_stats),
        'overall': _compute_overall_reliability(extra_stats),
        'rolling': _compute_rolling_reliability(aux_entry),
        'temporal': _compute_temporal_reliability(aux_entry),
        'sample': _compute_sample_reliability(aux_entry),
    }
    cleaned = {}
    for key, value in scores.items():
        if value is None or np.isnan(value):
            cleaned[key] = 0.6
        else:
            cleaned[key] = max(0.3, min(1.0, float(value)))
    return cleaned


def _label_reliability(score):
    if score is None:
        return "未知"
    if score >= 0.8:
        return "高"
    if score >= 0.6:
        return "中"
    return "低"


def _apply_reliability_weights(base_weights, reliability_scores, config=None):
    config = config or {}
    drop_threshold = config.get('drop_threshold', 0.35)
    scale_bounds = config.get('scale_bounds', (0.4, 1.6))
    min_ratio, max_ratio = config.get('normalized_bounds', (0.05, 0.55))
    base = base_weights.copy()
    scaled = {}
    for key, base_weight in base.items():
        rel = reliability_scores.get(key)
        if rel is None:
            scaled[key] = base_weight
            continue
        if rel < drop_threshold:
            scaled[key] = 0.0
            continue
        scale_min, scale_max = scale_bounds
        relative = (rel - drop_threshold) / (1.0 - drop_threshold) if rel < 1.0 else 1.0
        relative = max(0.0, min(1.0, relative))
        multiplier = scale_min + (scale_max - scale_min) * relative
        scaled[key] = base_weight * multiplier
    total = sum(scaled.values())
    if total <= 0:
        return bounded_normalize(base, min_ratio=min_ratio, max_ratio=max_ratio)
    normalized = {key: value / total for key, value in scaled.items()}
    normalized = bounded_normalize(normalized, min_ratio=min_ratio, max_ratio=max_ratio)
    return normalized


def _compute_base_score(base_result):
    ic_score = _score_ic_component(base_result.get('ic_mean'))
    ir_score = _score_ir_component(base_result.get('ir'))
    p_score = _score_p_value_component(base_result.get('p_value'))
    return_score = _score_return_component(
        (base_result.get('group_results') or {}).get('long_short_return')
    )
    win_rate = _compute_win_rate_from_group_results(base_result.get('group_results'))
    win_score = _score_win_rate_component(win_rate)
    combined_return = (return_score + win_score) / 2.0
    weighted = (
        ic_score * 0.4 +
        ir_score * 0.2 +
        p_score * 0.2 +
        combined_return * 0.2
    )
    return _clamp_score(weighted)


def _compute_rolling_score(rolling_stats):
    if not rolling_stats:
        return 55.0
    cv_scores, half_scores, maic_scores = [], [], []
    for info in rolling_stats.values():
        stability = info.get('stability') or {}
        decay = info.get('decay') or {}
        stats = info.get('stats') or {}
        cv_scores.append(_score_cv(stability.get('coefficient_of_variation')))
        half_scores.append(_score_half_life(decay.get('half_life')))
        ic_values = stats.get('ic_values')
        if ic_values:
            maic_scores.append(_score_maic(np.nanmean(np.abs(ic_values))))
        else:
            maic_scores.append(_score_maic(stability.get('mean_abs_ic')))
    avg = _average_scores([
        _average_scores(cv_scores),
        _average_scores(half_scores),
        _average_scores(maic_scores),
    ])
    return _clamp_score(avg)


def _compute_temporal_score(temporal_stats):
    if not temporal_stats:
        return 55.0
    ic_stability = temporal_stats.get('ic_stability') or {}
    trends = temporal_stats.get('temporal_trends') or {}
    rank_stats = temporal_stats.get('rank_stability') or {}
    scores = []
    scores.append(_score_lag_corr(ic_stability.get('autocorr_lag1')))
    scores.append(_score_trend_corr(trends.get('trend_correlation') or ic_stability.get('trend_correlation')))
    scores.append(_score_sign_changes(trends.get('sign_changes')))
    scores.append(_score_mean_reversion(trends.get('mean_reversion_strength')))
    scores.append(_score_rank_volatility(rank_stats.get('ranking_volatility')))
    return _clamp_score(_average_scores(scores))


def _compute_sample_score(sample_stats):
    if not sample_stats:
        return 55.0
    effects = sample_stats.get('sample_size_effects') or {}
    std_scores, iqr_scores, success_scores = [], [], []
    for stats in effects.values():
        std_scores.append(_score_sample_std(stats.get('ic_std')))
        if stats.get('ic_q75') is not None and stats.get('ic_q25') is not None:
            iqr_scores.append(_score_iqr_range(stats.get('ic_q75') - stats.get('ic_q25')))
        success_scores.append(_score_success_rate(stats.get('success_rate')))
    robustness = sample_stats.get('robustness_metrics') or {}
    variance_score = _score_cross_sample_variance(robustness.get('mean_variance_across_samples'))
    overall = _average_scores([
        _average_scores(std_scores),
        _average_scores(iqr_scores),
        _average_scores(success_scores),
        variance_score,
    ])
    return _clamp_score(overall)


def _map_score_to_rating(final_score):
    if final_score >= 90:
        return 'A+'
    if final_score >= 85:
        return 'A'
    if final_score >= 75:
        return 'A-'
    if final_score >= 65:
        return 'B+'
    if final_score >= 55:
        return 'B'
    if final_score >= 45:
        return 'C+'
    return 'C'


def _extract_overall_metrics(base_result):
    extra_stats = (base_result or {}).get('extra_stats') or {}
    metrics = extra_stats.get('overall_metrics') or {}
    if not metrics:
        metrics = {
            key: extra_stats.get(key)
            for key in extra_stats.keys()
            if key.startswith('overall_')
        }
    return {k: v for k, v in (metrics or {}).items() if v is not None}


def _compute_overall_score(base_result, fallback=None):
    metrics = _extract_overall_metrics(base_result)
    if not metrics:
        return fallback if fallback is not None else _compute_base_score(base_result)
    ic_score = _score_ic_component(metrics.get('overall_ic'))
    ir_score = _score_ir_component(metrics.get('overall_ir'))
    p_score = _score_p_value_component(metrics.get('overall_p_value'))
    return_score = _score_return_component(
        (base_result.get('group_results') or {}).get('long_short_return')
    )
    sample_score = _score_sample_size_component(metrics.get('overall_sample_size'))
    weighted = (
        ic_score * 0.45 +
        ir_score * 0.2 +
        p_score * 0.15 +
        return_score * 0.1 +
        sample_score * 0.1
    )
    return _clamp_score(weighted)


_COMPONENT_TITLES = {
    'base': '基础表现（日度IC/IR）',
    'overall': '整体样本表现（整体IC/IR/P值）',
    'rolling': '滚动稳定度',
    'temporal': '时序一致性',
    'sample': '样本稳健性',
}

_COMPONENT_ORDER = ['base', 'overall', 'rolling', 'temporal', 'sample']

_DEFAULT_WEIGHT_TEMPLATES = {
    'balanced': {
        'base': 0.35,
        'overall': 0.25,
        'rolling': 0.15,
        'temporal': 0.125,
        'sample': 0.125,
    },
    'stability': {
        'base': 0.25,
        'overall': 0.30,
        'rolling': 0.20,
        'temporal': 0.15,
        'sample': 0.10,
    },
    'aggressive': {
        'base': 0.45,
        'overall': 0.20,
        'rolling': 0.15,
        'temporal': 0.10,
        'sample': 0.10,
    },
}

_TEMPLATE_LABELS = {
    'balanced': "均衡",
    'stability': "稳健",
    'aggressive': "进取",
}


def _select_weight_template(extra_stats, cfg):
    """
    根据板块覆盖情况（集中度/有效板块数量）选择权重模板。
    """
    templates = dict(_DEFAULT_WEIGHT_TEMPLATES)
    templates.update(cfg.get('weight_templates') or {})
    if cfg.get('base_weights'):
        templates['balanced'] = cfg['base_weights']
    default_template = (templates.get('balanced') or _DEFAULT_WEIGHT_TEMPLATES['balanced']).copy()

    segment_concentration = extra_stats.get('segment_concentration')
    valid_segments = extra_stats.get('segment_valid_count', 0) or 0
    template_name = 'balanced'

    try:
        conc_value = float(segment_concentration) if segment_concentration is not None else None
    except (TypeError, ValueError):
        conc_value = None

    if conc_value is not None and conc_value >= 0.75:
        template_name = 'stability'
    elif conc_value is not None and conc_value <= 0.45 and valid_segments >= 2:
        template_name = 'aggressive'
    elif valid_segments <= 1:
        template_name = 'stability'

    weights = templates.get(template_name)
    if not weights:
        weights = default_template.copy()
        template_name = 'balanced'
    else:
        weights = weights.copy()
    return template_name, weights


def compute_integrated_factor_scores(base_result, auxiliary_stats_entry, reliability_config=None):
    """
    结合基础表现与辅助稳健性指标，输出综合评分结果，并根据可靠性动态调整权重。
    """
    aux_entry = auxiliary_stats_entry or {}
    extra_stats = (base_result or {}).get('extra_stats') or {}
    cfg = reliability_config or RELIABILITY_CONFIG or {}
    base_score = _compute_base_score(base_result)
    overall_score = _compute_overall_score(base_result, fallback=base_score)
    rolling_score = _compute_rolling_score(aux_entry.get('rolling'))
    temporal_score = _compute_temporal_score(aux_entry.get('temporal'))
    sample_score = _compute_sample_score(aux_entry.get('sample'))
    stability_score = _average_scores([rolling_score, temporal_score, sample_score])

    template_name, base_weights = _select_weight_template(extra_stats, cfg)

    reliability_scores = _derive_reliability_scores(base_result, aux_entry, cfg) if cfg else {}
    reliability_labels = {k: _label_reliability(v) for k, v in reliability_scores.items()} if reliability_scores else {}
    component_weights = _apply_reliability_weights(base_weights, reliability_scores, cfg) if reliability_scores else base_weights

    final_score = (
        base_score * component_weights.get('base', base_weights.get('base', 0.0)) +
        overall_score * component_weights.get('overall', base_weights.get('overall', 0.0)) +
        rolling_score * component_weights.get('rolling', base_weights.get('rolling', 0.0)) +
        temporal_score * component_weights.get('temporal', base_weights.get('temporal', 0.0)) +
        sample_score * component_weights.get('sample', base_weights.get('sample', 0.0))
    )

    weight_notes = []
    for key in _COMPONENT_ORDER:
        weight_pct = component_weights.get(key)
        if weight_pct is None:
            continue
        label = reliability_labels.get(key, "未知")
        weight_notes.append(f"{_COMPONENT_TITLES.get(key, key)} 权重{weight_pct * 100:.1f}%（可靠性{label}）")
    weight_notes_text = "；".join(weight_notes) if weight_notes else None
    template_label = _TEMPLATE_LABELS.get(template_name)
    if template_label:
        template_note = f"权重模板：{template_label}"
        weight_notes_text = f"{template_note}；{weight_notes_text}" if weight_notes_text else template_note
    primary_seg = extra_stats.get('segment_primary')
    primary_ratio = extra_stats.get('segment_primary_ratio')
    if primary_seg and primary_ratio and primary_ratio >= 0.6:
        segment_note = f"样本{primary_ratio * 100:.1f}%集中在{primary_seg}"
        weight_notes_text = f"{weight_notes_text}；{segment_note}" if weight_notes_text else segment_note
    segment_warnings = extra_stats.get('segment_warnings') or []
    if segment_warnings:
        notice = "；".join(segment_warnings)
        weight_notes_text = f"{weight_notes_text}；{notice}" if weight_notes_text else notice

    rating = _map_score_to_rating(final_score)
    return {
        'base_score': round(base_score, 2),
        'overall_score': round(overall_score, 2),
        'rolling_score': round(rolling_score, 2),
        'temporal_score': round(temporal_score, 2),
        'sample_score': round(sample_score, 2),
        'stability_score': round(stability_score, 2),
        'final_score': round(final_score, 2),
        'rating': rating,
        'component_weights': component_weights,
        'reliability_scores': reliability_scores,
        'reliability_labels': reliability_labels,
        'weight_notes': weight_notes_text,
    }
