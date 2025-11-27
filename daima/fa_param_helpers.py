# -*- coding: utf-8 -*-
"""
带参数因子分析辅助函数。

当前用于承载评分逻辑，后续可逐步扩展（例如分类、权重等）。
"""

import pandas as pd


def _fa_score_parameterized_factors(factor_results):
    """
    根据分组统计结果为每个参数区间打分，返回 DataFrame。

    Args:
        factor_results (dict): {因子名: {'group_stats': DataFrame, ...}}
    """
    all_scores = []
    for factor, results in factor_results.items():
        group_stats = results['group_stats']
        for _, group in group_stats.iterrows():
            param_range = group['参数区间']
            win_rate = group['胜率']
            max_drawdown = group['最大回撤']
            ann_return = group['年化收益率']
            ann_std = group['年化收益标准差']
            sharpe_ratio = group['年化夏普比率']

            if win_rate >= 0.7:
                win_score = 10
            elif win_rate >= 0.6:
                win_score = 8
            elif win_rate >= 0.5:
                win_score = 6
            elif win_rate >= 0.4:
                win_score = 4
            else:
                win_score = 2

            if max_drawdown >= 0:
                drawdown_score = 10
            elif max_drawdown >= -0.05:
                drawdown_score = 8
            elif max_drawdown >= -0.1:
                drawdown_score = 6
            elif max_drawdown >= -0.2:
                drawdown_score = 4
            else:
                drawdown_score = 2

            if ann_return >= 2.0:
                return_score = 10
            elif ann_return >= 1.0:
                return_score = 8
            elif ann_return >= 0.5:
                return_score = 6
            elif ann_return >= 0:
                return_score = 4
            else:
                return_score = 2

            if ann_std <= 0.5:
                std_score = 10
            elif ann_std <= 1.0:
                std_score = 8
            elif ann_std <= 2.0:
                std_score = 6
            elif ann_std <= 3.0:
                std_score = 4
            else:
                std_score = 2

            if sharpe_ratio >= 3.0:
                sharpe_score = 10
            elif sharpe_ratio >= 2.0:
                sharpe_score = 8
            elif sharpe_ratio >= 1.0:
                sharpe_score = 6
            elif sharpe_ratio >= 0:
                sharpe_score = 4
            else:
                sharpe_score = 2

            total_score = (
                win_score * 0.3
                + return_score * 0.25
                + sharpe_score * 0.25
                + std_score * 0.1
                + drawdown_score * 0.1
            )
            factor_direction = "正向" if ann_return >= 0 else "负向"

            all_scores.append({
                '因子名称': factor,
                '参数区间': param_range,
                '胜率': win_rate,
                '最大回撤': max_drawdown,
                '年化收益率': ann_return,
                '年化收益标准差': ann_std,
                '年化夏普比率': sharpe_ratio,
                '因子方向': factor_direction,
                '综合得分': total_score,
                '胜率得分': win_score,
                '收益率得分': return_score,
                '夏普得分': sharpe_score,
                '风险得分': std_score,
                '回撤得分': drawdown_score,
            })

    return pd.DataFrame(all_scores)

