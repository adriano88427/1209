# -*- coding: utf-8 -*-
"""
带参数因子分析报告模块。

该模块承载 ParameterizedFactorAnalyzer 生成 TXT/CSV 报告的逻辑，
主类只需调用公开函数即可，避免大段字符串常量留在主脚本中。
"""

from datetime import datetime

from daima.fa_config import DEFAULT_DATA_FILE, build_report_path


def _fa_generate_parameterized_report(self):
    """生成带参数因子综合分析报告并输出所有相关文件。"""
    print("开始生成带参数因子综合分析报告...")

    factor_results = {}
    for factor in self.factor_list:
        print(f"分析因子: {factor}")
        results = self.calculate_comprehensive_metrics(factor)
        if results:
            factor_results[factor] = results

    if not factor_results:
        print("错误: 没有有效的带参数因子分析结果")
        return None

    scores_df = self.score_factors(factor_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'带参数因子综合分析报告_{timestamp}.txt'
    report_path = build_report_path(report_filename)

    with open(report_path, 'w', encoding='utf-8') as f:
        # 报告头部
        f.write("=" * 80 + "\n")
        f.write("              带参数因子综合分析详细报告                \n")
        f.write("=" * 80 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件: {DEFAULT_DATA_FILE}\n")
        f.write(f"总因子数量: {len(self.factor_list)}\n")
        f.write(f"有效分析因子: {len(factor_results)}\n")
        f.write("分析指标: 胜率、最大回撤、年化收益率、年化收益标准差、年化夏普比率\n")
        f.write("评分体系: 每个参数区间作为独立单元进行评分\n\n")

        # 1. 参数区间排行榜
        f.write("1. 参数区间排行榜\n")
        f.write("=" * 50 + "\n\n")

        positive_factors = scores_df[scores_df['因子方向'] == '正向'].sort_values('综合得分', ascending=False)
        negative_factors = scores_df[scores_df['因子方向'] == '负向'].sort_values('综合得分', ascending=False)

        def _write_ranking(title, data_frame):
            f.write(title + "\n")
            f.write("=" * 40 + "\n")
            f.write(f"{'排名':<4} {'因子名称':<20} {'参数区间':<15} {'得分':<6} {'评级':<6} "
                    f"{'胜率':<6} {'年化收益':<8} {'夏普比率':<8} {'最大回撤':<8}\n")
            f.write("-" * 90 + "\n")
            for i, (_, row) in enumerate(data_frame.iterrows(), 1):
                rating = (
                    'A+' if row['综合得分'] >= 9 else
                    'A' if row['综合得分'] >= 8 else
                    'B+' if row['综合得分'] >= 7 else
                    'B' if row['综合得分'] >= 6 else 'C'
                )
                f.write(f"{i:<4} {row['因子名称']:<20} {row['参数区间']:<15} "
                        f"{row['综合得分']:<6.1f} {rating:<6} {row['胜率']:<6.1%} "
                        f"{row['年化收益率']:<8.3f} {row['年化夏普比率']:<8.3f} "
                        f"{row['最大回撤']:<8.1%}\n")

        _write_ranking("【正向参数区间排行榜】", positive_factors)
        if len(positive_factors) > 0:
            best_positive = positive_factors.iloc[0]
            f.write(f"\n最佳正向参数区间: {best_positive['因子名称']} {best_positive['参数区间']} (排名第1)\n\n")

        _write_ranking("【负向参数区间排行榜】", negative_factors)
        if len(negative_factors) > 0:
            best_negative = negative_factors.iloc[0]
            f.write(f"\n最佳负向参数区间: {best_negative['因子名称']} {best_negative['参数区间']} (排名第1)\n\n")

        # 2. 最优秀参数区间推荐
        f.write("2. 最优秀参数区间推荐\n")
        f.write("=" * 50 + "\n\n")
        top_5_positive = positive_factors.head(5)
        top_5_negative = negative_factors.head(5)

        def _write_top(title, frame):
            f.write(title + "\n")
            f.write("-" * 40 + "\n\n")
            for i, (_, factor) in enumerate(frame.iterrows(), 1):
                f.write(f"第{i}名: {factor['因子名称']} {factor['参数区间']}\n")
                f.write(f"综合得分: {factor['综合得分']:.1f}/10\n")
                f.write(f"胜率: {factor['胜率']:.1%}\n")
                f.write(f"年化收益率: {factor['年化收益率']:.3f}\n")
                f.write(f"年化收益标准差: {factor['年化收益标准差']:.3f}\n")
                f.write(f"年化夏普比率: {factor['年化夏普比率']:.3f}\n")
                f.write(f"最大回撤: {factor['最大回撤']:.1%}\n\n")

        if len(top_5_positive) > 0:
            _write_top("【最优秀的5个正向参数区间】", top_5_positive)
        if len(top_5_negative) > 0:
            _write_top("【最优秀的5个负向参数区间】", top_5_negative)

        # 3. 详细参数区间分析
        f.write("3. 详细参数区间分析\n")
        f.write("=" * 50 + "\n\n")
        all_factors_sorted = scores_df.sort_values('综合得分', ascending=False)

        for _, factor_row in all_factors_sorted.iterrows():
            f.write(f"【{factor_row['因子名称']} {factor_row['参数区间']}】\n")
            f.write("-" * 60 + "\n")
            f.write(f"综合得分: {factor_row['综合得分']:.1f}/10\n")
            f.write(f"因子方向: {factor_row['因子方向']}\n")
            if factor_row['综合得分'] >= 9:
                rating = "A级（优秀）"
            elif factor_row['综合得分'] >= 8:
                rating = "B+级（良好）"
            elif factor_row['综合得分'] >= 6:
                rating = "B级（一般）"
            else:
                rating = "C级（较差）"
            f.write(f"综合评级: {rating}\n")

            f.write("核心指标:\n")
            f.write(f"• 胜率: {factor_row['胜率']:.1%}\n")
            f.write(f"• 年化收益率: {factor_row['年化收益率']:.3f}\n")
            f.write(f"• 年化收益标准差: {factor_row['年化收益标准差']:.3f}\n")
            f.write(f"• 年化夏普比率: {factor_row['年化夏普比率']:.3f}\n")
            f.write(f"• 最大回撤: {factor_row['最大回撤']:.1%}\n\n")

            factor_name = factor_row['因子名称']
            if factor_name in factor_results:
                group_stats = factor_results[factor_name]['group_stats']
                param_range = factor_row['参数区间']
                group_data = group_stats[group_stats['参数区间'] == param_range]
                if len(group_data) > 0:
                    group = group_data.iloc[0]
                    f.write("分组详细数据:\n")
                    f.write(f"• 平均收益: {group['平均收益']:.3f}\n")
                    f.write(f"• 收益标准差: {group['收益标准差']:.3f}\n")
                    f.write(f"• 胜率: {group['胜率']:.1%}\n")
                    f.write(f"• 最大回撤: {group['最大回撤']:.1%}\n")
                    f.write(f"• 年化收益率: {group['年化收益率']:.3f}\n")
                    f.write(f"• 年化收益标准差: {group['年化收益标准差']:.3f}\n")
                    f.write(f"• 年化夏普比率: {group['年化夏普比率']:.3f}\n")

            f.write("=" * 60 + "\n\n")

        # 4. 投资策略建议
        f.write("4. 投资策略建议\n")
        f.write("=" * 50 + "\n\n")
        if len(top_5_positive) > 0:
            f.write("推荐参数区间配置:\n")
            f.write("-" * 30 + "\n")
            for i, (_, factor) in enumerate(top_5_positive.iterrows(), 1):
                weight = 0.25 - i * 0.03
                f.write(f"第{i}名 {factor['因子名称']} {factor['参数区间']}: {weight*100:.0f}% 权重\n")

            if len(top_5_negative) > 0:
                f.write("\n可选负向参数区间配置:\n")
                for i, (_, factor) in enumerate(top_5_negative.iterrows(), 1):
                    weight = 0.1 - i * 0.01
                    f.write(f"第{i}名 {factor['因子名称']} {factor['参数区间']}: {weight*100:.0f}% 权重\n")

            f.write("\n策略说明:\n")
            f.write("• 重点配置排名前5的正向参数区间\n")
            f.write("• 可选择性配置负向参数区间作为对冲\n")
            f.write("• 每个参数区间独立考虑收益风险特征\n")
            f.write("• 根据实际参数区间效果动态调整权重\n")
            f.write("• 定期重新评估参数区间有效性\n")
            f.write("• 严格控制单个参数区间仓位风险\n")

        # 5. 风险提示
        f.write("\n5. 风险提示\n")
        f.write("=" * 50 + "\n\n")
        f.write("• 历史表现不代表未来收益\n")
        f.write("• 带参数因子有效性可能随市场环境变化\n")
        f.write("• 参数区间设置需要谨慎验证\n")
        f.write("• 建议结合其他分析方法使用\n")
        f.write("• 注意分散投资，控制总体风险\n")
        f.write("• 每个参数区间需独立监控其表现\n")

    print(f"带参数因子综合分析报告已保存到 '{report_path}'")

    csv_filename = f'带参数因子分析数据_{timestamp}.csv'
    csv_path = build_report_path(csv_filename)
    scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细数据已保存到 '{csv_path}'")

    numeric_columns = [
        '平均收益', '收益标准差', '胜率', '最大回撤',
        '年化收益率', '年化收益标准差', '年化夏普比率', '年化索提诺比率'
    ]
    for factor_name, results in factor_results.items():
        factor_csv = build_report_path(f'带参数因子详细分析_{factor_name}_{timestamp}.csv')
        formatted_group_stats = results['group_stats'].copy()
        for col in numeric_columns:
            if col in formatted_group_stats.columns:
                formatted_group_stats[col] = formatted_group_stats[col].round(3)
        formatted_group_stats.to_csv(factor_csv, index=False, encoding='utf-8-sig')

    return report_path
