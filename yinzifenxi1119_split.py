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
from datetime import datetime

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from daima.fa_logging import Logger as ExternalLogger
from daima.fa_nonparam_analysis import FactorAnalysis
from daima.fa_param_analysis import ParameterizedFactorAnalyzer
from daima.fa_nonparam_helpers import (
    _fa_classify_factors_by_ic,
    _fa_generate_factor_classification_overview,
    _fa_get_suggested_weight,
    _fa_get_scoring_standards,
)
from daima.fa_nonparam_report import (
    _fa_generate_summary_report,
    _fa_generate_factor_analysis_report,
    _fa_generate_positive_factors_analysis,
    _fa_generate_negative_factors_analysis,
)

# 默认数据文件路径设置（方便用户修改）
DEFAULT_DATA_FILE = "创业板单日下跌14%详细交易日数据（清理后）1114.xlsx"

# 读取完整因子数据文件
print(f"[INFO] 使用指定的数据文件: {DEFAULT_DATA_FILE}")

# 检查文件是否存在
if os.path.exists(DEFAULT_DATA_FILE):
    print(f"[INFO] 数据文件路径: {os.path.abspath(DEFAULT_DATA_FILE)}")
else:
    print(f"[ERROR] 数据文件不存在，请检查文件路径")
    print(f"[ERROR] 期望的文件: {DEFAULT_DATA_FILE}")
    sys.exit(1)

# 尝试导入scipy.stats，如果不可用则设置标志
HAS_SCIPY = False
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    print("[WARN] scipy不可用，部分统计计算功能将被简化，但基本分析仍将继续")

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
    print("[WARN] matplotlib或seaborn不可用，可视化功能将被禁用，但核心分析仍将继续")

# 稳健性统计与敏感性分析函数现由 daima.fa_stat_utils 提供，主脚本不再重复定义。

# FactorAnalysis 已迁移至 daima.fa_nonparam_analysis 模块

# ParameterizedFactorAnalyzer 已迁移至 daima.fa_param_analysis 模块
# 使用时请从 daima.fa_param_analysis 导入


# 主函数示例
def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="运行非参数 + 带参数因子分析流程")
    parser.add_argument(
        "--summary-report",
        action="store_true",
        help="生成精简版主报告，只输出评分融合与稳健性摘要信息",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """主函数"""
    args = parse_cli_args(argv)
    # 初始化日志记录器
    logger = ExternalLogger()
    sys.stdout = logger  # 重定向输出到日志记录器
    
    print("[INFO] 因子分析程序启动")
    print(f"[INFO] 日志输出文件: {logger.log_file}")
    
    # 创建因子分析对象
    analyzer = FactorAnalysis()
    
    # 加载数据
    print("[INFO] 开始加载数据...")
    if not analyzer.load_data():
        print("[ERROR] 数据加载失败，程序退出")
        logger.close()
        return
    else:
        print(f"[INFO] 数据加载完成，原始样本量 {len(analyzer.data)} 行")
    
    # 预处理数据
    print("\n[INFO] === 数据预处理 ===")
    
    use_pearson = False
    process_factors = True
    factor_method = 'standardize'
    winsorize = True
    
    if not analyzer.preprocess_data(process_factors=process_factors, factor_method=factor_method, winsorize=winsorize):
        print("[ERROR] 数据预处理失败，程序退出")
        logger.close()
        return
    else:
        processed_rows = len(getattr(analyzer, 'processed_data', []))
        print(f"[INFO] 数据预处理完成，剩余样本 {processed_rows} 行")
    
    print("\n[INFO] === 因子分析选项 ===")
    print("\n[INFO] 可用的因子列表:")
    for i, factor in enumerate(analyzer.factors, 1):
        print(f"{i}. {factor}")
    
    print("\n[INFO] 执行全因子分析...")
    try:
        analyzer.run_factor_analysis(use_pearson=use_pearson)
        print(f"[INFO] 因子分析阶段完成，取得 {len(analyzer.analysis_results)} 个因子结果")

        has_results = hasattr(analyzer, 'analysis_results') and analyzer.analysis_results
        if has_results:
            print("\n[INFO] 整合辅助稳健性分析数据...")
            try:
                aux_stats = analyzer.generate_auxiliary_analysis_report()
                if not aux_stats:
                    print("[WARN] 未获取到辅助分析结果，报告将仅包含基础表现指标")
                else:
                    print("[INFO] 辅助稳健性分析数据整合完成")
            except Exception as aux_err:
                print(f"[ERROR] 整合辅助分析数据时出错: {str(aux_err)}")

            summary_df = _fa_generate_summary_report(analyzer)
            _fa_generate_factor_analysis_report(
                analyzer,
                summary_df,
                process_factors=process_factors,
                factor_method=factor_method,
                winsorize=winsorize,
                summary_mode=args.summary_report,
            )
            print("[INFO] 主报告生成完成")
        else:
            print("[WARN] 分析结果为空，无法生成报告")
    except Exception as e:
        print(f"[ERROR] 执行全因子分析时出错: {str(e)}")
    
    print("\n[INFO] 开始对所有因子执行10等分因子分析...")
    
    selected_factors = analyzer.factors.copy()
    for factor_name in selected_factors:
        print(f"\n[STEP] 分析因子: {factor_name}")
        try:
            ic_mean, ic_std, t_stat, p_value, _ = analyzer.calculate_ic(factor_name, use_pearson=use_pearson)
            group_results = analyzer.calculate_group_returns(factor_name, n_groups=10)
            
            if group_results:
                avg_returns = group_results['avg_returns']
                long_short_return = group_results['long_short_return'] if not np.isnan(group_results['long_short_return']) else 0
                ir = ic_mean / ic_std if ic_std != 0 else np.nan
                
                analyzer.analysis_results[factor_name] = {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'ir': ir,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'group_results': group_results,
                }
                
                print(f"IC均值: {ic_mean:.4f}")
                print(f"IC标准差: {ic_std:.4f}")
                print(f"信息比率: {ir:.4f}" if not np.isnan(ir) else "信息比率: N/A")
                print(f"多空收益: {long_short_return:.4f}%")
                print("\n10等分分组收益:")
                print(avg_returns.to_string(index=False, float_format='%.3f'))
                print(f"  因子 {factor_name} 的详细分组收益数据将由带参数因子分析器生成")
            else:
                print(f"[WARN] 无法计算因子 '{factor_name}' 的分组收益")
                continue
        except Exception as e:
            print(f"[ERROR] 分析因子 '{factor_name}' 时出错: {str(e)}")
    
    print("\n[INFO] === 因子分析结果已保存 ===")
    
    print("\n[INFO] 开始生成带参数因子综合分析报告...")
    try:
        parameterized_analyzer = ParameterizedFactorAnalyzer(analyzer.data.copy())
        if parameterized_analyzer.preprocess_data():
            report_filename = parameterized_analyzer.generate_parameterized_report()
            if report_filename:
                print(f"[OK] 带参数因子综合分析报告已生成: {report_filename}")
                print("该报告包含以下内容：")
                print("  • 因子排行榜（正向和负向因子分别排名）")
                print("  • 最优秀因子推荐（3个最优秀的正向和负向因子）")
                print("  • 详细因子分析（每个因子的完整指标分析）")
                print("  • 分组详细数据（每个因子的10等分分组表现）")
                print("  • 投资策略建议（基于因子表现的组合配置建议）")
                print("  • 风险提示（使用注意事项）")
                print("\n同时生成的CSV文件：")
                print("  • 带参数因子分析数据_[时间戳].csv（综合评分数据）")
                print("  • 带参数因子详细分析_[因子名称]_[时间戳].csv（各因子分组数据）")
            else:
                print("[ERROR] 带参数因子综合分析报告生成失败")
        else:
            print("[ERROR] 带参数因子数据预处理失败")
    except Exception as e:
        print(f"[ERROR] 生成带参数因子综合分析报告时出错: {str(e)}")
    
    print("\n[INFO] 因子分析程序已完成")
    logger.close()


if __name__ == "__main__":
    main()

