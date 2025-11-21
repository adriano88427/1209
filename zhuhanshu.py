"""
优化后的主函数
整合所有模块，提供完整的因子分析流程
"""

import argparse
import logging
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataProcessingError, exception_handler, cache_result

# 导入功能模块
from shujuchuli import OptimizedDataProcessor
from yinzifenxi import FactorAnalyzer
from yinzipingfen import FactorEvaluator
from baogaoshengcheng import ReportGenerator
from keshihua import FactorVisualizer
from zhukongzhiqi import FactorAnalysisController

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="因子分析系统")
    
    # 配置参数
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--env", type=str, default="production",
                        help="运行环境 (development, testing, production)")
    
    # 数据参数
    parser.add_argument("--data-config", type=str, default=None,
                        help="数据配置文件路径")
    parser.add_argument("--factor-dir", type=str, default=None,
                        help="因子数据目录")
    parser.add_argument("--return-file", type=str, default=None,
                        help="收益数据文件路径")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--report-format", type=str, default="html",
                        choices=["html", "pdf", "excel"],
                        help="报告格式")
    
    # 可视化参数
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--viz-backend", type=str, default="matplotlib",
                        choices=["matplotlib", "plotly", "seaborn"],
                        help="可视化后端")
    
    # 其他参数
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")
    parser.add_argument("--load-results", type=str, default=None,
                        help="加载已有分析结果")
    parser.add_argument("--save-results", action="store_true",
                        help="保存分析结果")
    
    return parser.parse_args()

def create_data_config(args) -> Dict[str, Any]:
    """创建数据配置"""
    data_config = {}
    
    # 如果提供了数据配置文件，直接加载
    if args.data_config and os.path.exists(args.data_config):
        with open(args.data_config, 'r', encoding='utf-8') as f:
            if args.data_config.endswith('.json'):
                data_config = json.load(f)
            else:
                import yaml
                data_config = yaml.safe_load(f)
        return data_config
    
    # 否则根据命令行参数构建配置
    data_config = {
        'factors': {},
        'returns': {}
    }
    
    # 处理因子数据
    if args.factor_dir and os.path.exists(args.factor_dir):
        factor_files = [f for f in os.listdir(args.factor_dir) 
                       if f.endswith(('.csv', '.xlsx', '.parquet'))]
        
        for file in factor_files:
            factor_name = os.path.splitext(file)[0]
            file_path = os.path.join(args.factor_dir, file)
            file_type = os.path.splitext(file)[1][1:]  # 去掉点
            
            data_config['factors'][factor_name] = {
                'path': file_path,
                'type': file_type
            }
    
    # 处理收益数据
    if args.return_file and os.path.exists(args.return_file):
        file_type = os.path.splitext(args.return_file)[1][1:]  # 去掉点
        data_config['returns'] = {
            'path': args.return_file,
            'type': file_type
        }
    
    return data_config

def ensure_output_directory(output_dir: str):
    """确保输出目录存在"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = ['charts', 'reports', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def run_factor_analysis(args) -> Dict[str, Any]:
    """运行因子分析"""
    # 创建控制器
    controller = FactorAnalysisController(args.config, args.env)
    
    # 加载已有结果（如果指定）
    if args.load_results and os.path.exists(args.load_results):
        controller.load_analysis_results(args.load_results)
        logging.info(f"已加载分析结果: {args.load_results}")
        return controller.get_analysis_results()
    
    # 创建数据配置
    data_config = create_data_config(args)
    
    if not data_config['factors']:
        logging.error("未找到因子数据，请检查 --factor-dir 或 --data-config 参数")
        sys.exit(1)
    
    # 运行因子分析
    results = controller.run_factor_analysis(data_config)
    
    # 保存结果（如果指定）
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, 'data', f'factor_analysis_results_{timestamp}.json')
        controller.save_analysis_results(results_path)
        logging.info(f"分析结果已保存到: {results_path}")
    
    return results

def generate_visualizations(results: Dict[str, Any], args):
    """生成可视化图表"""
    # 创建可视化器
    config = ConfigManager(args.config, args.env)
    cache = CacheManager(
        config.get('cache.memory_size', 100),
        config.get('cache.disk_size', 1000)
    )
    error_handler = ErrorHandler(config)
    
    visualizer = FactorVisualizer(config, cache, error_handler)
    
    # 设置可视化后端
    if args.viz_backend:
        config.set('visualization.backend', args.viz_backend)
    
    # 确保输出目录存在
    charts_dir = os.path.join(args.output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 生成各种图表
    factor_data = results.get('factor_data', {})
    analysis_results = results.get('analysis_results', {})
    factor_scores = results.get('factor_scores', {})
    factor_rankings = results.get('factor_rankings')
    
    # 因子分布图
    for factor_name, data in factor_data.items():
        try:
            dist_path = visualizer.plot_factor_distribution(
                data, factor_name, 
                backend=args.viz_backend,
                save_path=os.path.join(charts_dir, f"{factor_name}_distribution.{args.viz_backend}")
            )
            logging.info(f"因子分布图已保存: {dist_path}")
        except Exception as e:
            logging.error(f"生成因子分布图失败 {factor_name}: {e}")
    
    # IC序列图
    for factor_name, analysis in analysis_results.items():
        if 'ic_test' in analysis and 'ic_series' in analysis['ic_test']:
            try:
                ic_path = visualizer.plot_ic_series(
                    analysis['ic_test']['ic_series'], factor_name,
                    backend=args.viz_backend,
                    save_path=os.path.join(charts_dir, f"{factor_name}_ic_series.{args.viz_backend}")
                )
                logging.info(f"IC序列图已保存: {ic_path}")
            except Exception as e:
                logging.error(f"生成IC序列图失败 {factor_name}: {e}")
    
    # 分组收益图
    for factor_name, analysis in analysis_results.items():
        if 'group_return_test' in analysis and 'group_returns' in analysis['group_return_test']:
            try:
                group_path = visualizer.plot_group_returns(
                    analysis['group_return_test']['group_returns'], factor_name,
                    backend=args.viz_backend,
                    save_path=os.path.join(charts_dir, f"{factor_name}_group_returns.{args.viz_backend}")
                )
                logging.info(f"分组收益图已保存: {group_path}")
            except Exception as e:
                logging.error(f"生成分组收益图失败 {factor_name}: {e}")
    
    # 因子相关性热力图
    if factor_data:
        try:
            all_factor_data = pd.concat(factor_data.values(), axis=1)
            corr_path = visualizer.plot_factor_correlation(
                all_factor_data,
                backend=args.viz_backend,
                save_path=os.path.join(charts_dir, f"factor_correlation.{args.viz_backend}")
            )
            logging.info(f"因子相关性图已保存: {corr_path}")
        except Exception as e:
            logging.error(f"生成因子相关性图失败: {e}")
    
    # 因子表现图
    if factor_scores:
        try:
            # 将因子评分转换为DataFrame
            perf_data = []
            for factor_name, scores in factor_scores.items():
                if isinstance(scores, dict) and 'overall_score' in scores:
                    perf_data.append({
                        'factor': factor_name,
                        'score': scores['overall_score']
                    })
                elif isinstance(scores, (int, float)):
                    perf_data.append({
                        'factor': factor_name,
                        'score': scores
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_path = visualizer.plot_factor_performance(
                    perf_df,
                    backend=args.viz_backend,
                    save_path=os.path.join(charts_dir, f"factor_performance.{args.viz_backend}")
                )
                logging.info(f"因子表现图已保存: {perf_path}")
        except Exception as e:
            logging.error(f"生成因子表现图失败: {e}")
    
    # 因子排名图
    if factor_rankings is not None:
        try:
            rank_path = visualizer.plot_factor_ranking(
                factor_rankings,
                backend=args.viz_backend,
                save_path=os.path.join(charts_dir, f"factor_ranking.{args.viz_backend}")
            )
            logging.info(f"因子排名图已保存: {rank_path}")
        except Exception as e:
            logging.error(f"生成因子排名图失败: {e}")
    
    # 因子分析仪表板
    try:
        # 准备仪表板数据
        dashboard_factor_data = {name: data for name, data in factor_data.items()}
        
        dashboard_ic_data = {}
        for factor_name, analysis in analysis_results.items():
            if 'ic_test' in analysis and 'ic_series' in analysis['ic_test']:
                dashboard_ic_data[factor_name] = analysis['ic_test']['ic_series']
        
        dashboard_group_return_data = {}
        for factor_name, analysis in analysis_results.items():
            if 'group_return_test' in analysis and 'group_returns' in analysis['group_return_test']:
                dashboard_group_return_data[factor_name] = analysis['group_return_test']['group_returns']
        
        # 因子表现数据
        perf_data = []
        for factor_name, scores in factor_scores.items():
            if isinstance(scores, dict) and 'overall_score' in scores:
                perf_data.append({
                    'factor': factor_name,
                    'score': scores['overall_score']
                })
            elif isinstance(scores, (int, float)):
                perf_data.append({
                    'factor': factor_name,
                    'score': scores
                })
        
        dashboard_factor_performance = pd.DataFrame(perf_data) if perf_data else pd.DataFrame()
        
        # 创建仪表板
        if dashboard_factor_data and dashboard_ic_data and dashboard_group_return_data:
            dashboard_path = visualizer.create_factor_analysis_dashboard(
                dashboard_factor_data,
                dashboard_ic_data,
                dashboard_group_return_data,
                dashboard_factor_performance,
                factor_rankings,
                backend=args.viz_backend,
                save_path=os.path.join(charts_dir, f"factor_analysis_dashboard.{args.viz_backend}")
            )
            logging.info(f"因子分析仪表板已保存: {dashboard_path}")
    except Exception as e:
        logging.error(f"生成因子分析仪表板失败: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    # 确保输出目录存在
    ensure_output_directory(args.output_dir)
    
    # 运行因子分析
    logging.info("开始运行因子分析...")
    results = run_factor_analysis(args)
    logging.info("因子分析完成")
    
    # 生成可视化图表（如果指定）
    if args.visualize:
        logging.info("开始生成可视化图表...")
        generate_visualizations(results, args)
        logging.info("可视化图表生成完成")
    
    # 打印摘要信息
    factor_count = len(results.get('factor_data', {}))
    has_return_data = results.get('return_data') is not None
    report_path = results.get('report_path')
    
    print("\n=== 因子分析摘要 ===")
    print(f"分析因子数量: {factor_count}")
    print(f"包含收益数据: {'是' if has_return_data else '否'}")
    if report_path:
        print(f"报告路径: {report_path}")
    print(f"输出目录: {args.output_dir}")
    print("==================")

if __name__ == "__main__":
    main()