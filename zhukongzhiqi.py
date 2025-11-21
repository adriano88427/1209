"""
优化后的主控制器模块
负责协调各模块的工作
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataProcessingError, exception_handler, cache_result

# 导入各功能模块
from shujuchuli import OptimizedDataProcessor
from yinzifenxi import FactorAnalyzer
from yinzipingfen import FactorEvaluator
from baogaoshengcheng import ReportGenerator
from keshihua import FactorVisualizer
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import multiprocessing as mp
from functools import partial
import hashlib
import pickle
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 导入系统组件
# 删除这些旧的导入语句
# from enhanced_factor_analysis_system import (
#     ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError,
#     AnalysisError, exception_handler, cache_result, DAGTaskScheduler, TaskStatus
# )

# 导入各模块
# from optimized_data_processor import OptimizedDataProcessor
# from optimized_factor_analyzer import OptimizedFactorAnalyzer
# from optimized_factor_scorer import FactorScorer, FactorRanker
# from optimized_report_generator import ReportGenerator

class AnalysisWorkflow:
    """分析工作流"""
    
    def __init__(self, name: str, config_manager: ConfigManager, 
                 task_scheduler=None):
        self.name = name
        self.config = config_manager
        self.scheduler = task_scheduler
        self.tasks = {}
        self.dependencies = {}
    
    def add_task(self, task_id: str, task_func: Callable, 
                 dependencies: List[str] = None, **kwargs):
        """添加任务"""
        self.tasks[task_id] = {
            'func': task_func,
            'kwargs': kwargs
        }
        self.dependencies[task_id] = dependencies or []
    
    def execute(self, **shared_data) -> Dict[str, Any]:
        """执行工作流"""
        # 简化版工作流执行，按依赖顺序执行任务
        results = {}
        
        # 获取所有任务
        all_tasks = set(self.tasks.keys())
        executed_tasks = set()
        
        # 按依赖顺序执行任务
        while len(executed_tasks) < len(all_tasks):
            # 找出可以执行的任务（所有依赖都已执行）
            ready_tasks = []
            for task_id in all_tasks - executed_tasks:
                dependencies = self.dependencies[task_id]
                if all(dep in executed_tasks for dep in dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                raise ValueError("Circular dependency detected in workflow")
            
            # 执行准备好的任务
            for task_id in ready_tasks:
                task_info = self.tasks[task_id]
                # 合并共享数据和之前的结果
                task_data = {**shared_data, **results}
                # 执行任务
                result = task_info['func'](**task_data)
                results[task_id] = result
                executed_tasks.add(task_id)
        
        return results

class FactorAnalysisController:
    """因子分析控制器"""
    
    def __init__(self, config_path: str = "config.yaml", 
                 environment: str = "production"):
        # 初始化核心组件
        self.config = ConfigManager(config_path, environment)
        self.cache = CacheManager(
            self.config.get('cache.memory_size', 100),
            self.config.get('cache.disk_size', 1000)
        )
        self.error_handler = ErrorHandler(self.config)
        
        # 初始化各模块
        self.data_processor = OptimizedDataProcessor(
            self.config, self.cache, self.error_handler
        )
        self.factor_analyzer = FactorAnalyzer(
            self.config, self.cache, self.error_handler
        )
        self.factor_evaluator = FactorEvaluator(
            self.config, self.cache, self.error_handler
        )
        self.report_generator = ReportGenerator(
            self.config, self.cache, self.error_handler
        )
        
        # 初始化工作流
        self.workflows = {}
        self._initialize_workflows()
        
        # 分析结果存储
        self.analysis_results = {}
    
    def _initialize_workflows(self):
        """初始化工作流"""
        # 因子分析工作流
        factor_analysis_workflow = AnalysisWorkflow(
            "factor_analysis", self.config, None  # 暂时移除scheduler参数
        )
        
        # 添加数据加载任务
        factor_analysis_workflow.add_task(
            "load_data",
            self._load_data_task
        )
        
        # 添加数据预处理任务
        factor_analysis_workflow.add_task(
            "preprocess_data",
            self._preprocess_data_task,
            dependencies=["load_data"]
        )
        
        # 添加因子分析任务
        factor_analysis_workflow.add_task(
            "analyze_factors",
            self._analyze_factors_task,
            dependencies=["preprocess_data"]
        )
        
        # 添加因子评分任务
        factor_analysis_workflow.add_task(
            "score_factors",
            self._score_factors_task,
            dependencies=["analyze_factors"]
        )
        
        # 添加因子排名任务
        factor_analysis_workflow.add_task(
            "rank_factors",
            self._rank_factors_task,
            dependencies=["score_factors"]
        )
        
        # 添加报告生成任务
        factor_analysis_workflow.add_task(
            "generate_report",
            self._generate_report_task,
            dependencies=["rank_factors"]
        )
        
        self.workflows["factor_analysis"] = factor_analysis_workflow
    
    @exception_handler()
    def _load_data_task(self, **kwargs) -> Dict[str, Any]:
        """数据加载任务"""
        data_config = kwargs.get('data_config', {})
        
        # 加载因子数据
        factor_data = {}
        for factor_name, factor_config in data_config.get('factors', {}).items():
            factor_path = factor_config.get('path')
            if factor_path and os.path.exists(factor_path):
                factor_data[factor_name] = self.data_processor.load_data(
                    factor_path, 
                    data_type=factor_config.get('type', 'csv')
                )
        
        # 加载收益数据
        return_config = data_config.get('returns', {})
        return_path = return_config.get('path')
        return_data = None
        if return_path and os.path.exists(return_path):
            return_data = self.data_processor.load_data(
                return_path,
                data_type=return_config.get('type', 'csv')
            )
        
        return {
            'factor_data': factor_data,
            'return_data': return_data
        }
    
    @exception_handler()
    def _preprocess_data_task(self, **kwargs) -> Dict[str, Any]:
        """数据预处理任务"""
        factor_data = kwargs.get('factor_data', {})
        return_data = kwargs.get('return_data')
        
        # 预处理因子数据
        processed_factor_data = {}
        for factor_name, data in factor_data.items():
            processed_factor_data[factor_name] = self.data_processor.preprocess_data(
                data,
                methods=self.config.get(f'preprocessing.{factor_name}', ['standardize'])
            )
        
        # 预处理收益数据
        processed_return_data = None
        if return_data is not None:
            processed_return_data = self.data_processor.preprocess_data(
                return_data,
                methods=self.config.get('preprocessing.returns', ['fill_na'])
            )
        
        return {
            'factor_data': processed_factor_data,
            'return_data': processed_return_data
        }
    
    @exception_handler()
    def _analyze_factors_task(self, **kwargs) -> Dict[str, Any]:
        """因子分析任务"""
        factor_data = kwargs.get('factor_data', {})
        return_data = kwargs.get('return_data')
        
        # 分析因子
        analysis_results = {}
        for factor_name, data in factor_data.items():
            analysis_results[factor_name] = self.factor_analyzer.analyze_factor(
                data,
                return_data=return_data,
                tests=self.config.get(f'analysis.{factor_name}.tests', ['ic_test', 'group_return_test'])
            )
        
        return {
            'analysis_results': analysis_results
        }
    
    @exception_handler()
    def _score_factors_task(self, **kwargs) -> Dict[str, Any]:
        """因子评分任务"""
        factor_data = kwargs.get('factor_data', {})
        return_data = kwargs.get('return_data')
        
        # 评分因子
        scoring_method = self.config.get('scoring.default_method', 'standard')
        factor_scores = {}
        
        # 使用FactorEvaluator进行因子评分
        for factor_name, data in factor_data.items():
            factor_scores[factor_name] = self.factor_evaluator.evaluate_factor(
                data,
                return_data=return_data,
                method=scoring_method
            )
        
        return {
            'factor_scores': factor_scores
        }
    
    @exception_handler()
    def _rank_factors_task(self, **kwargs) -> Dict[str, Any]:
        """因子排名任务"""
        factor_scores = kwargs.get('factor_scores', {})
        
        # 简化版因子排名，基于评分结果排序
        factor_rankings = sorted(
            factor_scores.items(),
            key=lambda x: x[1].get('overall_score', 0) if isinstance(x[1], dict) else x[1],
            reverse=True
        )
        
        # 转换为带排名的字典
        ranked_factors = {}
        for rank, (factor_name, score) in enumerate(factor_rankings, 1):
            ranked_factors[factor_name] = {
                'rank': rank,
                'score': score
            }
        
        return {
            'factor_rankings': ranked_factors
        }
    
    @exception_handler()
    def _generate_report_task(self, **kwargs) -> Dict[str, Any]:
        """报告生成任务"""
        factor_data = kwargs.get('factor_data', {})
        return_data = kwargs.get('return_data')
        factor_scores = kwargs.get('factor_scores', {})
        factor_rankings = kwargs.get('factor_rankings')
        
        # 生成报告
        report_format = self.config.get('reports.default_format', 'html')
        report_path = self.report_generator.generate_factor_analysis_report(
            factor_data,
            return_data,
            factor_scores,
            factor_rankings,
            format=report_format
        )
        
        return {
            'report_path': report_path
        }
    
    @exception_handler()
    def run_factor_analysis(self, data_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """运行因子分析"""
        if data_config is None:
            data_config = self.config.get('data', {})
        
        # 执行因子分析工作流
        workflow = self.workflows["factor_analysis"]
        results = workflow.execute(data_config=data_config)
        
        # 存储结果
        self.analysis_results = {
            'factor_data': results.get('preprocess_data', {}).get('factor_data', {}),
            'return_data': results.get('preprocess_data', {}).get('return_data'),
            'analysis_results': results.get('analyze_factors', {}).get('analysis_results', {}),
            'factor_scores': results.get('score_factors', {}).get('factor_scores', {}),
            'factor_rankings': results.get('rank_factors', {}).get('factor_rankings'),
            'report_path': results.get('generate_report', {}).get('report_path')
        }
        
        logging.info("因子分析完成")
        return self.analysis_results
    
    @exception_handler()
    def run_custom_workflow(self, workflow_name: str, **kwargs) -> Dict[str, Any]:
        """运行自定义工作流"""
        if workflow_name not in self.workflows:
            raise ValueError(f"未找到工作流: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        results = workflow.execute(**kwargs)
        
        logging.info(f"自定义工作流 {workflow_name} 完成")
        return results
    
    @exception_handler()
    def add_custom_workflow(self, workflow_name: str, workflow: AnalysisWorkflow):
        """添加自定义工作流"""
        self.workflows[workflow_name] = workflow
        logging.info(f"添加自定义工作流: {workflow_name}")
    
    @exception_handler()
    def get_analysis_results(self) -> Dict[str, Any]:
        """获取分析结果"""
        return self.analysis_results
    
    @exception_handler()
    def save_analysis_results(self, output_path: str = None):
        """保存分析结果"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"factor_analysis_results_{timestamp}.json"
        
        # 准备可序列化的结果
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        serializable_results[key][sub_key] = sub_value.to_dict()
                    elif isinstance(sub_value, pd.Series):
                        serializable_results[key][sub_key] = sub_value.to_dict()
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"分析结果已保存到: {output_path}")
        return output_path
    
    @exception_handler()
    def load_analysis_results(self, input_path: str):
        """加载分析结果"""
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
        
        # 转换回pandas对象
        analysis_results = {}
        for key, value in loaded_results.items():
            if isinstance(value, dict):
                # 检查是否是DataFrame格式
                if all(isinstance(k, str) for k in value.keys()) and all(isinstance(v, dict) for v in value.values()):
                    try:
                        # 尝试转换为DataFrame
                        df = pd.DataFrame.from_dict(value, orient='index')
                        if not df.empty:
                            analysis_results[key] = df
                            continue
                    except:
                        pass
                
                # 检查是否是Series格式
                if all(isinstance(k, str) for k in value.keys()) and all(isinstance(v, (int, float, str)) for v in value.values()):
                    try:
                        # 尝试转换为Series
                        series = pd.Series(value)
                        if not series.empty:
                            analysis_results[key] = series
                            continue
                    except:
                        pass
                
                # 处理嵌套字典
                nested_dict = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        # 尝试转换为DataFrame
                        try:
                            df = pd.DataFrame.from_dict(sub_value, orient='index')
                            if not df.empty:
                                nested_dict[sub_key] = df
                                continue
                        except:
                            pass
                        
                        # 尝试转换为Series
                        try:
                            series = pd.Series(sub_value)
                            if not series.empty:
                                nested_dict[sub_key] = series
                                continue
                        except:
                            pass
                    
                    nested_dict[sub_key] = sub_value
                
                analysis_results[key] = nested_dict
            else:
                analysis_results[key] = value
        
        self.analysis_results = analysis_results
        logging.info(f"分析结果已从 {input_path} 加载")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'config': self.config.get_all(),
            'cache': {
                'memory_size': self.cache.memory_cache_size,
                'disk_size': self.cache.disk_cache_size
            },
            'workflows': list(self.workflows.keys()),
            'analysis_results': {
                'factor_count': len(self.analysis_results.get('factor_data', {})),
                'has_return_data': self.analysis_results.get('return_data') is not None,
                'has_report': self.analysis_results.get('report_path') is not None
            }
        }

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制器
    controller = FactorAnalysisController("config.yaml", "production")
    
    # 模拟数据配置
    data_config = {
        'factors': {
            'factor_1': {
                'path': 'data/factor_1.csv',
                'type': 'csv'
            },
            'factor_2': {
                'path': 'data/factor_2.csv',
                'type': 'csv'
            }
        },
        'returns': {
            'path': 'data/returns.csv',
            'type': 'csv'
        }
    }
    
    # 运行因子分析
    try:
        results = controller.run_factor_analysis(data_config)
        print("因子分析结果:")
        print(f"因子数量: {len(results.get('factor_data', {}))}")
        print(f"报告路径: {results.get('report_path')}")
        
        # 保存分析结果
        results_path = controller.save_analysis_results()
        print(f"分析结果已保存到: {results_path}")
        
        # 获取系统状态
        status = controller.get_status()
        print("系统状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"因子分析失败: {e}")