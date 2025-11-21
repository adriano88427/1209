"""
优化后的因子分析模块
支持并行计算、缓存优化和高级分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import os
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import multiprocessing as mp
from functools import partial
import pickle
import hashlib
import time
from scipy import stats
from statsmodels.api import OLS
from statsmodels.regression.rolling import RollingOLS

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataProcessingError, exception_handler, cache_result

class FactorTest(ABC):
    """抽象因子测试类"""
    
    @abstractmethod
    def test(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Dict[str, Any]:
        """执行因子测试"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取测试名称"""
        pass

class ICTest(FactorTest):
    """IC测试"""
    
    def __init__(self, method: str = 'pearson', min_periods: int = 20):
        self.method = method
        self.min_periods = min_periods
    
    def test(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Dict[str, Any]:
        """执行IC测试"""
        try:
            # 确保数据对齐
            common_index = factor_data.index.intersection(return_data.index)
            if len(common_index) < self.min_periods:
                raise AnalysisError(f"数据点不足，需要至少{self.min_periods}个点，实际只有{len(common_index)}个")
            
            factor_aligned = factor_data.loc[common_index]
            return_aligned = return_data.loc[common_index]
            
            # 计算IC值
            if self.method == 'pearson':
                ic_values = factor_aligned.apply(lambda x: x.corr(return_aligned.iloc[:, 0], method='pearson'), axis=1)
            elif self.method == 'spearman':
                ic_values = factor_aligned.apply(lambda x: x.corr(return_aligned.iloc[:, 0], method='spearman'), axis=1)
            else:
                raise AnalysisError(f"不支持的IC计算方法: {self.method}")
            
            # 计算IC统计量
            ic_mean = ic_values.mean()
            ic_std = ic_values.std()
            ic_ir = ic_mean / ic_std if ic_std != 0 else 0
            ic_skew = ic_values.skew()
            ic_kurt = ic_values.kurtosis()
            
            # 计算IC绝对值
            ic_abs_mean = ic_values.abs().mean()
            ic_abs_std = ic_values.abs().std()
            
            # 计算IC比率
            ic_positive_ratio = (ic_values > 0).mean()
            ic_negative_ratio = (ic_values < 0).mean()
            
            # 计算t检验
            t_stat, p_value = stats.ttest_1samp(ic_values.dropna(), 0)
            
            # 计算胜率
            win_rate = (ic_values > 0).mean()
            
            # 计算月度IC
            monthly_ic = ic_values.resample('M').mean()
            monthly_ic_std = monthly_ic.std()
            monthly_ic_ir = monthly_ic.mean() / monthly_ic_std if monthly_ic_std != 0 else 0
            
            return {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_skew': ic_skew,
                'ic_kurt': ic_kurt,
                'ic_abs_mean': ic_abs_mean,
                'ic_abs_std': ic_abs_std,
                'ic_positive_ratio': ic_positive_ratio,
                'ic_negative_ratio': ic_negative_ratio,
                't_stat': t_stat,
                'p_value': p_value,
                'win_rate': win_rate,
                'monthly_ic_mean': monthly_ic.mean(),
                'monthly_ic_std': monthly_ic_std,
                'monthly_ic_ir': monthly_ic_ir,
                'ic_values': ic_values
            }
        except Exception as e:
            raise AnalysisError(f"IC测试失败: {e}")
    
    def get_name(self) -> str:
        """获取测试名称"""
        return f"IC测试({self.method})"

class GroupReturnTest(FactorTest):
    """分组收益测试"""
    
    def __init__(self, num_groups: int = 5, method: str = 'quantile'):
        self.num_groups = num_groups
        self.method = method
    
    def test(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Dict[str, Any]:
        """执行分组收益测试"""
        try:
            # 确保数据对齐
            common_index = factor_data.index.intersection(return_data.index)
            if len(common_index) < 2:
                raise AnalysisError(f"数据点不足，需要至少2个点，实际只有{len(common_index)}个")
            
            factor_aligned = factor_data.loc[common_index]
            return_aligned = return_data.loc[common_index]
            
            # 初始化结果
            group_returns = pd.DataFrame(index=common_index, columns=range(1, self.num_groups + 1))
            group_cumulative_returns = pd.DataFrame(index=common_index, columns=range(1, self.num_groups + 1))
            
            # 按日期分组计算
            for date in common_index:
                factor_date = factor_aligned.loc[date]
                return_date = return_aligned.loc[date]
                
                # 去除缺失值
                valid_mask = factor_date.notna() & return_date.notna()
                factor_valid = factor_date[valid_mask]
                return_valid = return_date[valid_mask]
                
                if len(factor_valid) < self.num_groups:
                    continue
                
                # 分组
                if self.method == 'quantile':
                    # 等频分组
                    try:
                        groups = pd.qcut(factor_valid, self.num_groups, labels=False, duplicates='drop')
                    except ValueError:
                        # 如果无法等频分组，使用等距分组
                        groups = pd.cut(factor_valid, self.num_groups, labels=False)
                elif self.method == 'equal_width':
                    # 等距分组
                    groups = pd.cut(factor_valid, self.num_groups, labels=False)
                else:
                    raise AnalysisError(f"不支持的分组方法: {self.method}")
                
                # 计算每组收益
                for group_id in range(self.num_groups):
                    mask = groups == group_id
                    if mask.sum() > 0:
                        group_return = return_valid[mask].mean()
                        group_returns.loc[date, group_id + 1] = group_return
            
            # 计算累计收益
            group_cumulative_returns = (1 + group_returns).cumprod()
            
            # 计算多空组合收益
            long_short_returns = group_returns.iloc[:, self.num_groups - 1] - group_returns.iloc[:, 0]
            long_short_cumulative_returns = (1 + long_short_returns).cumprod()
            
            # 计算统计量
            group_mean_returns = group_returns.mean()
            group_std_returns = group_returns.std()
            group_sharpe_ratios = group_mean_returns / group_std_returns
            
            # 计算年化收益和波动率
            trading_days_per_year = 252
            group_annual_returns = group_mean_returns * trading_days_per_year
            group_annual_volatility = group_std_returns * np.sqrt(trading_days_per_year)
            group_annual_sharpe = group_annual_returns / group_annual_volatility
            
            # 计算最大回撤
            group_max_drawdowns = {}
            for group_id in range(1, self.num_groups + 1):
                cumulative = group_cumulative_returns[group_id]
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                group_max_drawdowns[group_id] = drawdown.min()
            
            # 计算多空组合统计量
            long_short_mean = long_short_returns.mean()
            long_short_std = long_short_returns.std()
            long_short_sharpe = long_short_mean / long_short_std if long_short_std != 0 else 0
            long_short_annual_return = long_short_mean * trading_days_per_year
            long_short_annual_volatility = long_short_std * np.sqrt(trading_days_per_year)
            long_short_annual_sharpe = long_short_annual_return / long_short_annual_volatility
            
            # 计算多空组合最大回撤
            long_short_running_max = long_short_cumulative_returns.expanding().max()
            long_short_drawdown = (long_short_cumulative_returns - long_short_running_max) / long_short_running_max
            long_short_max_drawdown = long_short_drawdown.min()
            
            return {
                'group_returns': group_returns,
                'group_cumulative_returns': group_cumulative_returns,
                'group_mean_returns': group_mean_returns,
                'group_std_returns': group_std_returns,
                'group_sharpe_ratios': group_sharpe_ratios,
                'group_annual_returns': group_annual_returns,
                'group_annual_volatility': group_annual_volatility,
                'group_annual_sharpe': group_annual_sharpe,
                'group_max_drawdowns': group_max_drawdowns,
                'long_short_returns': long_short_returns,
                'long_short_cumulative_returns': long_short_cumulative_returns,
                'long_short_mean': long_short_mean,
                'long_short_std': long_short_std,
                'long_short_sharpe': long_short_sharpe,
                'long_short_annual_return': long_short_annual_return,
                'long_short_annual_volatility': long_short_annual_volatility,
                'long_short_annual_sharpe': long_short_annual_sharpe,
                'long_short_max_drawdown': long_short_max_drawdown
            }
        except Exception as e:
            raise AnalysisError(f"分组收益测试失败: {e}")
    
    def get_name(self) -> str:
        """获取测试名称"""
        return f"分组收益测试({self.num_groups}组,{self.method})"

class TurnoverTest(FactorTest):
    """换手率测试"""
    
    def __init__(self, num_groups: int = 5, method: str = 'quantile'):
        self.num_groups = num_groups
        self.method = method
    
    def test(self, factor_data: pd.DataFrame, return_data: pd.DataFrame = None) -> Dict[str, Any]:
        """执行换手率测试"""
        try:
            # 只需要因子数据
            dates = sorted(factor_data.index.unique())
            
            # 初始化结果
            group_turnovers = pd.DataFrame(index=dates[1:], columns=range(1, self.num_groups + 1))
            
            # 计算每日分组
            daily_groups = {}
            for date in dates:
                factor_date = factor_data.loc[date]
                
                # 去除缺失值
                factor_valid = factor_date.dropna()
                
                if len(factor_valid) < self.num_groups:
                    daily_groups[date] = pd.Series(index=factor_valid.index, dtype=int)
                    continue
                
                # 分组
                if self.method == 'quantile':
                    # 等频分组
                    try:
                        groups = pd.qcut(factor_valid, self.num_groups, labels=False, duplicates='drop')
                    except ValueError:
                        # 如果无法等频分组，使用等距分组
                        groups = pd.cut(factor_valid, self.num_groups, labels=False)
                elif self.method == 'equal_width':
                    # 等距分组
                    groups = pd.cut(factor_valid, self.num_groups, labels=False)
                else:
                    raise AnalysisError(f"不支持的分组方法: {self.method}")
                
                daily_groups[date] = groups
            
            # 计算换手率
            for i, date in enumerate(dates[1:]):
                prev_date = dates[i]
                
                prev_groups = daily_groups[prev_date]
                curr_groups = daily_groups[date]
                
                # 找出共同股票
                common_stocks = prev_groups.index.intersection(curr_groups.index)
                
                if len(common_stocks) == 0:
                    continue
                
                prev_groups_common = prev_groups.loc[common_stocks]
                curr_groups_common = curr_groups.loc[common_stocks]
                
                # 计算每组换手率
                for group_id in range(self.num_groups):
                    # 上一期在该组的股票
                    prev_mask = prev_groups_common == group_id
                    prev_stocks = prev_groups_common[prev_mask].index
                    
                    # 本期在该组的股票
                    curr_mask = curr_groups_common == group_id
                    curr_stocks = curr_groups_common[curr_mask].index
                    
                    # 计算换手率
                    if len(prev_stocks) > 0:
                        # 新进入的股票
                        new_stocks = curr_stocks.difference(prev_stocks)
                        # 离开的股票
                        out_stocks = prev_stocks.difference(curr_stocks)
                        
                        # 换手率 = (新进入 + 离开) / 2 / 上一期股票数
                        turnover = (len(new_stocks) + len(out_stocks)) / 2 / len(prev_stocks)
                        group_turnovers.loc[date, group_id + 1] = turnover
            
            # 计算平均换手率
            mean_turnover = group_turnovers.mean()
            
            # 计算年化换手率
            trading_days_per_year = 252
            annual_turnover = mean_turnover * trading_days_per_year
            
            # 计算换手率标准差
            std_turnover = group_turnovers.std()
            
            return {
                'group_turnovers': group_turnovers,
                'mean_turnover': mean_turnover,
                'annual_turnover': annual_turnover,
                'std_turnover': std_turnover
            }
        except Exception as e:
            raise AnalysisError(f"换手率测试失败: {e}")
    
    def get_name(self) -> str:
        """获取测试名称"""
        return f"换手率测试({self.num_groups}组,{self.method})"

class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self, config_manager: ConfigManager, cache_manager: CacheManager, 
                 error_handler: ErrorHandler):
        self.config = config_manager
        self.cache = cache_manager
        self.error_handler = error_handler
        
        # 初始化测试器
        self.tests = {}
        self._initialize_tests()
        
        # 注册配置监听器
        self.config.watch('analysis.ic_method', self._update_ic_method)
        self.config.watch('analysis.num_groups', self._update_num_groups)
    
    def _initialize_tests(self):
        """初始化测试器"""
        # IC测试
        ic_method = self.config.get('analysis.ic_method', 'pearson')
        self.tests['ic'] = ICTest(method=ic_method)
        
        # 分组收益测试
        num_groups = self.config.get('analysis.num_groups', 5)
        self.tests['group_return'] = GroupReturnTest(num_groups=num_groups)
        
        # 换手率测试
        self.tests['turnover'] = TurnoverTest(num_groups=num_groups)
    
    def _update_ic_method(self, key: str, new_value: Any, old_value: Any):
        """更新IC计算方法"""
        self.tests['ic'] = ICTest(method=new_value)
        logging.info(f"更新IC计算方法: {old_value} -> {new_value}")
    
    def _update_num_groups(self, key: str, new_value: Any, old_value: Any):
        """更新分组数"""
        self.tests['group_return'] = GroupReturnTest(num_groups=new_value)
        self.tests['turnover'] = TurnoverTest(num_groups=new_value)
        logging.info(f"更新分组数: {old_value} -> {new_value}")
    
    @exception_handler()
    def analyze_factor(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                      tests: List[str] = None) -> Dict[str, Any]:
        """分析因子"""
        if tests is None:
            tests = list(self.tests.keys())
        
        results = {}
        
        for test_name in tests:
            if test_name not in self.tests:
                logging.warning(f"未知的测试: {test_name}")
                continue
            
            # 尝试从缓存加载
            cache_key = f"factor_test_{test_name}_{hash(str(factor_data.values.tobytes()))}_{hash(str(return_data.values.tobytes()))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logging.info(f"从缓存加载{test_name}测试结果")
                results[test_name] = cached_result
                continue
            
            # 执行测试
            test = self.tests[test_name]
            try:
                if test_name == 'turnover':
                    # 换手率测试不需要收益数据
                    result = test.test(factor_data)
                else:
                    result = test.test(factor_data, return_data)
                
                results[test_name] = result
                
                # 缓存结果
                self.cache.set(cache_key, result)
                
                logging.info(f"完成{test.get_name()}")
            except Exception as e:
                logging.error(f"{test.get_name()}失败: {e}")
                self.error_handler.handle_error(e, f"{test.get_name()}失败")
        
        return results
    
    @exception_handler()
    def batch_analyze_factors(self, factor_dict: Dict[str, pd.DataFrame], 
                             return_data: pd.DataFrame, 
                             tests: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """批量分析因子"""
        results = {}
        
        for factor_name, factor_data in factor_dict.items():
            logging.info(f"分析因子: {factor_name}")
            try:
                factor_results = self.analyze_factor(factor_data, return_data, tests)
                results[factor_name] = factor_results
            except Exception as e:
                logging.error(f"分析因子{factor_name}失败: {e}")
                self.error_handler.handle_error(e, f"分析因子{factor_name}失败")
        
        return results
    
    def add_test(self, name: str, test: FactorTest):
        """添加测试"""
        self.tests[name] = test
        logging.info(f"添加测试: {name}")
    
    def remove_test(self, name: str):
        """移除测试"""
        if name in self.tests:
            del self.tests[name]
            logging.info(f"移除测试: {name}")
    
    def get_test_names(self) -> List[str]:
        """获取测试名称列表"""
        return list(self.tests.keys())

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
        # 评估权重
        self.weights = self.config.get('evaluation.weights', {
            'ic_mean': 0.3,
            'ic_ir': 0.2,
            'long_short_sharpe': 0.3,
            'long_short_max_drawdown': 0.1,
            'turnover': 0.1
        })
    
    def evaluate_factor(self, factor_name: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估因子"""
        try:
            # 提取关键指标
            ic_results = analysis_results.get('ic', {})
            group_return_results = analysis_results.get('group_return', {})
            turnover_results = analysis_results.get('turnover', {})
            
            # 计算得分
            scores = {}
            
            # IC均值得分
            ic_mean = ic_results.get('ic_mean', 0)
            scores['ic_mean_score'] = self._normalize_score(ic_mean, 0, 0.1)
            
            # IC IR得分
            ic_ir = ic_results.get('ic_ir', 0)
            scores['ic_ir_score'] = self._normalize_score(ic_ir, 0, 1)
            
            # 多空夏普比率得分
            long_short_sharpe = group_return_results.get('long_short_annual_sharpe', 0)
            scores['long_short_sharpe_score'] = self._normalize_score(long_short_sharpe, 0, 2)
            
            # 多空最大回撤得分
            long_short_max_drawdown = group_return_results.get('long_short_max_drawdown', 0)
            scores['long_short_max_drawdown_score'] = self._normalize_score(-long_short_max_drawdown, -0.5, 0)
            
            # 换手率得分（越低越好）
            annual_turnover = turnover_results.get('annual_turnover', pd.Series()).mean()
            scores['turnover_score'] = self._normalize_score(-annual_turnover, -10, 0)
            
            # 计算综合得分
            total_score = 0
            for metric, weight in self.weights.items():
                score_key = f"{metric}_score"
                if score_key in scores:
                    total_score += scores[score_key] * weight
            
            scores['total_score'] = total_score
            
            # 评级
            rating = self._get_rating(total_score)
            
            return {
                'factor_name': factor_name,
                'scores': scores,
                'total_score': total_score,
                'rating': rating,
                'metrics': {
                    'ic_mean': ic_mean,
                    'ic_ir': ic_ir,
                    'long_short_annual_sharpe': long_short_sharpe,
                    'long_short_max_drawdown': long_short_max_drawdown,
                    'annual_turnover': annual_turnover
                }
            }
        except Exception as e:
            raise AnalysisError(f"评估因子{factor_name}失败: {e}")
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """标准化得分"""
        if max_val <= min_val:
            return 0.5
        
        # 限制在范围内
        value = max(min_val, min(max_val, value))
        
        # 标准化到0-1
        return (value - min_val) / (max_val - min_val)
    
    def _get_rating(self, score: float) -> str:
        """获取评级"""
        if score >= 0.8:
            return 'A+'
        elif score >= 0.7:
            return 'A'
        elif score >= 0.6:
            return 'A-'
        elif score >= 0.5:
            return 'B+'
        elif score >= 0.4:
            return 'B'
        elif score >= 0.3:
            return 'B-'
        elif score >= 0.2:
            return 'C+'
        elif score >= 0.1:
            return 'C'
        else:
            return 'D'
    
    def compare_factors(self, factor_evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """比较因子"""
        # 创建比较表
        comparison_data = []
        
        for factor_name, evaluation in factor_evaluations.items():
            metrics = evaluation.get('metrics', {})
            scores = evaluation.get('scores', {})
            
            row = {
                'Factor': factor_name,
                'Total Score': evaluation.get('total_score', 0),
                'Rating': evaluation.get('rating', 'D'),
                'IC Mean': metrics.get('ic_mean', 0),
                'IC IR': metrics.get('ic_ir', 0),
                'Long-Short Sharpe': metrics.get('long_short_annual_sharpe', 0),
                'Long-Short Max DD': metrics.get('long_short_max_drawdown', 0),
                'Annual Turnover': metrics.get('annual_turnover', 0)
            }
            
            comparison_data.append(row)
        
        # 创建DataFrame并排序
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Total Score', ascending=False)
        
        return comparison_df

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 初始化组件
    config_manager = ConfigManager("config.yaml", "production")
    cache_manager = CacheManager("cache", memory_size=50, disk_size=500)
    error_handler = ErrorHandler(config_manager)
    
    # 创建因子分析器
    factor_analyzer = FactorAnalyzer(config_manager, cache_manager, error_handler)
    
    # 创建因子评估器
    factor_evaluator = FactorEvaluator(config_manager)
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    stocks = [f'stock_{i}' for i in range(100)]
    
    # 创建因子数据
    factor_data = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)),
        index=dates,
        columns=stocks
    )
    
    # 创建收益数据
    return_data = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)) * 0.01,
        index=dates,
        columns=stocks
    )
    
    # 分析因子
    try:
        results = factor_analyzer.analyze_factor(factor_data, return_data)
        print("因子分析结果:")
        for test_name, result in results.items():
            print(f"{test_name}: {result}")
        
        # 评估因子
        evaluation = factor_evaluator.evaluate_factor("test_factor", results)
        print("\n因子评估结果:")
        print(f"综合得分: {evaluation['total_score']:.4f}")
        print(f"评级: {evaluation['rating']}")
        
    except Exception as e:
        print(f"分析失败: {e}")