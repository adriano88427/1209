"""
优化后的因子评分模块
支持多种评分方法和动态权重调整
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataProcessingError, exception_handler, cache_result

class ScoringMethod(ABC):
    """抽象评分方法"""
    
    @abstractmethod
    def score(self, factor_data: pd.DataFrame, return_data: pd.DataFrame = None) -> pd.Series:
        """计算因子得分"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取方法名称"""
        pass

class StandardScalerScoring(ScoringMethod):
    """标准化评分方法"""
    
    def __init__(self, method: str = 'zscore'):
        self.method = method
    
    def score(self, factor_data: pd.DataFrame, return_data: pd.DataFrame = None) -> pd.Series:
        """计算标准化得分"""
        if self.method == 'zscore':
            # Z-score标准化
            return factor_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        elif self.method == 'minmax':
            # 最小-最大标准化
            return factor_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
        elif self.method == 'rank':
            # 排名标准化
            return factor_data.apply(lambda x: x.rank(pct=True), axis=1)
        else:
            raise AnalysisError(f"不支持的标准化方法: {self.method}")
    
    def get_name(self) -> str:
        """获取方法名称"""
        return f"标准化评分({self.method})"

class DecileScoring(ScoringMethod):
    """十分位数评分方法"""
    
    def __init__(self, num_groups: int = 10):
        self.num_groups = num_groups
    
    def score(self, factor_data: pd.DataFrame, return_data: pd.DataFrame = None) -> pd.Series:
        """计算十分位数得分"""
        return factor_data.apply(lambda x: pd.qcut(x, self.num_groups, labels=False, duplicates='drop') + 1, axis=1)
    
    def get_name(self) -> str:
        """获取方法名称"""
        return f"十分位数评分({self.num_groups}组)"

class ICScoreScoring(ScoringMethod):
    """IC得分评分方法"""
    
    def __init__(self, window: int = 20, method: str = 'pearson'):
        self.window = window
        self.method = method
    
    def score(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> pd.Series:
        """计算IC得分"""
        if return_data is None:
            raise AnalysisError("IC评分需要收益数据")
        
        # 确保数据对齐
        common_index = factor_data.index.intersection(return_data.index)
        factor_aligned = factor_data.loc[common_index]
        return_aligned = return_data.loc[common_index]
        
        # 计算滚动IC
        ic_scores = pd.DataFrame(index=factor_aligned.index, columns=factor_aligned.columns)
        
        for i in range(self.window - 1, len(factor_aligned)):
            window_factor = factor_aligned.iloc[i - self.window + 1:i + 1]
            window_return = return_aligned.iloc[i - self.window + 1:i + 1]
            
            # 计算每只股票的IC
            for stock in factor_aligned.columns:
                if stock in window_factor.columns and stock in window_return.columns:
                    factor_stock = window_factor[stock].dropna()
                    return_stock = window_return[stock].dropna()
                    
                    common_dates = factor_stock.index.intersection(return_stock.index)
                    if len(common_dates) >= 3:  # 至少需要3个点计算相关性
                        factor_common = factor_stock.loc[common_dates]
                        return_common = return_stock.loc[common_dates]
                        
                        if self.method == 'pearson':
                            ic = factor_common.corr(return_common, method='pearson')
                        elif self.method == 'spearman':
                            ic = factor_common.corr(return_common, method='spearman')
                        else:
                            raise AnalysisError(f"不支持的IC计算方法: {self.method}")
                        
                        ic_scores.iloc[i, ic_scores.columns.get_loc(stock)] = ic
        
        # 标准化IC得分
        return ic_scores.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def get_name(self) -> str:
        """获取方法名称"""
        return f"IC评分(窗口={self.window},方法={self.method})"

class InformationRatioScoring(ScoringMethod):
    """信息比率评分方法"""
    
    def __init__(self, window: int = 20, method: str = 'pearson'):
        self.window = window
        self.method = method
    
    def score(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> pd.Series:
        """计算信息比率得分"""
        if return_data is None:
            raise AnalysisError("信息比率评分需要收益数据")
        
        # 确保数据对齐
        common_index = factor_data.index.intersection(return_data.index)
        factor_aligned = factor_data.loc[common_index]
        return_aligned = return_data.loc[common_index]
        
        # 计算滚动IC
        ic_scores = pd.DataFrame(index=factor_aligned.index, columns=factor_aligned.columns)
        ic_stds = pd.DataFrame(index=factor_aligned.index, columns=factor_aligned.columns)
        
        for i in range(self.window - 1, len(factor_aligned)):
            window_factor = factor_aligned.iloc[i - self.window + 1:i + 1]
            window_return = return_aligned.iloc[i - self.window + 1:i + 1]
            
            # 计算每只股票的IC
            for stock in factor_aligned.columns:
                if stock in window_factor.columns and stock in window_return.columns:
                    factor_stock = window_factor[stock].dropna()
                    return_stock = window_return[stock].dropna()
                    
                    common_dates = factor_stock.index.intersection(return_stock.index)
                    if len(common_dates) >= 3:  # 至少需要3个点计算相关性
                        factor_common = factor_stock.loc[common_dates]
                        return_common = return_stock.loc[common_dates]
                        
                        if self.method == 'pearson':
                            ic = factor_common.corr(return_common, method='pearson')
                        elif self.method == 'spearman':
                            ic = factor_common.corr(return_common, method='spearman')
                        else:
                            raise AnalysisError(f"不支持的IC计算方法: {self.method}")
                        
                        ic_scores.iloc[i, ic_scores.columns.get_loc(stock)] = ic
                        
                        # 计算IC标准差
                        if len(common_dates) >= 5:  # 至少需要5个点计算标准差
                            ic_values = []
                            for j in range(len(common_dates) - 1):
                                factor_sub = factor_common.iloc[:j+1]
                                return_sub = return_common.iloc[1:j+2]
                                
                                if self.method == 'pearson':
                                    sub_ic = factor_sub.corr(return_sub, method='pearson')
                                elif self.method == 'spearman':
                                    sub_ic = factor_sub.corr(return_sub, method='spearman')
                                
                                if not np.isnan(sub_ic):
                                    ic_values.append(sub_ic)
                            
                            if ic_values:
                                ic_stds.iloc[i, ic_stds.columns.get_loc(stock)] = np.std(ic_values)
        
        # 计算信息比率
        ir_scores = ic_scores / ic_stds
        
        # 标准化信息比率得分
        return ir_scores.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def get_name(self) -> str:
        """获取方法名称"""
        return f"信息比率评分(窗口={self.window},方法={self.method})"

class FactorScorer:
    """因子评分器"""
    
    def __init__(self, config_manager: ConfigManager, cache_manager: CacheManager, 
                 error_handler: ErrorHandler):
        self.config = config_manager
        self.cache = cache_manager
        self.error_handler = error_handler
        
        # 初始化评分方法
        self.methods = {}
        self._initialize_methods()
        
        # 注册配置监听器
        self.config.watch('scoring.default_method', self._update_default_method)
    
    def _initialize_methods(self):
        """初始化评分方法"""
        # 标准化评分
        standard_method = self.config.get('scoring.standard_method', 'zscore')
        self.methods['standard'] = StandardScalerScoring(method=standard_method)
        
        # 十分位数评分
        num_groups = self.config.get('scoring.num_groups', 10)
        self.methods['decile'] = DecileScoring(num_groups=num_groups)
        
        # IC评分
        ic_window = self.config.get('scoring.ic_window', 20)
        ic_method = self.config.get('scoring.ic_method', 'pearson')
        self.methods['ic'] = ICScoreScoring(window=ic_window, method=ic_method)
        
        # 信息比率评分
        ir_window = self.config.get('scoring.ir_window', 20)
        ir_method = self.config.get('scoring.ir_method', 'pearson')
        self.methods['ir'] = InformationRatioScoring(window=ir_window, method=ir_method)
    
    def _update_default_method(self, key: str, new_value: Any, old_value: Any):
        """更新默认评分方法"""
        logging.info(f"更新默认评分方法: {old_value} -> {new_value}")
    
    @exception_handler()
    def score_factor(self, factor_data: pd.DataFrame, return_data: pd.DataFrame = None, 
                    method: str = None) -> pd.Series:
        """对因子进行评分"""
        if method is None:
            method = self.config.get('scoring.default_method', 'standard')
        
        if method not in self.methods:
            raise AnalysisError(f"未知的评分方法: {method}")
        
        # 尝试从缓存加载
        cache_key = f"factor_score_{method}_{hash(str(factor_data.values.tobytes()))}"
        if return_data is not None:
            cache_key += f"_{hash(str(return_data.values.tobytes()))}"
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logging.info(f"从缓存加载{method}评分结果")
            return cached_result
        
        # 执行评分
        scoring_method = self.methods[method]
        try:
            if method in ['ic', 'ir']:
                # IC和信息比率评分需要收益数据
                if return_data is None:
                    raise AnalysisError(f"{scoring_method.get_name()}需要收益数据")
                result = scoring_method.score(factor_data, return_data)
            else:
                # 其他评分方法只需要因子数据
                result = scoring_method.score(factor_data)
            
            # 缓存结果
            self.cache.set(cache_key, result)
            
            logging.info(f"完成{scoring_method.get_name()}")
            return result
        except Exception as e:
            logging.error(f"{scoring_method.get_name()}失败: {e}")
            self.error_handler.handle_error(e, f"{scoring_method.get_name()}失败")
            raise
    
    @exception_handler()
    def batch_score_factors(self, factor_dict: Dict[str, pd.DataFrame], 
                           return_data: pd.DataFrame = None, 
                           method: str = None) -> Dict[str, pd.Series]:
        """批量对因子进行评分"""
        results = {}
        
        for factor_name, factor_data in factor_dict.items():
            logging.info(f"评分因子: {factor_name}")
            try:
                factor_score = self.score_factor(factor_data, return_data, method)
                results[factor_name] = factor_score
            except Exception as e:
                logging.error(f"评分因子{factor_name}失败: {e}")
                self.error_handler.handle_error(e, f"评分因子{factor_name}失败")
        
        return results
    
    @exception_handler()
    def compare_scoring_methods(self, factor_data: pd.DataFrame, 
                               return_data: pd.DataFrame = None) -> pd.DataFrame:
        """比较不同评分方法"""
        results = {}
        
        for method_name, scoring_method in self.methods.items():
            try:
                if method_name in ['ic', 'ir']:
                    # IC和信息比率评分需要收益数据
                    if return_data is None:
                        logging.warning(f"跳过{scoring_method.get_name()}，缺少收益数据")
                        continue
                    score = scoring_method.score(factor_data, return_data)
                else:
                    # 其他评分方法只需要因子数据
                    score = scoring_method.score(factor_data)
                
                # 计算统计量
                stats = {
                    'mean': score.mean().mean(),
                    'std': score.std().mean(),
                    'min': score.min().min(),
                    'max': score.max().max(),
                    'skew': score.skew().mean(),
                    'kurt': score.kurtosis().mean()
                }
                
                results[method_name] = stats
                logging.info(f"完成{scoring_method.get_name()}")
            except Exception as e:
                logging.error(f"{scoring_method.get_name()}失败: {e}")
                self.error_handler.handle_error(e, f"{scoring_method.get_name()}失败")
        
        # 创建比较表
        comparison_df = pd.DataFrame.from_dict(results, orient='index')
        
        return comparison_df
    
    def add_method(self, name: str, method: ScoringMethod):
        """添加评分方法"""
        self.methods[name] = method
        logging.info(f"添加评分方法: {name}")
    
    def remove_method(self, name: str):
        """移除评分方法"""
        if name in self.methods:
            del self.methods[name]
            logging.info(f"移除评分方法: {name}")
    
    def get_method_names(self) -> List[str]:
        """获取评分方法名称列表"""
        return list(self.methods.keys())

class FactorRanker:
    """因子排名器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
    
    @exception_handler()
    def rank_factors(self, factor_scores: Dict[str, pd.Series], 
                    method: str = 'mean') -> pd.DataFrame:
        """对因子进行排名"""
        # 确保所有分数Series有相同的索引
        common_index = None
        for score in factor_scores.values():
            if common_index is None:
                common_index = score.index
            else:
                common_index = common_index.intersection(score.index)
        
        if common_index is None or len(common_index) == 0:
            raise AnalysisError("没有共同的索引可以排名")
        
        # 对齐数据
        aligned_scores = {}
        for name, score in factor_scores.items():
            aligned_scores[name] = score.loc[common_index]
        
        # 创建排名DataFrame
        rankings = pd.DataFrame(index=common_index)
        
        if method == 'mean':
            # 平均排名
            for name, score in aligned_scores.items():
                rankings[name] = score.rank(axis=1, ascending=True)
            
            # 计算平均排名
            rankings['mean_rank'] = rankings.mean(axis=1)
            
            # 根据平均排名排序
            rankings = rankings.sort_values('mean_rank')
            
        elif method == 'weighted':
            # 加权排名
            weights = self.config.get('ranking.weights', {})
            
            for name, score in aligned_scores.items():
                weight = weights.get(name, 1.0)
                rankings[name] = score.rank(axis=1, ascending=True) * weight
            
            # 计算加权排名
            rankings['weighted_rank'] = rankings.dropna(axis=1).sum(axis=1)
            
            # 根据加权排名排序
            rankings = rankings.sort_values('weighted_rank')
            
        else:
            raise AnalysisError(f"不支持的排名方法: {method}")
        
        return rankings
    
    @exception_handler()
    def get_top_factors(self, factor_scores: Dict[str, pd.Series], 
                       n: int = 10, method: str = 'mean') -> pd.DataFrame:
        """获取排名前n的因子"""
        rankings = self.rank_factors(factor_scores, method)
        
        if method == 'mean':
            return rankings.head(n)
        elif method == 'weighted':
            return rankings.head(n)
        else:
            raise AnalysisError(f"不支持的排名方法: {method}")

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
    
    # 创建因子评分器
    factor_scorer = FactorScorer(config_manager, cache_manager, error_handler)
    
    # 创建因子排名器
    factor_ranker = FactorRanker(config_manager)
    
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
    
    # 评分因子
    try:
        # 使用不同方法评分
        standard_score = factor_scorer.score_factor(factor_data, method='standard')
        decile_score = factor_scorer.score_factor(factor_data, method='decile')
        ic_score = factor_scorer.score_factor(factor_data, return_data, method='ic')
        
        print("标准化评分结果:")
        print(standard_score.head())
        
        print("\n十分位数评分结果:")
        print(decile_score.head())
        
        print("\nIC评分结果:")
        print(ic_score.head())
        
        # 比较不同评分方法
        comparison = factor_scorer.compare_scoring_methods(factor_data, return_data)
        print("\n评分方法比较:")
        print(comparison)
        
        # 因子排名
        factor_scores = {
            'standard': standard_score.iloc[-1],  # 使用最新一天的分数
            'decile': decile_score.iloc[-1],
            'ic': ic_score.iloc[-1]
        }
        
        rankings = factor_ranker.rank_factors(factor_scores)
        print("\n因子排名:")
        print(rankings.head(10))
        
    except Exception as e:
        print(f"评分失败: {e}")