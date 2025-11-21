"""
优化后的数据处理模块
支持并行计算、数据管道和缓存优化
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
from pathlib import Path

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataLoadError, DataProcessingError, exception_handler, cache_result

class DataSource(ABC):
    """抽象数据源类"""
    
    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """加载数据"""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """验证数据"""
        pass

class CSVDataSource(DataSource):
    """CSV数据源"""
    
    def __init__(self, file_path: str, **csv_kwargs):
        self.file_path = Path(file_path)
        self.csv_kwargs = csv_kwargs
    
    def load(self, **kwargs) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            if not self.file_path.exists():
                raise DataLoadError(f"文件不存在: {self.file_path}")
            
            # 合并参数
            params = {**self.csv_kwargs, **kwargs}
            
            # 加载数据
            data = pd.read_csv(self.file_path, **params)
            
            # 验证数据
            if not self.validate(data):
                raise DataLoadError(f"数据验证失败: {self.file_path}")
            
            logging.info(f"成功加载CSV数据: {self.file_path}, 形状: {data.shape}")
            return data
        except Exception as e:
            raise DataLoadError(f"加载CSV数据失败: {e}")
    
    def validate(self, data: pd.DataFrame) -> bool:
        """验证CSV数据"""
        if data.empty:
            return False
        
        # 检查是否有重复的索引
        if data.index.duplicated().any():
            logging.warning("数据中存在重复索引")
            return False
        
        return True

class ExcelDataSource(DataSource):
    """Excel数据源"""
    
    def __init__(self, file_path: str, sheet_name: Union[str, int] = 0, **excel_kwargs):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.excel_kwargs = excel_kwargs
    
    def load(self, **kwargs) -> pd.DataFrame:
        """加载Excel数据"""
        try:
            if not self.file_path.exists():
                raise DataLoadError(f"文件不存在: {self.file_path}")
            
            # 合并参数
            params = {**self.excel_kwargs, **kwargs}
            
            # 加载数据
            data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, **params)
            
            # 验证数据
            if not self.validate(data):
                raise DataLoadError(f"数据验证失败: {self.file_path}")
            
            logging.info(f"成功加载Excel数据: {self.file_path}, 工作表: {self.sheet_name}, 形状: {data.shape}")
            return data
        except Exception as e:
            raise DataLoadError(f"加载Excel数据失败: {e}")
    
    def validate(self, data: pd.DataFrame) -> bool:
        """验证Excel数据"""
        if data.empty:
            return False
        
        # 检查是否有重复的索引
        if data.index.duplicated().any():
            logging.warning("数据中存在重复索引")
            return False
        
        return True

class DatabaseDataSource(DataSource):
    """数据库数据源"""
    
    def __init__(self, connection_string: str, query: str, **db_kwargs):
        self.connection_string = connection_string
        self.query = query
        self.db_kwargs = db_kwargs
    
    def load(self, **kwargs) -> pd.DataFrame:
        """从数据库加载数据"""
        try:
            import sqlalchemy
            
            # 创建数据库连接
            engine = sqlalchemy.create_engine(self.connection_string, **self.db_kwargs)
            
            # 执行查询
            data = pd.read_sql(self.query, engine, **kwargs)
            
            # 验证数据
            if not self.validate(data):
                raise DataLoadError(f"数据验证失败: {self.query}")
            
            logging.info(f"成功从数据库加载数据, 形状: {data.shape}")
            return data
        except ImportError:
            raise DataLoadError("缺少sqlalchemy库，无法连接数据库")
        except Exception as e:
            raise DataLoadError(f"从数据库加载数据失败: {e}")
    
    def validate(self, data: pd.DataFrame) -> bool:
        """验证数据库数据"""
        if data.empty:
            return False
        
        # 检查是否有重复的索引
        if data.index.duplicated().any():
            logging.warning("数据中存在重复索引")
            return False
        
        return True

class DataTransformer(ABC):
    """抽象数据转换器"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """获取转换参数"""
        pass

class StandardScaler(DataTransformer):
    """标准化转换器"""
    
    def __init__(self, columns: List[str] = None, with_mean: bool = True, with_std: bool = True):
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.means_ = None
        self.stds_ = None
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据"""
        result = data.copy()
        
        if self.columns is None:
            self.columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 计算均值和标准差
        if self.with_mean:
            self.means_ = data[self.columns].mean()
        if self.with_std:
            self.stds_ = data[self.columns].std()
        
        # 标准化
        for col in self.columns:
            if col in data.columns:
                if self.with_mean:
                    result[col] = result[col] - self.means_[col]
                if self.with_std and self.stds_[col] > 0:
                    result[col] = result[col] / self.stds_[col]
        
        return result
    
    def get_params(self) -> Dict:
        """获取转换参数"""
        return {
            'columns': self.columns,
            'with_mean': self.with_mean,
            'with_std': self.with_std,
            'means_': self.means_.to_dict() if self.means_ is not None else None,
            'stds_': self.stds_.to_dict() if self.stds_ is not None else None
        }

class MinMaxScaler(DataTransformer):
    """最小-最大标准化转换器"""
    
    def __init__(self, columns: List[str] = None, feature_range: Tuple[float, float] = (0, 1)):
        self.columns = columns
        self.feature_range = feature_range
        self.mins_ = None
        self.maxs_ = None
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """最小-最大标准化数据"""
        result = data.copy()
        
        if self.columns is None:
            self.columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 计算最小值和最大值
        self.mins_ = data[self.columns].min()
        self.maxs_ = data[self.columns].max()
        
        # 标准化
        min_val, max_val = self.feature_range
        for col in self.columns:
            if col in data.columns:
                if self.maxs_[col] > self.mins_[col]:
                    result[col] = (result[col] - self.mins_[col]) / (self.maxs_[col] - self.mins_[col])
                    result[col] = result[col] * (max_val - min_val) + min_val
                else:
                    result[col] = min_val
        
        return result
    
    def get_params(self) -> Dict:
        """获取转换参数"""
        return {
            'columns': self.columns,
            'feature_range': self.feature_range,
            'mins_': self.mins_.to_dict() if self.mins_ is not None else None,
            'maxs_': self.maxs_.to_dict() if self.maxs_ is not None else None
        }

class DataFilter(DataTransformer):
    """数据过滤器"""
    
    def __init__(self, filters: List[Dict]):
        self.filters = filters
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """过滤数据"""
        result = data.copy()
        
        for filter_config in self.filters:
            column = filter_config.get('column')
            operation = filter_config.get('operation')
            value = filter_config.get('value')
            
            if column not in result.columns:
                continue
            
            if operation == 'eq':
                result = result[result[column] == value]
            elif operation == 'ne':
                result = result[result[column] != value]
            elif operation == 'gt':
                result = result[result[column] > value]
            elif operation == 'ge':
                result = result[result[column] >= value]
            elif operation == 'lt':
                result = result[result[column] < value]
            elif operation == 'le':
                result = result[result[column] <= value]
            elif operation == 'in':
                result = result[result[column].isin(value)]
            elif operation == 'not_in':
                result = result[~result[column].isin(value)]
            elif operation == 'isnull':
                result = result[result[column].isnull()]
            elif operation == 'notnull':
                result = result[result[column].notnull()]
        
        return result
    
    def get_params(self) -> Dict:
        """获取过滤参数"""
        return {'filters': self.filters}

class DataAggregator(DataTransformer):
    """数据聚合器"""
    
    def __init__(self, group_by: List[str], agg_dict: Dict[str, Union[str, List[str]]]):
        self.group_by = group_by
        self.agg_dict = agg_dict
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """聚合数据"""
        if not all(col in data.columns for col in self.group_by):
            raise DataProcessingError(f"聚合列不存在: {self.group_by}")
        
        if not all(col in data.columns for col in self.agg_dict.keys()):
            raise DataProcessingError(f"聚合目标列不存在: {self.agg_dict.keys()}")
        
        return data.groupby(self.group_by).agg(self.agg_dict).reset_index()
    
    def get_params(self) -> Dict:
        """获取聚合参数"""
        return {
            'group_by': self.group_by,
            'agg_dict': self.agg_dict
        }

class DataPipeline:
    """数据管道"""
    
    def __init__(self, steps: List[Tuple[str, Union[DataSource, DataTransformer]]]):
        self.steps = steps
        self._validate_pipeline()
    
    def _validate_pipeline(self):
        """验证管道"""
        if not self.steps:
            raise DataProcessingError("管道步骤不能为空")
        
        # 第一步必须是数据源
        if not isinstance(self.steps[0][1], DataSource):
            raise DataProcessingError("管道第一步必须是数据源")
        
        # 后续步骤必须是转换器
        for i, (name, step) in enumerate(self.steps[1:], 1):
            if not isinstance(step, DataTransformer):
                raise DataProcessingError(f"管道第{i}步({name})必须是转换器")
    
    def execute(self, **kwargs) -> pd.DataFrame:
        """执行管道"""
        data = None
        
        for name, step in self.steps:
            if isinstance(step, DataSource):
                data = step.load(**kwargs)
            elif isinstance(step, DataTransformer):
                if data is None:
                    raise DataProcessingError(f"转换器{name}没有输入数据")
                data = step.transform(data)
            else:
                raise DataProcessingError(f"未知的管道步骤类型: {type(step)}")
        
        return data
    
    def get_params(self) -> Dict:
        """获取管道参数"""
        params = {}
        for name, step in self.steps:
            if isinstance(step, DataTransformer):
                params[name] = step.get_params()
        return params

class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        
        if use_processes:
            self.executor_class = ProcessPoolExecutor
        else:
            self.executor_class = ThreadPoolExecutor
    
    def parallel_apply(self, data: pd.DataFrame, func: Callable, 
                       column: str = None, axis: int = 0, 
                       chunk_size: Optional[int] = None) -> pd.DataFrame:
        """并行应用函数"""
        if column is not None:
            # 按列并行处理
            return self._parallel_apply_column(data, func, column, chunk_size)
        else:
            # 按行并行处理
            return self._parallel_apply_axis(data, func, axis, chunk_size)
    
    def _parallel_apply_column(self, data: pd.DataFrame, func: Callable, 
                              column: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """按列并行处理"""
        if column not in data.columns:
            raise DataProcessingError(f"列不存在: {column}")
        
        # 分块处理
        if chunk_size is None:
            chunk_size = max(1, len(data) // self.max_workers)
        
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        results = []
        with self.executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, chunk[column]) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"并行处理失败: {e}")
                    raise DataProcessingError(f"并行处理失败: {e}")
        
        # 合并结果
        return pd.concat(results, ignore_index=True)
    
    def _parallel_apply_axis(self, data: pd.DataFrame, func: Callable, 
                            axis: int, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """按轴并行处理"""
        # 分块处理
        if chunk_size is None:
            if axis == 0:  # 按行分块
                chunk_size = max(1, len(data) // self.max_workers)
                chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            else:  # 按列分块
                chunk_size = max(1, len(data.columns) // self.max_workers)
                chunks = [data.iloc[:, i:i+chunk_size] for i in range(0, len(data.columns), chunk_size)]
        else:
            if axis == 0:  # 按行分块
                chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            else:  # 按列分块
                chunks = [data.iloc[:, i:i+chunk_size] for i in range(0, len(data.columns), chunk_size)]
        
        results = []
        with self.executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, chunk, axis=axis) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"并行处理失败: {e}")
                    raise DataProcessingError(f"并行处理失败: {e}")
        
        # 合并结果
        if axis == 0:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.concat(results, axis=1)
    
    def parallel_groupby_apply(self, data: pd.DataFrame, group_by: List[str], 
                              func: Callable) -> pd.DataFrame:
        """并行分组应用函数"""
        if not all(col in data.columns for col in group_by):
            raise DataProcessingError(f"分组列不存在: {group_by}")
        
        # 分组
        groups = data.groupby(group_by)
        group_keys = list(groups.groups.keys())
        
        # 并行处理每个分组
        results = []
        with self.executor_class(max_workers=self.max_workers) as executor:
            futures = {executor.submit(func, group): key for key, group in groups}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    key = futures[future]
                    if isinstance(result, pd.DataFrame):
                        result[group_by] = key
                        results.append(result)
                    else:
                        logging.warning(f"分组处理结果不是DataFrame: {type(result)}")
                except Exception as e:
                    logging.error(f"分组处理失败: {e}")
                    raise DataProcessingError(f"分组处理失败: {e}")
        
        # 合并结果
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame(columns=data.columns)

class OptimizedDataProcessor:
    """优化的数据处理器"""
    
    def __init__(self, config_manager: ConfigManager, cache_manager: CacheManager, 
                 error_handler: ErrorHandler):
        self.config = config_manager
        self.cache = cache_manager
        self.error_handler = error_handler
        
        # 初始化并行处理器
        max_workers = self.config.get('processing.max_workers', min(32, (os.cpu_count() or 1) + 4))
        use_processes = self.config.get('processing.use_processes', False)
        self.parallel_processor = ParallelProcessor(max_workers, use_processes)
        
        # 注册配置监听器
        self.config.watch('processing.max_workers', self._update_max_workers)
        self.config.watch('processing.use_processes', self._update_use_processes)
    
    def _update_max_workers(self, key: str, new_value: Any, old_value: Any):
        """更新最大工作线程数"""
        self.parallel_processor.max_workers = new_value
        logging.info(f"更新最大工作线程数: {old_value} -> {new_value}")
    
    def _update_use_processes(self, key: str, new_value: Any, old_value: Any):
        """更新是否使用进程"""
        self.parallel_processor.use_processes = new_value
        if new_value:
            self.parallel_processor.executor_class = ProcessPoolExecutor
        else:
            self.parallel_processor.executor_class = ThreadPoolExecutor
        logging.info(f"更新并行处理模式: {old_value} -> {new_value}")
    
    @exception_handler()
    def load_data(self, source: Union[str, DataSource], **kwargs) -> pd.DataFrame:
        """加载数据"""
        if isinstance(source, str):
            # 根据文件扩展名创建数据源
            if source.endswith('.csv'):
                data_source = CSVDataSource(source, **kwargs)
            elif source.endswith(('.xlsx', '.xls')):
                data_source = ExcelDataSource(source, **kwargs)
            else:
                raise DataLoadError(f"不支持的数据源类型: {source}")
        else:
            data_source = source
        
        # 尝试从缓存加载
        cache_key = f"load_data_{hash(str(source))}_{hash(str(kwargs))}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info(f"从缓存加载数据: {source}")
            return cached_data
        
        # 加载数据
        data = data_source.load(**kwargs)
        
        # 缓存数据
        self.cache.set(cache_key, data)
        
        return data
    
    @exception_handler()
    def preprocess_data(self, data: pd.DataFrame, transformers: List[DataTransformer]) -> pd.DataFrame:
        """预处理数据"""
        # 尝试从缓存加载
        cache_key = f"preprocess_data_{hash(str(data.values.tobytes()))}_{hash(str([t.get_params() for t in transformers]))}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info("从缓存加载预处理数据")
            return cached_data
        
        # 应用转换器
        result = data.copy()
        for transformer in transformers:
            result = transformer.transform(result)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    @exception_handler()
    def parallel_apply(self, data: pd.DataFrame, func: Callable, 
                      column: str = None, axis: int = 0, 
                      chunk_size: Optional[int] = None) -> pd.DataFrame:
        """并行应用函数"""
        # 尝试从缓存加载
        cache_key = f"parallel_apply_{hash(str(data.values.tobytes()))}_{hash(str(func))}_{column}_{axis}_{chunk_size}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info("从缓存加载并行处理结果")
            return cached_data
        
        # 并行处理
        result = self.parallel_processor.parallel_apply(data, func, column, axis, chunk_size)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    @exception_handler()
    def parallel_groupby_apply(self, data: pd.DataFrame, group_by: List[str], 
                              func: Callable) -> pd.DataFrame:
        """并行分组应用函数"""
        # 尝试从缓存加载
        cache_key = f"parallel_groupby_apply_{hash(str(data.values.tobytes()))}_{hash(str(group_by))}_{hash(str(func))}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info("从缓存加载并行分组处理结果")
            return cached_data
        
        # 并行处理
        result = self.parallel_processor.parallel_groupby_apply(data, group_by, func)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    @exception_handler()
    def execute_pipeline(self, pipeline: DataPipeline, **kwargs) -> pd.DataFrame:
        """执行数据管道"""
        # 尝试从缓存加载
        cache_key = f"execute_pipeline_{hash(str(pipeline.steps))}_{hash(str(kwargs))}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info("从缓存加载管道执行结果")
            return cached_data
        
        # 执行管道
        result = pipeline.execute(**kwargs)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def create_pipeline(self, config: Dict) -> DataPipeline:
        """创建数据管道"""
        steps = []
        
        # 数据源
        source_config = config.get('source', {})
        source_type = source_config.get('type', 'csv')
        
        if source_type == 'csv':
            source = CSVDataSource(
                source_config.get('path'),
                **source_config.get('params', {})
            )
        elif source_type == 'excel':
            source = ExcelDataSource(
                source_config.get('path'),
                source_config.get('sheet_name', 0),
                **source_config.get('params', {})
            )
        elif source_type == 'database':
            source = DatabaseDataSource(
                source_config.get('connection_string'),
                source_config.get('query'),
                **source_config.get('params', {})
            )
        else:
            raise DataProcessingError(f"不支持的数据源类型: {source_type}")
        
        steps.append(('source', source))
        
        # 转换器
        transformers_config = config.get('transformers', [])
        for i, transformer_config in enumerate(transformers_config):
            transformer_type = transformer_config.get('type')
            params = transformer_config.get('params', {})
            
            if transformer_type == 'standard_scaler':
                transformer = StandardScaler(**params)
            elif transformer_type == 'min_max_scaler':
                transformer = MinMaxScaler(**params)
            elif transformer_type == 'filter':
                transformer = DataFilter(**params)
            elif transformer_type == 'aggregator':
                transformer = DataAggregator(**params)
            else:
                raise DataProcessingError(f"不支持的转换器类型: {transformer_type}")
            
            steps.append((f'transformer_{i}', transformer))
        
        return DataPipeline(steps)

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
    
    # 创建数据处理器
    data_processor = OptimizedDataProcessor(config_manager, cache_manager, error_handler)
    
    # 创建数据管道
    pipeline_config = {
        'source': {
            'type': 'csv',
            'path': 'data.csv',
            'params': {
                'encoding': 'utf-8',
                'index_col': 0
            }
        },
        'transformers': [
            {
                'type': 'filter',
                'params': {
                    'filters': [
                        {'column': 'value', 'operation': 'gt', 'value': 0}
                    ]
                }
            },
            {
                'type': 'standard_scaler',
                'params': {
                    'columns': ['value', 'factor']
                }
            }
        ]
    }
    
    pipeline = data_processor.create_pipeline(pipeline_config)
    
    # 执行管道
    try:
        data = data_processor.execute_pipeline(pipeline)
        print(f"处理完成，数据形状: {data.shape}")
    except Exception as e:
        print(f"处理失败: {e}")