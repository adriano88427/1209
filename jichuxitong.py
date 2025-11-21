"""
优化后的因子分析代码拆分方案
基于Sequential Thinking MCP分析结果的具体实现
"""

import os
import sys
import logging
import json
import yaml
import pickle
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import inspect
import traceback
from functools import wraps
import weakref

# ==================== 配置管理系统 ====================

class ConfigFormat(Enum):
    """配置文件格式枚举"""
    YAML = "yaml"
    JSON = "json"
    INI = "ini"

class ConfigManager:
    """增强的配置管理器，支持多环境、热更新和配置验证"""
    
    def __init__(self, config_path: str = "config.yaml", env: str = "default", 
                 format: ConfigFormat = ConfigFormat.YAML, auto_reload: bool = True):
        self.config_path = Path(config_path)
        self.env = env
        self.format = format
        self.auto_reload = auto_reload
        self.config = {}
        self.watchers = []
        self.validators = {}
        self._last_modified = 0
        self._lock = threading.RLock()
        self._reload_thread = None
        self._stop_event = threading.Event()
        
        self._load_config()
        
        if self.auto_reload:
            self._start_auto_reload()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with self._lock:
                if not self.config_path.exists():
                    logging.warning(f"配置文件不存在: {self.config_path}")
                    self.config = {}
                    return
                
                # 检查文件修改时间
                current_modified = self.config_path.stat().st_mtime
                if current_modified <= self._last_modified:
                    return
                
                self._last_modified = current_modified
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.format == ConfigFormat.YAML:
                        all_config = yaml.safe_load(f)
                    elif self.format == ConfigFormat.JSON:
                        all_config = json.load(f)
                    else:
                        raise ValueError(f"不支持的配置格式: {self.format}")
                
                # 获取环境特定配置
                if isinstance(all_config, dict):
                    self.config = all_config.get(self.env, all_config.get('default', {}))
                else:
                    self.config = all_config
                
                # 验证配置
                self._validate_config()
                
                logging.info(f"配置加载成功: {self.config_path}, 环境: {self.env}")
        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            self.config = {}
    
    def _validate_config(self):
        """验证配置"""
        for key, validator in self.validators.items():
            value = self.get(key)
            if value is not None and not validator(value):
                logging.warning(f"配置验证失败: {key} = {value}")
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的嵌套键"""
        with self._lock:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    def set(self, key: str, value, persist=False):
        """设置配置值"""
        with self._lock:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            old_value = config.get(keys[-1])
            config[keys[-1]] = value
            
            # 持久化到文件
            if persist:
                self._save_config()
            
            # 通知监听器
            if old_value != value:
                self._notify_watchers(key, value, old_value)
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            # 读取完整配置
            all_config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.format == ConfigFormat.YAML:
                        all_config = yaml.safe_load(f) or {}
                    elif self.format == ConfigFormat.JSON:
                        all_config = json.load(f)
            
            # 更新环境配置
            if not isinstance(all_config, dict):
                all_config = {}
            all_config[self.env] = self.config
            
            # 写入文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.format == ConfigFormat.YAML:
                    yaml.dump(all_config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif self.format == ConfigFormat.JSON:
                    json.dump(all_config, f, indent=2, ensure_ascii=False)
            
            self._last_modified = self.config_path.stat().st_mtime
            logging.info(f"配置保存成功: {self.config_path}")
        except Exception as e:
            logging.error(f"配置保存失败: {e}")
    
    def watch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """添加配置变更监听器"""
        with self._lock:
            self.watchers.append((key, callback))
    
    def add_validator(self, key: str, validator: Callable[[Any], bool]):
        """添加配置验证器"""
        self.validators[key] = validator
    
    def _notify_watchers(self, key: str, new_value: Any, old_value: Any):
        """通知所有监听器"""
        for watch_key, callback in self.watchers:
            if key.startswith(watch_key):
                try:
                    callback(key, new_value, old_value)
                except Exception as e:
                    logging.error(f"配置监听器错误: {e}")
    
    def _start_auto_reload(self):
        """启动自动重载线程"""
        self._reload_thread = threading.Thread(target=self._auto_reload_worker, daemon=True)
        self._reload_thread.start()
    
    def _auto_reload_worker(self):
        """自动重载工作线程"""
        while not self._stop_event.wait(1.0):  # 每秒检查一次
            try:
                if self.config_path.exists():
                    current_modified = self.config_path.stat().st_mtime
                    if current_modified > self._last_modified:
                        self._load_config()
            except Exception as e:
                logging.error(f"自动重载配置失败: {e}")
    
    def reload(self):
        """手动重新加载配置"""
        self._load_config()
    
    def stop(self):
        """停止配置管理器"""
        if self._reload_thread:
            self._stop_event.set()
            self._reload_thread.join(timeout=2.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# ==================== 异常处理系统 ====================

class FactorAnalysisError(Exception):
    """因子分析基础异常类"""
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self):
        """转换为字典"""
        return {
            'message': str(self),
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }

class DataLoadError(FactorAnalysisError):
    """数据加载异常"""
    pass

class DataProcessingError(FactorAnalysisError):
    """数据处理异常"""
    pass

class FactorCalculationError(FactorAnalysisError):
    """因子计算异常"""
    pass

class ReportGenerationError(FactorAnalysisError):
    """报告生成异常"""
    pass

class ConfigurationError(FactorAnalysisError):
    """配置异常"""
    pass

class RetryStrategy:
    """重试策略"""
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
                 exceptions: Tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
    
    def execute(self, func: Callable, *args, **kwargs):
        """执行函数，支持重试"""
        last_exception = None
        current_delay = self.delay
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    logging.warning(f"执行失败，{current_delay}秒后重试 (尝试 {attempt + 1}/{self.max_attempts}): {e}")
                    time.sleep(current_delay)
                    current_delay *= self.backoff
                else:
                    logging.error(f"执行失败，已达最大重试次数: {e}")
        
        raise last_exception

class ErrorHandler:
    """增强的全局异常处理器"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.handlers = {}
        self.retry_strategies = {}
        self.config = config_manager
        self.error_stats = {}
        self.register_default_handlers()
    
    def register_handler(self, exception_type: type, handler: Callable):
        """注册异常处理器"""
        self.handlers[exception_type] = handler
    
    def register_retry_strategy(self, exception_type: type, strategy: RetryStrategy):
        """注册重试策略"""
        self.retry_strategies[exception_type] = strategy
    
    def register_default_handlers(self):
        """注册默认异常处理器"""
        self.register_handler(DataLoadError, self._handle_data_load_error)
        self.register_handler(DataProcessingError, self._handle_data_processing_error)
        self.register_handler(FactorCalculationError, self._handle_factor_calculation_error)
        self.register_handler(ReportGenerationError, self._handle_report_generation_error)
        self.register_handler(ConfigurationError, self._handle_configuration_error)
        
        # 注册默认重试策略
        self.register_retry_strategy(DataLoadError, RetryStrategy(max_attempts=3, delay=1.0))
        self.register_retry_strategy(DataProcessingError, RetryStrategy(max_attempts=2, delay=0.5))
        self.register_retry_strategy(FactorCalculationError, RetryStrategy(max_attempts=1, delay=0.1))
    
    def handle(self, exception: Exception, context: Dict = None) -> Any:
        """处理异常"""
        exception_type = type(exception)
        
        # 记录错误统计
        self._record_error(exception_type)
        
        # 检查是否有重试策略
        if exception_type in self.retry_strategies and context and 'retry_func' in context:
            try:
                strategy = self.retry_strategies[exception_type]
                return strategy.execute(context['retry_func'], *context.get('retry_args', []), 
                                       **context.get('retry_kwargs', {}))
            except Exception as retry_error:
                logging.error(f"重试失败: {retry_error}")
                exception = retry_error
                exception_type = type(retry_error)
        
        # 获取处理器
        handler = self.handlers.get(exception_type, self._handle_generic_error)
        return handler(exception, context or {})
    
    def _record_error(self, exception_type: type):
        """记录错误统计"""
        error_name = exception_type.__name__
        if error_name not in self.error_stats:
            self.error_stats[error_name] = {'count': 0, 'last_occurrence': None}
        
        self.error_stats[error_name]['count'] += 1
        self.error_stats[error_name]['last_occurrence'] = datetime.now()
    
    def _handle_data_load_error(self, exception: DataLoadError, context: Dict):
        """处理数据加载异常"""
        logging.error(f"数据加载失败: {exception}")
        
        # 尝试使用备用数据源
        if 'backup_source' in context:
            try:
                return context['backup_source']()
            except Exception as e:
                logging.error(f"备用数据源也失败: {e}")
        
        # 尝试使用缓存数据
        if 'cache_manager' in context and 'cache_key' in context:
            try:
                cached_data = context['cache_manager'].get(context['cache_key'])
                if cached_data is not None:
                    logging.info(f"使用缓存数据: {context['cache_key']}")
                    return cached_data
            except Exception as e:
                logging.error(f"获取缓存数据失败: {e}")
        
        raise exception
    
    def _handle_data_processing_error(self, exception: DataProcessingError, context: Dict):
        """处理数据处理异常"""
        logging.error(f"数据处理失败: {exception}")
        
        # 尝试使用默认处理参数
        if 'default_processor' in context and 'default_params' in context:
            try:
                return context['default_processor'](**context['default_params'])
            except Exception as e:
                logging.error(f"默认处理参数也失败: {e}")
        
        raise exception
    
    def _handle_factor_calculation_error(self, exception: FactorCalculationError, context: Dict):
        """处理因子计算异常"""
        logging.error(f"因子计算失败: {exception}")
        
        # 尝试使用简化计算方法
        if 'simplified_method' in context:
            try:
                return context['simplified_method']()
            except Exception as e:
                logging.error(f"简化计算方法也失败: {e}")
        
        raise exception
    
    def _handle_report_generation_error(self, exception: ReportGenerationError, context: Dict):
        """处理报告生成异常"""
        logging.error(f"报告生成失败: {exception}")
        
        # 尝试生成简化报告
        if 'simplified_report' in context:
            try:
                return context['simplified_report']()
            except Exception as e:
                logging.error(f"简化报告生成也失败: {e}")
        
        raise exception
    
    def _handle_configuration_error(self, exception: ConfigurationError, context: Dict):
        """处理配置异常"""
        logging.error(f"配置错误: {exception}")
        
        # 尝试使用默认配置
        if 'default_config' in context:
            try:
                return context['default_config']
            except Exception as e:
                logging.error(f"默认配置也失败: {e}")
        
        raise exception
    
    def _handle_generic_error(self, exception: Exception, context: Dict):
        """处理通用异常"""
        logging.error(f"未处理的异常: {exception}")
        raise exception
    
    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        return self.error_stats.copy()
    
    def clear_error_stats(self):
        """清除错误统计"""
        self.error_stats.clear()

def exception_handler(error_handler: ErrorHandler = None):
    """异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    # 提取函数信息作为上下文
                    context = {
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': args,
                        'kwargs': kwargs
                    }
                    return error_handler.handle(e, context)
                else:
                    raise e
        return wrapper
    return decorator

# ==================== 缓存系统 ====================

class CacheKey:
    """缓存键生成器"""
    
    @staticmethod
    def generate(func_name: str, args: tuple = None, kwargs: dict = None) -> str:
        """生成缓存键"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

class CacheManager:
    """增强的缓存管理器，支持多级缓存和缓存策略"""
    
    def __init__(self, cache_dir: str = "cache", memory_size: int = 100, 
                 disk_size: int = 1000, default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache = {}
        self.memory_size = memory_size
        self.disk_size = disk_size
        self.default_ttl = default_ttl
        
        self.memory_access_order = []
        self.disk_metadata = {}
        
        self._lock = threading.RLock()
        self._load_disk_metadata()
    
    def _load_disk_metadata(self):
        """加载磁盘缓存元数据"""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.disk_metadata = json.load(f)
            except Exception as e:
                logging.error(f"加载磁盘缓存元数据失败: {e}")
                self.disk_metadata = {}
    
    def _save_disk_metadata(self):
        """保存磁盘缓存元数据"""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.disk_metadata, f)
        except Exception as e:
            logging.error(f"保存磁盘缓存元数据失败: {e}")
    
    def get(self, key: str, use_disk: bool = True) -> Any:
        """获取缓存值"""
        with self._lock:
            # 先检查内存缓存
            if key in self.memory_cache:
                self._update_memory_access_order(key)
                return self.memory_cache[key]['data']
            
            # 检查磁盘缓存
            if use_disk and key in self.disk_metadata:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # 检查TTL
                        metadata = self.disk_metadata[key]
                        if time.time() - metadata['timestamp'] < metadata.get('ttl', self.default_ttl):
                            # 加载到内存缓存
                            self._add_to_memory_cache(key, data)
                            return data
                        else:
                            # 缓存过期，删除
                            cache_file.unlink()
                            del self.disk_metadata[key]
                            self._save_disk_metadata()
                    except Exception as e:
                        logging.error(f"加载磁盘缓存失败: {e}")
            
            return None
    
    def set(self, key: str, value: Any, use_disk: bool = True, ttl: int = None):
        """设置缓存值"""
        with self._lock:
            ttl = ttl or self.default_ttl
            
            # 设置内存缓存
            self._add_to_memory_cache(key, value)
            
            # 设置磁盘缓存
            if use_disk:
                cache_file = self.cache_dir / f"{key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(value, f)
                    
                    # 更新元数据
                    self.disk_metadata[key] = {
                        'timestamp': time.time(),
                        'ttl': ttl,
                        'size': cache_file.stat().st_size
                    }
                    
                    # 检查磁盘缓存大小限制
                    self._check_disk_cache_size()
                    
                    self._save_disk_metadata()
                except Exception as e:
                    logging.error(f"保存磁盘缓存失败: {e}")
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """添加到内存缓存"""
        # 如果已存在，更新访问顺序
        if key in self.memory_cache:
            self._update_memory_access_order(key)
            self.memory_cache[key]['data'] = value
            return
        
        # 检查内存缓存大小限制
        while len(self.memory_cache) >= self.memory_size:
            oldest_key = self.memory_access_order.pop(0)
            del self.memory_cache[oldest_key]
        
        # 添加新缓存
        self.memory_cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
        self.memory_access_order.append(key)
    
    def _update_memory_access_order(self, key: str):
        """更新内存缓存访问顺序"""
        if key in self.memory_access_order:
            self.memory_access_order.remove(key)
        self.memory_access_order.append(key)
    
    def _check_disk_cache_size(self):
        """检查磁盘缓存大小限制"""
        while len(self.disk_metadata) > self.disk_size:
            # 找到最旧的缓存
            oldest_key = min(self.disk_metadata.keys(), 
                            key=lambda k: self.disk_metadata[k]['timestamp'])
            
            # 删除缓存文件
            cache_file = self.cache_dir / f"{oldest_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # 删除元数据
            del self.disk_metadata[oldest_key]
    
    def clear(self, key: str = None):
        """清除缓存"""
        with self._lock:
            if key:
                # 清除特定缓存
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    if key in self.memory_access_order:
                        self.memory_access_order.remove(key)
                
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                if key in self.disk_metadata:
                    del self.disk_metadata[key]
                    self._save_disk_metadata()
            else:
                # 清除所有缓存
                self.memory_cache.clear()
                self.memory_access_order.clear()
                
                for file in self.cache_dir.glob("*.pkl"):
                    file.unlink()
                
                self.disk_metadata.clear()
                self._save_disk_metadata()
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'memory_cache_size': len(self.memory_cache),
                'memory_cache_limit': self.memory_size,
                'disk_cache_size': len(self.disk_metadata),
                'disk_cache_limit': self.disk_size,
                'disk_cache_dir': str(self.cache_dir)
            }

def cache_result(cache_manager: CacheManager = None, ttl: int = None, 
                use_disk: bool = True, key_func: Callable = None):
    """缓存结果装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = CacheKey.generate(func.__name__, args, kwargs)
            
            # 尝试从缓存获取
            if cache_manager:
                cached_result = cache_manager.get(cache_key, use_disk)
                if cached_result is not None:
                    return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            if cache_manager:
                cache_manager.set(cache_key, result, use_disk, ttl)
            
            return result
        return wrapper
    return decorator

# ==================== 任务调度系统 ====================

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()

class Task(ABC):
    """抽象任务类"""
    
    def __init__(self, task_id: str, name: str = None, priority: int = 0, 
                 timeout: Optional[float] = None, retry_count: int = 0):
        self.task_id = task_id
        self.name = name or task_id
        self.priority = priority
        self.timeout = timeout
        self.retry_count = retry_count
        self.max_retries = retry_count
        self.dependencies = []
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.metadata = {}
        self.progress = 0.0
        self._lock = threading.Lock()
    
    @abstractmethod
    def execute(self, context: Dict = None) -> Any:
        """执行任务"""
        pass
    
    def add_dependency(self, task: 'Task'):
        """添加任务依赖"""
        if task not in self.dependencies:
            self.dependencies.append(task)
    
    def remove_dependency(self, task: 'Task'):
        """移除任务依赖"""
        if task in self.dependencies:
            self.dependencies.remove(task)
    
    def can_execute(self) -> bool:
        """检查任务是否可以执行"""
        if self.status != TaskStatus.PENDING:
            return False
        
        for dep in self.dependencies:
            if dep.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def update_progress(self, progress: float):
        """更新任务进度"""
        with self._lock:
            self.progress = max(0.0, min(1.0, progress))
    
    def set_metadata(self, key: str, value: Any):
        """设置元数据"""
        with self._lock:
            self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        """获取元数据"""
        with self._lock:
            return self.metadata.get(key, default)
    
    def __repr__(self):
        return f"Task({self.task_id}, status={self.status.value}, progress={self.progress:.2f})"

class DAGTaskScheduler:
    """增强的基于DAG的任务调度器"""
    
    def __init__(self, max_workers: int = 4, config_manager: ConfigManager = None):
        self.max_workers = max_workers
        self.config = config_manager
        self.tasks = {}
        self.dag = nx.DiGraph()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
        self.running = False
        self.paused = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._event_handlers = {}
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'skipped_tasks': 0,
            'total_execution_time': 0.0
        }
    
    def add_task(self, task: Task):
        """添加任务"""
        with self._lock:
            self.tasks[task.task_id] = task
            self.dag.add_node(task.task_id, task=task)
            self._stats['total_tasks'] += 1
    
    def add_dependency(self, task_id: str, depends_on: str):
        """添加任务依赖"""
        with self._lock:
            if task_id in self.tasks and depends_on in self.tasks:
                self.dag.add_edge(depends_on, task_id)
                self.tasks[task_id].add_dependency(self.tasks[depends_on])
    
    def remove_dependency(self, task_id: str, depends_on: str):
        """移除任务依赖"""
        with self._lock:
            if task_id in self.tasks and depends_on in self.tasks:
                if self.dag.has_edge(depends_on, task_id):
                    self.dag.remove_edge(depends_on, task_id)
                    self.tasks[task_id].remove_dependency(self.tasks[depends_on])
    
    def add_event_handler(self, event: str, handler: Callable[[Task, TaskResult], None]):
        """添加事件处理器"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _trigger_event(self, event: str, task: Task, result: TaskResult):
        """触发事件"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(task, result)
                except Exception as e:
                    logging.error(f"事件处理器错误: {e}")
    
    def _validate_dag(self) -> bool:
        """验证DAG是否有效（无环）"""
        try:
            nx.find_cycle(self.dag)
            return False  # 有环
        except nx.NetworkXNoCycle:
            return True  # 无环
    
    def _get_ready_tasks(self) -> List[Task]:
        """获取准备执行的任务"""
        with self._lock:
            ready_tasks = []
            for task_id, task in self.tasks.items():
                if task.can_execute():
                    ready_tasks.append(task)
            return ready_tasks
    
    def _execute_task(self, task: Task, context: Dict = None):
        """执行单个任务"""
        try:
            with task._lock:
                task.status = TaskStatus.RUNNING
                task.start_time = datetime.now()
            
            # 触发任务开始事件
            self._trigger_event('task_start', task, None)
            
            # 执行任务
            if task.timeout:
                future = self.executor.submit(task.execute, context or {})
                try:
                    result = future.result(timeout=task.timeout)
                except Exception as e:
                    future.cancel()
                    raise e
            else:
                result = task.execute(context or {})
            
            end_time = datetime.now()
            
            with task._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.end_time = end_time
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=task.start_time,
                end_time=end_time,
                metadata=task.metadata.copy()
            )
            
            with self._lock:
                self.results[task.task_id] = task_result
                self._stats['completed_tasks'] += 1
                if task_result.execution_time:
                    self._stats['total_execution_time'] += task_result.execution_time
            
            logging.info(f"任务 {task.task_id} 执行成功")
            
            # 触发任务完成事件
            self._trigger_event('task_complete', task, task_result)
            
            # 通知等待的线程
            with self._condition:
                self._condition.notify_all()
            
        except Exception as e:
            end_time = datetime.now()
            
            with task._lock:
                task.status = TaskStatus.FAILED
                task.error = e
                task.end_time = end_time
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=e,
                start_time=task.start_time,
                end_time=end_time,
                metadata=task.metadata.copy()
            )
            
            with self._lock:
                self.results[task.task_id] = task_result
                self._stats['failed_tasks'] += 1
            
            logging.error(f"任务 {task.task_id} 执行失败: {e}")
            
            # 触发任务失败事件
            self._trigger_event('task_failed', task, task_result)
            
            # 通知等待的线程
            with self._condition:
                self._condition.notify_all()
    
    def run(self, context: Dict = None) -> Dict[str, TaskResult]:
        """运行所有任务"""
        if not self._validate_dag():
            raise ValueError("任务依赖关系存在循环，无法执行")
        
        self.running = True
        context = context or {}
        
        # 触发调度器开始事件
        self._trigger_event('scheduler_start', None, None)
        
        try:
            while self.running:
                with self._lock:
                    if self.paused:
                        self._condition.wait()
                        continue
                    
                    ready_tasks = self._get_ready_tasks()
                    
                    if not ready_tasks:
                        # 检查是否所有任务都已完成或失败
                        all_done = all(
                            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, 
                                          TaskStatus.CANCELLED, TaskStatus.SKIPPED] 
                            for task in self.tasks.values()
                        )
                        if all_done:
                            break
                        
                        # 等待任务状态变化
                        self._condition.wait(timeout=0.1)
                        continue
                    
                    # 按优先级排序
                    ready_tasks.sort(key=lambda t: -t.priority)
                    
                    # 提交任务执行
                    for task in ready_tasks[:self.max_workers]:
                        self.executor.submit(self._execute_task, task, context)
                
                time.sleep(0.1)
        finally:
            self.executor.shutdown(wait=True)
            self.running = False
            
            # 触发调度器结束事件
            self._trigger_event('scheduler_stop', None, None)
        
        return self.results
    
    def pause(self):
        """暂停调度器"""
        with self._lock:
            self.paused = True
    
    def resume(self):
        """恢复调度器"""
        with self._lock:
            self.paused = False
            self._condition.notify_all()
    
    def cancel_task(self, task_id: str):
        """取消任务"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    task.end_time = datetime.now()
                    
                    task_result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        start_time=task.start_time,
                        end_time=task.end_time
                    )
                    
                    self.results[task_id] = task_result
                    self._stats['cancelled_tasks'] += 1
                    
                    # 触发任务取消事件
                    self._trigger_event('task_cancelled', task, task_result)
    
    def skip_task(self, task_id: str, reason: str = None):
        """跳过任务"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
                    task.end_time = datetime.now()
                    if reason:
                        task.set_metadata('skip_reason', reason)
                    
                    task_result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED,
                        start_time=task.start_time,
                        end_time=task.end_time,
                        metadata=task.metadata.copy()
                    )
                    
                    self.results[task_id] = task_result
                    self._stats['skipped_tasks'] += 1
                    
                    # 触发任务跳过事件
                    self._trigger_event('task_skipped', task, task_result)
    
    def stop(self):
        """停止调度器"""
        self.running = False
        with self._condition:
            self._condition.notify_all()
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            if task_id in self.tasks:
                return self.tasks[task_id].status
            return None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        with self._lock:
            return self.results.get(task_id)
    
    def get_stats(self) -> Dict:
        """获取调度器统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats['pending_tasks'] = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            stats['running_tasks'] = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
            return stats
    
    def get_task_graph(self) -> nx.DiGraph:
        """获取任务依赖图"""
        return self.dag.copy()
    
    def visualize_graph(self, output_file: str = None):
        """可视化任务依赖图"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # 创建位置布局
            pos = nx.spring_layout(self.dag)
            
            # 绘制节点
            node_colors = []
            for node in self.dag.nodes():
                task = self.tasks[node]
                if task.status == TaskStatus.COMPLETED:
                    node_colors.append('green')
                elif task.status == TaskStatus.FAILED:
                    node_colors.append('red')
                elif task.status == TaskStatus.RUNNING:
                    node_colors.append('blue')
                elif task.status == TaskStatus.CANCELLED:
                    node_colors.append('orange')
                elif task.status == TaskStatus.SKIPPED:
                    node_colors.append('gray')
                else:
                    node_colors.append('lightgray')
            
            nx.draw_networkx_nodes(self.dag, pos, node_color=node_colors, node_size=1000)
            nx.draw_networkx_edges(self.dag, pos, edge_color='black', arrows=True)
            nx.draw_networkx_labels(self.dag, pos, font_size=10)
            
            plt.title("Task Dependency Graph")
            plt.axis('off')
            
            if output_file:
                plt.savefig(output_file)
                logging.info(f"任务依赖图已保存到: {output_file}")
            else:
                plt.show()
        except ImportError:
            logging.warning("matplotlib未安装，无法可视化任务依赖图")
        except Exception as e:
            logging.error(f"可视化任务依赖图失败: {e}")

# ==================== 基础任务实现 ====================

class LoadDataTask(Task):
    """数据加载任务"""
    
    def __init__(self, data_path: str, cache_manager: CacheManager = None, **kwargs):
        super().__init__("load_data", "加载数据", **kwargs)
        self.data_path = data_path
        self.cache_manager = cache_manager
    
    def execute(self, context: Dict = None) -> pd.DataFrame:
        """执行数据加载"""
        context = context or {}
        data_processor = context.get('data_processor')
        if not data_processor:
            raise ValueError("上下文中缺少data_processor")
        
        # 尝试从缓存加载
        cache_key = f"load_data_{self.data_path}"
        if self.cache_manager:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                logging.info(f"从缓存加载数据: {self.data_path}")
                return cached_data
        
        # 从文件加载
        data = data_processor.load_data(self.data_path)
        
        # 缓存数据
        if self.cache_manager:
            self.cache_manager.set(cache_key, data)
        
        return data

class PreprocessDataTask(Task):
    """数据预处理任务"""
    
    def __init__(self, **kwargs):
        super().__init__("preprocess_data", "预处理数据", priority=1, **kwargs)
    
    def execute(self, context: Dict = None) -> pd.DataFrame:
        """执行数据预处理"""
        context = context or {}
        data_processor = context.get('data_processor')
        if not data_processor:
            raise ValueError("上下文中缺少data_processor")
        
        raw_data = context.get('raw_data')
        if raw_data is None:
            raise ValueError("上下文中缺少raw_data")
        
        # 尝试从缓存加载
        cache_manager = context.get('cache_manager')
        if cache_manager:
            cache_key = f"preprocess_data_{hash(str(raw_data.values.tobytes()))}"
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                logging.info("从缓存加载预处理数据")
                return cached_data
        
        # 预处理数据
        processed_data = data_processor.preprocess_data(raw_data)
        
        # 缓存预处理结果
        if cache_manager:
            cache_manager.set(cache_key, processed_data)
        
        return processed_data

# ==================== 主程序入口 ====================

def main():
    """示例使用优化后的代码拆分方案"""
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
    # data_processor = OptimizedDataProcessor(config_manager, cache_manager, error_handler)
    
    # 创建任务调度器
    scheduler = DAGTaskScheduler(max_workers=2, config_manager=config_manager)
    
    # 添加事件处理器
    def on_task_complete(task, result):
        logging.info(f"任务 {task.task_id} 完成，耗时: {result.execution_time:.2f}秒")
    
    def on_task_failed(task, result):
        logging.error(f"任务 {task.task_id} 失败: {result.error}")
    
    scheduler.add_event_handler('task_complete', on_task_complete)
    scheduler.add_event_handler('task_failed', on_task_failed)
    
    # 创建任务实例
    load_task = LoadDataTask("data.csv", cache_manager=cache_manager)
    preprocess_task = PreprocessDataTask()
    
    # 添加任务到调度器
    scheduler.add_task(load_task)
    scheduler.add_task(preprocess_task)
    
    # 设置任务依赖
    scheduler.add_dependency("preprocess_data", "load_data")
    
    # 准备执行上下文
    context = {
        # 'data_processor': data_processor,
        'cache_manager': cache_manager,
        'error_handler': error_handler
    }
    
    try:
        # 运行任务
        results = scheduler.run(context)
        
        # 输出结果
        for task_id, result in results.items():
            print(f"任务 {task_id}: {result.status.value}")
            if result.error:
                print(f"  错误: {result.error}")
            elif result.result is not None:
                print(f"  结果类型: {type(result.result)}")
                if hasattr(result.result, 'shape'):
                    print(f"  数据形状: {result.result.shape}")
        
        # 输出统计信息
        stats = scheduler.get_stats()
        print(f"任务统计: {stats}")
        
        # 可视化任务依赖图
        scheduler.visualize_graph("task_graph.png")
        
    except KeyboardInterrupt:
        logging.info("用户中断，正在停止调度器...")
        scheduler.stop()
    except Exception as e:
        logging.error(f"执行失败: {e}")
        scheduler.stop()
    finally:
        # 清理资源
        config_manager.stop()

if __name__ == "__main__":
    main()