"""
优化后的可视化模块
支持多种图表类型和可视化后端
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import io
import base64
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path

# 导入基础系统组件
from jichuxitong import ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError, DataProcessingError, exception_handler, cache_result
from enhanced_factor_analysis_system import (
    ConfigManager, CacheManager, ErrorHandler, FactorAnalysisError,
    VisualizationError, exception_handler, cache_result
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChartType:
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    PAIR = "pair"
    CORRELATION = "correlation"

class VisualizationBackend:
    """可视化后端"""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"

class BaseVisualizer(ABC):
    """可视化器基类"""
    
    def __init__(self, config_manager: ConfigManager, 
                 cache_manager: CacheManager,
                 error_handler: ErrorHandler):
        self.config = config_manager
        self.cache = cache_manager
        self.error_handler = error_handler
        self.backend = config_manager.get('visualization.backend', VisualizationBackend.MATPLOTLIB)
        
        # 设置样式
        self._setup_style()
    
    @abstractmethod
    def _setup_style(self):
        """设置样式"""
        pass
    
    @abstractmethod
    def create_line_chart(self, data: pd.DataFrame, x: str, y: str, 
                         title: str = None, **kwargs) -> Any:
        """创建折线图"""
        pass
    
    @abstractmethod
    def create_bar_chart(self, data: pd.DataFrame, x: str, y: str, 
                        title: str = None, **kwargs) -> Any:
        """创建柱状图"""
        pass
    
    @abstractmethod
    def create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                          title: str = None, **kwargs) -> Any:
        """创建散点图"""
        pass
    
    @abstractmethod
    def create_histogram(self, data: pd.Series, bins: int = 30, 
                         title: str = None, **kwargs) -> Any:
        """创建直方图"""
        pass
    
    @abstractmethod
    def create_heatmap(self, data: pd.DataFrame, title: str = None, **kwargs) -> Any:
        """创建热力图"""
        pass
    
    @abstractmethod
    def save_chart(self, chart: Any, path: str, **kwargs) -> str:
        """保存图表"""
        pass

class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib可视化器"""
    
    def _setup_style(self):
        """设置样式"""
        plt.style.use(self.config.get('visualization.matplotlib.style', 'default'))
        self.figure_size = self.config.get('visualization.matplotlib.figure_size', (10, 6))
        self.dpi = self.config.get('visualization.matplotlib.dpi', 100)
        self.color_palette = self.config.get('visualization.matplotlib.color_palette', 'viridis')
    
    @exception_handler()
    def create_line_chart(self, data: pd.DataFrame, x: str, y: str, 
                         title: str = None, **kwargs) -> plt.Figure:
        """创建折线图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.plot(data[x], data[y], **kwargs)
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_bar_chart(self, data: pd.DataFrame, x: str, y: str, 
                        title: str = None, **kwargs) -> plt.Figure:
        """创建柱状图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.bar(data[x], data[y], **kwargs)
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                          title: str = None, **kwargs) -> plt.Figure:
        """创建散点图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.scatter(data[x], data[y], **kwargs)
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_histogram(self, data: pd.Series, bins: int = 30, 
                         title: str = None, **kwargs) -> plt.Figure:
        """创建直方图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.hist(data, bins=bins, **kwargs)
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(data.name if data.name else 'Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_heatmap(self, data: pd.DataFrame, title: str = None, **kwargs) -> plt.Figure:
        """创建热力图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.heatmap(data, annot=True, cmap=self.color_palette, 
                   ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def save_chart(self, chart: plt.Figure, path: str, **kwargs) -> str:
        """保存图表"""
        format_type = kwargs.get('format', 'png')
        chart.savefig(path, format=format_type, dpi=self.dpi, bbox_inches='tight')
        plt.close(chart)
        return path

class PlotlyVisualizer(BaseVisualizer):
    """Plotly可视化器"""
    
    def _setup_style(self):
        """设置样式"""
        self.template = self.config.get('visualization.plotly.template', 'plotly_white')
        self.color_palette = self.config.get('visualization.plotly.color_palette', px.colors.qualitative.Plotly)
    
    @exception_handler()
    def create_line_chart(self, data: pd.DataFrame, x: str, y: str, 
                         title: str = None, **kwargs) -> go.Figure:
        """创建折线图"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='lines',
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            template=self.template
        )
        
        return fig
    
    @exception_handler()
    def create_bar_chart(self, data: pd.DataFrame, x: str, y: str, 
                        title: str = None, **kwargs) -> go.Figure:
        """创建柱状图"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data[x],
            y=data[y],
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            template=self.template
        )
        
        return fig
    
    @exception_handler()
    def create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                          title: str = None, **kwargs) -> go.Figure:
        """创建散点图"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='markers',
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            template=self.template
        )
        
        return fig
    
    @exception_handler()
    def create_histogram(self, data: pd.Series, bins: int = 30, 
                         title: str = None, **kwargs) -> go.Figure:
        """创建直方图"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=data.name if data.name else 'Value',
            yaxis_title='Frequency',
            template=self.template
        )
        
        return fig
    
    @exception_handler()
    def create_heatmap(self, data: pd.DataFrame, title: str = None, **kwargs) -> go.Figure:
        """创建热力图"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            template=self.template
        )
        
        return fig
    
    @exception_handler()
    def save_chart(self, chart: go.Figure, path: str, **kwargs) -> str:
        """保存图表"""
        format_type = kwargs.get('format', 'html')
        
        if format_type == 'html':
            chart.write_html(path)
        elif format_type == 'png':
            chart.write_image(path)
        elif format_type == 'pdf':
            chart.write_image(path)
        elif format_type == 'svg':
            chart.write_image(path)
        
        return path

class SeabornVisualizer(BaseVisualizer):
    """Seaborn可视化器"""
    
    def _setup_style(self):
        """设置样式"""
        sns.set_style(self.config.get('visualization.seaborn.style', 'whitegrid'))
        self.figure_size = self.config.get('visualization.seaborn.figure_size', (10, 6))
        self.dpi = self.config.get('visualization.seaborn.dpi', 100)
        self.color_palette = self.config.get('visualization.seaborn.color_palette', 'viridis')
    
    @exception_handler()
    def create_line_chart(self, data: pd.DataFrame, x: str, y: str, 
                         title: str = None, **kwargs) -> plt.Figure:
        """创建折线图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.lineplot(data=data, x=x, y=y, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_bar_chart(self, data: pd.DataFrame, x: str, y: str, 
                        title: str = None, **kwargs) -> plt.Figure:
        """创建柱状图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.barplot(data=data, x=x, y=y, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                          title: str = None, **kwargs) -> plt.Figure:
        """创建散点图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.scatterplot(data=data, x=x, y=y, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_histogram(self, data: pd.Series, bins: int = 30, 
                         title: str = None, **kwargs) -> plt.Figure:
        """创建直方图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.histplot(data, bins=bins, ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def create_heatmap(self, data: pd.DataFrame, title: str = None, **kwargs) -> plt.Figure:
        """创建热力图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sns.heatmap(data, annot=True, cmap=self.color_palette, 
                   ax=ax, **kwargs)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @exception_handler()
    def save_chart(self, chart: plt.Figure, path: str, **kwargs) -> str:
        """保存图表"""
        format_type = kwargs.get('format', 'png')
        chart.savefig(path, format=format_type, dpi=self.dpi, bbox_inches='tight')
        plt.close(chart)
        return path

class FactorVisualizer:
    """因子可视化器"""
    
    def __init__(self, config_manager: ConfigManager, 
                 cache_manager: CacheManager,
                 error_handler: ErrorHandler):
        self.config = config_manager
        self.cache = cache_manager
        self.error_handler = error_handler
        
        # 初始化可视化器
        self.visualizers = {
            VisualizationBackend.MATPLOTLIB: MatplotlibVisualizer(config_manager, cache_manager, error_handler),
            VisualizationBackend.PLOTLY: PlotlyVisualizer(config_manager, cache_manager, error_handler),
            VisualizationBackend.SEABORN: SeabornVisualizer(config_manager, cache_manager, error_handler)
        }
        
        self.default_backend = config_manager.get('visualization.backend', VisualizationBackend.MATPLOTLIB)
        self.output_dir = config_manager.get('visualization.output_dir', 'charts')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_visualizer(self, backend: str = None) -> BaseVisualizer:
        """获取可视化器"""
        backend = backend or self.default_backend
        if backend not in self.visualizers:
            backend = self.default_backend
        
        return self.visualizers[backend]
    
    @cache_result()
    @exception_handler()
    def plot_factor_distribution(self, factor_data: pd.DataFrame, 
                                 factor_name: str,
                                 backend: str = None,
                                 save_path: str = None) -> str:
        """绘制因子分布"""
        visualizer = self._get_visualizer(backend)
        
        # 获取因子数据
        if factor_name not in factor_data.columns:
            raise VisualizationError(f"因子 {factor_name} 不存在于数据中")
        
        factor_values = factor_data[factor_name].dropna()
        
        # 创建直方图
        chart = visualizer.create_histogram(
            factor_values,
            bins=self.config.get('visualization.factor_distribution.bins', 30),
            title=f"{factor_name} 分布"
        )
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{factor_name}_distribution.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @cache_result()
    @exception_handler()
    def plot_ic_series(self, ic_data: pd.DataFrame, 
                      factor_name: str,
                      backend: str = None,
                      save_path: str = None) -> str:
        """绘制IC序列"""
        visualizer = self._get_visualizer(backend)
        
        # 创建折线图
        chart = visualizer.create_line_chart(
            ic_data,
            x='date',
            y='ic',
            title=f"{factor_name} IC序列"
        )
        
        # 添加零线
        if backend == VisualizationBackend.PLOTLY:
            chart.add_hline(y=0, line_dash="dash", line_color="gray")
        else:
            chart.axhline(y=0, color='gray', linestyle='--')
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{factor_name}_ic_series.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @cache_result()
    @exception_handler()
    def plot_group_returns(self, group_return_data: pd.DataFrame,
                           factor_name: str,
                           backend: str = None,
                           save_path: str = None) -> str:
        """绘制分组收益"""
        visualizer = self._get_visualizer(backend)
        
        # 创建折线图
        chart = visualizer.create_line_chart(
            group_return_data,
            x='date',
            y='return',
            color='group',
            title=f"{factor_name} 分组收益"
        )
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{factor_name}_group_returns.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @cache_result()
    @exception_handler()
    def plot_factor_correlation(self, factor_data: pd.DataFrame,
                                backend: str = None,
                                save_path: str = None) -> str:
        """绘制因子相关性热力图"""
        visualizer = self._get_visualizer(backend)
        
        # 计算相关性矩阵
        corr_matrix = factor_data.corr()
        
        # 创建热力图
        chart = visualizer.create_heatmap(
            corr_matrix,
            title="因子相关性热力图"
        )
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"factor_correlation.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @cache_result()
    @exception_handler()
    def plot_factor_performance(self, factor_performance: pd.DataFrame,
                                 backend: str = None,
                                 save_path: str = None) -> str:
        """绘制因子表现"""
        visualizer = self._get_visualizer(backend)
        
        # 创建柱状图
        chart = visualizer.create_bar_chart(
            factor_performance,
            x='factor',
            y='score',
            title="因子表现评分"
        )
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"factor_performance.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @cache_result()
    @exception_handler()
    def plot_factor_ranking(self, factor_ranking: pd.DataFrame,
                           backend: str = None,
                           save_path: str = None) -> str:
        """绘制因子排名"""
        visualizer = self._get_visualizer(backend)
        
        # 创建柱状图
        chart = visualizer.create_bar_chart(
            factor_ranking,
            x='factor',
            y='rank',
            title="因子排名"
        )
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"factor_ranking.{backend or self.default_backend}")
        
        return visualizer.save_chart(chart, save_path)
    
    @exception_handler()
    def create_factor_analysis_dashboard(self, factor_data: Dict[str, pd.DataFrame],
                                        ic_data: Dict[str, pd.DataFrame],
                                        group_return_data: Dict[str, pd.DataFrame],
                                        factor_performance: pd.DataFrame,
                                        factor_ranking: pd.DataFrame,
                                        backend: str = None,
                                        save_path: str = None) -> str:
        """创建因子分析仪表板"""
        backend = backend or self.default_backend
        
        if backend == VisualizationBackend.PLOTLY:
            # 使用Plotly创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("因子表现评分", "因子排名", "因子相关性", "IC分布"),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "heatmap"}, {"type": "histogram"}]]
            )
            
            # 添加因子表现评分
            fig.add_trace(
                go.Bar(x=factor_performance['factor'], y=factor_performance['score'],
                      name="因子表现评分"),
                row=1, col=1
            )
            
            # 添加因子排名
            fig.add_trace(
                go.Bar(x=factor_ranking['factor'], y=factor_ranking['rank'],
                      name="因子排名"),
                row=1, col=2
            )
            
            # 添加因子相关性
            all_factor_data = pd.concat(factor_data.values(), axis=1)
            corr_matrix = all_factor_data.corr()
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.index,
                          name="因子相关性"),
                row=2, col=1
            )
            
            # 添加IC分布
            all_ic_data = pd.concat(ic_data.values(), axis=0)
            fig.add_trace(
                go.Histogram(x=all_ic_data['ic'], name="IC分布"),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="因子分析仪表板",
                height=800,
                showlegend=False
            )
            
            # 保存图表
            if save_path is None:
                save_path = os.path.join(self.output_dir, "factor_analysis_dashboard.html")
            
            fig.write_html(save_path)
            return save_path
        
        else:
            # 使用Matplotlib创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 因子表现评分
            axes[0, 0].bar(factor_performance['factor'], factor_performance['score'])
            axes[0, 0].set_title("因子表现评分")
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 因子排名
            axes[0, 1].bar(factor_ranking['factor'], factor_ranking['rank'])
            axes[0, 1].set_title("因子排名")
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 因子相关性
            all_factor_data = pd.concat(factor_data.values(), axis=1)
            corr_matrix = all_factor_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title("因子相关性")
            
            # IC分布
            all_ic_data = pd.concat(ic_data.values(), axis=0)
            axes[1, 1].hist(all_ic_data['ic'], bins=30)
            axes[1, 1].set_title("IC分布")
            axes[1, 1].axvline(x=0, color='red', linestyle='--')
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                save_path = os.path.join(self.output_dir, "factor_analysis_dashboard.png")
            
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return save_path
    
    @exception_handler()
    def export_chart_as_base64(self, chart_path: str) -> str:
        """将图表导出为Base64字符串"""
        with open(chart_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded_string

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建可视化器
    config = ConfigManager("config.yaml", "production")
    cache = CacheManager(100, 1000)
    error_handler = ErrorHandler(config)
    
    visualizer = FactorVisualizer(config, cache, error_handler)
    
    # 模拟数据
    factor_data = pd.DataFrame({
        'factor_1': np.random.normal(0, 1, 1000),
        'factor_2': np.random.normal(0, 1, 1000),
        'factor_3': np.random.normal(0, 1, 1000)
    })
    
    ic_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'ic': np.random.normal(0.05, 0.1, 100)
    })
    
    group_return_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'group': np.random.choice(['G1', 'G2', 'G3', 'G4', 'G5'], 100),
        'return': np.random.normal(0.01, 0.05, 100)
    })
    
    factor_performance = pd.DataFrame({
        'factor': ['factor_1', 'factor_2', 'factor_3'],
        'score': [0.8, 0.6, 0.7]
    })
    
    factor_ranking = pd.DataFrame({
        'factor': ['factor_1', 'factor_2', 'factor_3'],
        'rank': [1, 3, 2]
    })
    
    try:
        # 绘制因子分布
        dist_path = visualizer.plot_factor_distribution(factor_data, 'factor_1')
        print(f"因子分布图已保存到: {dist_path}")
        
        # 绘制IC序列
        ic_path = visualizer.plot_ic_series(ic_data, 'factor_1')
        print(f"IC序列图已保存到: {ic_path}")
        
        # 绘制分组收益
        group_path = visualizer.plot_group_returns(group_return_data, 'factor_1')
        print(f"分组收益图已保存到: {group_path}")
        
        # 绘制因子相关性
        corr_path = visualizer.plot_factor_correlation(factor_data)
        print(f"因子相关性图已保存到: {corr_path}")
        
        # 绘制因子表现
        perf_path = visualizer.plot_factor_performance(factor_performance)
        print(f"因子表现图已保存到: {perf_path}")
        
        # 绘制因子排名
        rank_path = visualizer.plot_factor_ranking(factor_ranking)
        print(f"因子排名图已保存到: {rank_path}")
        
        # 创建因子分析仪表板
        dashboard_path = visualizer.create_factor_analysis_dashboard(
            {'factor_1': factor_data[['factor_1']]},
            {'factor_1': ic_data},
            {'factor_1': group_return_data},
            factor_performance,
            factor_ranking
        )
        print(f"因子分析仪表板已保存到: {dashboard_path}")
        
    except Exception as e:
        print(f"可视化失败: {e}")