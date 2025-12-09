# 因子分析系统EXCEL解析优化方案

## 1. 项目背景与需求分析

### 1.1 当前问题
- 现有代码对EXCEL文件解析能力有限
- 面对复杂的表头结构、多工作表、合并单元格等场景适应性差
- 缺乏智能的数据区域识别机制
- 无法处理格式不规范的数据文件

### 1.2 优化目标
- **泛化能力**：适应各种EXCEL表格格式
- **智能化**：自动识别表头和数据区域
- **容错性**：优雅处理各种异常情况
- **可扩展性**：支持新格式的快速适配
- **配置化**：通过配置文件而非代码修改适应新表格

## 2. 整体架构设计

### 2.1 分层解析架构

```
┌─────────────────────────────────────────┐
│           配置层 (Config Layer)          │
│  ┌─────────────────────────────────────┐ │
│  │        解析规则配置                  │ │
│  │    - 表头映射规则                    │ │
│  │    - 数据区域识别规则                │ │
│  │    - 数据类型转换规则                │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│           解析引擎层 (Engine Layer)      │
│  ┌─────────────────────────────────────┐ │
│  │         智能解析器                   │ │
│  │    - 文件结构分析                    │ │
│  │    - 表头智能检测                    │ │
│  │    - 数据区域定位                    │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│           数据处理层 (Processor Layer)   │
│  ┌─────────────────────────────────────┐ │
│  │         数据处理器                   │ │
│  │    - 数据类型转换                    │ │
│  │    - 缺失值处理                      │ │
│  │    - 数据验证与清洗                  │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│           应用层 (Application Layer)     │
│  ┌─────────────────────────────────────┐ │
│  │       现有因子分析系统               │ │
│  │    - 数据加载接口                    │ │
│  │    - 统一数据格式输出                │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 2.2 核心组件设计

#### 2.2.1 ExcelIntelligentParser (智能解析器)
- **文件结构分析器**：识别工作表、数据区域、格式信息
- **表头智能检测器**：多种表头识别策略
- **数据区域定位器**：自动定位有效数据范围
- **格式解析器**：处理合并单元格、数字格式等

#### 2.2.2 DataTypeConverter (数据类型转换器)
- **智能类型推断**：基于数据内容推断数据类型
- **百分比处理**：智能处理百分比格式
- **日期时间解析**：多种日期格式识别
- **数值标准化**：处理千分位、货币符号等

#### 2.2.3 ConfigurationManager (配置管理器)
- **规则配置**：解析规则的动态配置
- **映射管理**：列名映射和字段转换
- **验证规则**：数据验证和清洗规则
- **扩展机制**：新规则的插件化扩展

## 3. 核心解析策略

### 3.1 多策略表头检测

```python
class HeaderDetectionStrategy:
    """表头检测策略基类"""
    
    def detect_header(self, sheet_data, **kwargs) -> HeaderInfo:
        """检测表头信息，返回位置、列数、字段名等"""
        pass

class MultipleRowHeaderStrategy(HeaderDetectionStrategy):
    """多行表头检测策略"""
    
    def detect_header(self, sheet_data, **kwargs) -> HeaderInfo:
        # 检测多行合并的表头
        # 处理跨行跨列的字段名
        pass

class SkipRowsHeaderStrategy(HeaderDetectionStrategy):
    """跳行表头检测策略"""
    
    def detect_header(self, sheet_data, **kwargs) -> HeaderInfo:
        # 跳过标题行、说明行等
        # 智能识别实际表头位置
        pass

class SmartHeaderStrategy(HeaderDetectionStrategy):
        """智能表头检测策略"""
    
    def detect_header(self, sheet_data, **kwargs) -> HeaderInfo:
        # 基于字段名模式匹配
        # 结合数据特征识别表头
        pass
```

### 3.2 数据区域智能定位

```python
class DataRegionLocator:
    """数据区域定位器"""
    
    def locate_data_region(self, sheet) -> DataRegion:
        """智能定位数据区域"""
        
        # 1. 空白行/列检测
        empty_rows = self.detect_empty_rows(sheet)
        empty_cols = self.detect_empty_cols(sheet)
        
        # 2. 数据密度分析
        data_density = self.analyze_data_density(sheet)
        
        # 3. 结构化区域识别
        structured_region = self.identify_structured_region(
            empty_rows, empty_cols, data_density
        )
        
        # 4. 返回优化后的数据区域
        return structured_region
```

### 3.3 数据类型智能转换

```python
class IntelligentDataConverter:
    """智能数据转换器"""
    
    def convert_data(self, data, column_info):
        """智能转换数据"""
        
        conversion_rules = [
            self.try_percentage_conversion,
            self.try_date_conversion,
            self.try_numeric_conversion,
            self.try_currency_conversion,
            self.try_text_cleaning
        ]
        
        for rule in conversion_rules:
            try:
                converted_data = rule(data, column_info)
                if self.is_meaningful_conversion(data, converted_data):
                    return converted_data
            except Exception as e:
                continue
        
        return data  # 保持原始数据
```

## 4. 配置系统设计

### 4.1 解析规则配置 (excel_parsing_rules.yaml)

```yaml
# 基础解析配置
base_config:
  # 默认工作表选择策略
  default_sheet_selection: "first_sheet"  # first_sheet | largest_sheet | by_name
  
  # 默认表头检测策略
  default_header_strategy: "smart_detection"  # smart_detection | multiple_row | skip_rows
  
  # 数据区域定位策略
  data_region_strategy: "density_analysis"  # density_analysis | boundary_detection
  
  # 空值处理策略
  na_values: ["", " ", "NA", "N/A", "NULL", "null", "-", "—", "--"]
  keep_default_na: true

# 字段映射配置
field_mappings:
  # 标准字段映射
  standard_fields:
    "股票代码": ["code", "stock_code", "证券代码", "代码"]
    "股票名称": ["name", "stock_name", "证券名称", "名称"]
    "信号日期": ["date", "signal_date", "日期", "交易日期"]
    "次日开盘买入持股两日收益率": ["return", "收益率", "收益", "return_rate"]
  
  # 数值字段配置
  numeric_fields:
    "当日回调":
      type: "float"
      required: true
      validation_rules:
        - min_value: -1
        - max_value: 1
        - treat_percentage: true
    
    "机构持股比例(%)":
      type: "percentage"
      required: false
      validation_rules:
        - min_value: 0
        - max_value: 1

# 工作表特定配置
sheet_configs:
  "Sheet1":
    header_row: 1
    data_start_row: 2
    header_strategy: "multiple_row"
    column_mapping:
      0: "股票代码"
      1: "股票名称"
      2: "信号日期"
      3: "次日开盘买入持股两日收益率"
  
  "详细数据":
    header_row: 0
    data_start_row: 1
    skip_rows: [0, 1]  # 跳过的行
    special_handling:
      merged_cells: true
      multi_level_header: true

# 错误处理配置
error_handling:
  # 容错级别
  tolerance_level: "medium"  # low | medium | high
  
  # 缺失数据处理
  missing_data_strategy: "use_available"  # use_available | skip_column | skip_row
  
  # 数据类型转换失败处理
  conversion_error_strategy: "keep_original"  # keep_original | skip_row | use_default
  
  # 自动修复策略
  auto_fixes:
    - fix_column_names: true
    - fix_data_types: true
    - fill_missing_dates: true
```

### 4.2 配置管理器实现

```python
class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or "excel_parsing_rules.yaml"
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_parsing_rules(self, sheet_name=None) -> dict:
        """获取特定工作表的解析规则"""
        sheet_config = self.config.get('sheet_configs', {}).get(sheet_name, {})
        return {
            **self.config.get('base_config', {}),
            **sheet_config
        }
    
    def get_field_mapping(self, column_name) -> str:
        """获取字段映射"""
        mappings = self.config.get('field_mappings', {}).get('standard_fields', {})
        for standard_name, aliases in mappings.items():
            if column_name in aliases or column_name == standard_name:
                return standard_name
        return column_name
```

## 5. 核心类实现

### 5.1 ExcelIntelligentParser 类

```python
# -*- coding: utf-8 -*-
"""
智能EXCEL解析器
支持复杂表格结构的多层次解析
"""

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import re
from pathlib import Path

from .fa_config import build_report_path
from .fa_intelligent_parsing_helpers import (
    HeaderDetectionStrategy,
    DataRegionLocator,
    IntelligentDataConverter,
    ConfigurationManager
)


class ExcelIntelligentParser:
    """智能EXCEL解析器"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigurationManager(config_path)
        self.header_detector = HeaderDetectionStrategy()
        self.region_locator = DataRegionLocator()
        self.data_converter = IntelligentDataConverter()
        
        # 解析统计
        self.parse_statistics = {
            'total_files': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'error_logs': []
        }
    
    def parse_excel_file(self, file_path: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        智能解析EXCEL文件
        
        Args:
            file_path: EXCEL文件路径
            sheet_name: 指定工作表名称，None表示自动选择
            
        Returns:
            解析结果字典
        """
        self.parse_statistics['total_files'] += 1
        
        try:
            # 1. 文件结构分析
            file_info = self._analyze_file_structure(file_path)
            
            # 2. 工作表选择
            if sheet_name is None:
                sheet_name = self._select_optimal_sheet(file_info)
            
            # 3. 加载工作表数据
            sheet_data = self._load_sheet_data(file_path, sheet_name)
            
            # 4. 表头检测
            header_info = self._detect_header(sheet_data, sheet_name)
            
            # 5. 数据区域定位
            data_region = self._locate_data_region(sheet_data, header_info, sheet_name)
            
            # 6. 数据提取
            raw_data = self._extract_data(sheet_data, data_region, header_info)
            
            # 7. 数据转换
            converted_data = self._convert_data(raw_data, sheet_name)
            
            # 8. 数据验证
            validated_data = self._validate_data(converted_data)
            
            # 9. 生成解析报告
            parse_report = self._generate_parse_report(
                file_path, sheet_name, header_info, data_region, validated_data
            )
            
            self.parse_statistics['successful_parses'] += 1
            
            return {
                'success': True,
                'data': validated_data,
                'file_info': file_info,
                'header_info': header_info,
                'data_region': data_region,
                'parse_report': parse_report
            }
            
        except Exception as e:
            self.parse_statistics['failed_parses'] += 1
            self.parse_statistics['error_logs'].append({
                'file': file_path,
                'sheet': sheet_name,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return {
                'success': False,
                'error': str(e),
                'file_info': {'path': file_path, 'sheet': sheet_name}
            }
    
    def _analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """分析文件结构"""
        try:
            workbook = load_workbook(file_path, data_only=True)
            
            sheet_info = {}
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                
                # 获取实际使用的数据范围
                max_row = sheet.max_row
                max_col = sheet.max_column
                
                # 分析数据密度
                data_density = self._calculate_data_density(sheet, max_row, max_col)
                
                sheet_info[sheet_name] = {
                    'max_row': max_row,
                    'max_column': max_col,
                    'data_density': data_density,
                    'has_data': data_density > 0.01
                }
            
            return {
                'file_path': file_path,
                'sheets': sheet_info,
                'total_sheets': len(workbook.sheetnames)
            }
            
        except Exception as e:
            raise Exception(f"文件结构分析失败: {e}")
    
    def _calculate_data_density(self, sheet, max_row: int, max_col: int) -> float:
        """计算数据密度"""
        non_empty_cells = 0
        total_cells = min(max_row * max_col, 10000)  # 限制检查范围
        
        for row in range(1, min(max_row + 1, 101)):  # 检查前100行
            for col in range(1, min(max_col + 1, 51)):  # 检查前50列
                cell_value = sheet.cell(row=row, column=col).value
                if cell_value is not None and str(cell_value).strip():
                    non_empty_cells += 1
        
        return non_empty_cells / total_cells if total_cells > 0 else 0
    
    def _select_optimal_sheet(self, file_info: Dict[str, Any]) -> str:
        """选择最优工作表"""
        sheets = file_info['sheets']
        
        # 按数据密度排序
        sorted_sheets = sorted(
            sheets.items(),
            key=lambda x: x[1]['data_density'],
            reverse=True
        )
        
        # 选择数据密度最高且有数据的工作表
        for sheet_name, info in sorted_sheets:
            if info['has_data'] and info['data_density'] > 0.1:
                return sheet_name
        
        # 如果没有高密度工作表，选择第一个有数据的工作表
        for sheet_name, info in sorted_sheets:
            if info['has_data']:
                return sheet_name
        
        # 如果都没有数据，选择第一个工作表
        return list(sheets.keys())[0]
    
    def _load_sheet_data(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """加载工作表数据"""
        try:
            # 使用pandas加载，保持原始格式
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=None,
                dtype=str  # 先全部作为字符串读取，后续再转换
            )
            return df
        except Exception as e:
            raise Exception(f"工作表数据加载失败: {e}")
    
    def _detect_header(self, sheet_data: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """检测表头"""
        parsing_rules = self.config_manager.get_parsing_rules(sheet_name)
        strategy_name = parsing_rules.get('header_strategy', 'smart_detection')
        
        # 根据策略选择检测方法
        if strategy_name == 'multiple_row':
            header_info = self._detect_multiple_row_header(sheet_data, parsing_rules)
        elif strategy_name == 'skip_rows':
            header_info = self._detect_skip_rows_header(sheet_data, parsing_rules)
        else:  # smart_detection
            header_info = self._smart_detect_header(sheet_data, parsing_rules)
        
        return header_info
    
    def _smart_detect_header(self, sheet_data: pd.DataFrame, parsing_rules: Dict) -> Dict[str, Any]:
        """智能表头检测"""
        header_candidates = []
        
        # 检测前10行作为可能的表头
        for row_idx in range(min(10, len(sheet_data))):
            row_data = sheet_data.iloc[row_idx].dropna()
            
            if len(row_data) == 0:
                continue
            
            # 计算该行作为表头的可能性分数
            score = self._calculate_header_score(row_data, row_idx)
            
            header_candidates.append({
                'row_index': row_idx,
                'score': score,
                'columns': list(row_data.index),
                'values': list(row_data.values)
            })
        
        # 选择得分最高的候选者
        best_candidate = max(header_candidates, key=lambda x: x['score'])
        
        # 生成字段映射
        field_mapping = self._generate_field_mapping(best_candidate['values'])
        
        return {
            'header_row': best_candidate['row_index'],
            'column_count': len(best_candidate['values']),
            'field_mapping': field_mapping,
            'raw_header': best_candidate['values'],
            'confidence': best_candidate['score']
        }
    
    def _calculate_header_score(self, row_data: pd.Series, row_idx: int) -> float:
        """计算表头可能性分数"""
        score = 0.0
        
        # 字段名模式匹配分数
        standard_fields = self.config_manager.config.get('field_mappings', {}).get('standard_fields', {})
        
        for value in row_data.values:
            if pd.isna(value):
                continue
            
            value_str = str(value).strip()
            
            # 匹配标准字段名
            for standard_name, aliases in standard_fields.items():
                if value_str in aliases or value_str == standard_name:
                    score += 10.0
                    break
            
            # 匹配常见表头模式
            if re.match(r'.*代码.*|.*名称.*|.*日期.*|.*收益率.*|.*比例.*', value_str):
                score += 5.0
            
            # 避免纯数字作为表头
            if value_str.replace('.', '').replace('-', '').isdigit():
                score -= 3.0
        
        # 行位置加权（前面的行得分更高）
        score *= (10 - row_idx) / 10
        
        return score
    
    def _generate_field_mapping(self, header_values: List[str]) -> Dict[str, str]:
        """生成字段映射"""
        mapping = {}
        
        for i, value in enumerate(header_values):
            if pd.isna(value):
                continue
            
            value_str = str(value).strip()
            standard_field = self.config_manager.get_field_mapping(value_str)
            mapping[f'col_{i}'] = standard_field
        
        return mapping
    
    def _locate_data_region(self, sheet_data: pd.DataFrame, header_info: Dict, sheet_name: str) -> Dict[str, Any]:
        """定位数据区域"""
        parsing_rules = self.config_manager.get_parsing_rules(sheet_name)
        
        header_row = header_info['header_row']
        
        # 数据起始行
        data_start_row = header_row + 1
        
        # 处理跳行配置
        skip_rows = parsing_rules.get('skip_rows', [])
        if header_row in skip_rows:
            skip_rows.remove(header_row)
        
        # 寻找最后一个有效数据行
        data_end_row = self._find_last_data_row(sheet_data, data_start_row, skip_rows)
        
        # 寻找最后一个有效数据列
        data_end_col = self._find_last_data_col(sheet_data, header_row, data_end_row)
        
        return {
            'start_row': data_start_row,
            'end_row': data_end_row,
            'start_col': 0,
            'end_col': data_end_col,
            'header_row': header_row,
            'skip_rows': skip_rows
        }
    
    def _find_last_data_row(self, sheet_data: pd.DataFrame, start_row: int, skip_rows: List[int]) -> int:
        """寻找最后有效数据行"""
        last_data_row = start_row
        
        for row_idx in range(start_row, len(sheet_data)):
            if row_idx in skip_rows:
                continue
            
            row_data = sheet_data.iloc[row_idx]
            # 检查是否至少有一个非空值
            if row_data.dropna().empty:
                continue
            
            # 检查是否包含有效数据（不全是空白字符）
            has_meaningful_data = any(
                str(val).strip() for val in row_data.values 
                if pd.notna(val) and str(val).strip()
            )
            
            if has_meaningful_data:
                last_data_row = row_idx
        
        return last_data_row + 1  # 返回下一个位置作为结束
    
    def _find_last_data_col(self, sheet_data: pd.DataFrame, header_row: int, end_row: int) -> int:
        """寻找最后有效数据列"""
        last_data_col = 0
        
        for row_idx in range(header_row, end_row):
            if row_idx >= len(sheet_data):
                break
            
            row_data = sheet_data.iloc[row_idx]
            
            # 从右往左查找最后一个非空列
            for col_idx in range(len(row_data) - 1, -1, -1):
                cell_value = row_data.iloc[col_idx]
                
                if pd.notna(cell_value) and str(cell_value).strip():
                    last_data_col = max(last_data_col, col_idx + 1)
                    break
        
        return last_data_col + 1  # 返回下一个位置作为结束
    
    def _extract_data(self, sheet_data: pd.DataFrame, data_region: Dict, header_info: Dict) -> pd.DataFrame:
        """提取数据"""
        start_row = data_region['start_row']
        end_row = data_region['end_row']
        start_col = data_region['start_col']
        end_col = data_region['end_col']
        
        # 提取数据区域
        raw_data = sheet_data.iloc[start_row:end_row, start_col:end_col].copy()
        
        # 设置列名
        field_mapping = header_info['field_mapping']
        raw_columns = [f'col_{i}' for i in range(len(raw_data.columns))]
        
        # 应用字段映射
        mapped_columns = []
        for i, col in enumerate(raw_columns):
            mapped_columns.append(field_mapping.get(col, col))
        
        raw_data.columns = mapped_columns
        
        return raw_data
    
    def _convert_data(self, raw_data: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """转换数据"""
        converted_data = raw_data.copy()
        
        # 获取数值字段配置
        numeric_fields = self.config_manager.config.get('field_mappings', {}).get('numeric_fields', {})
        
        for column in converted_data.columns:
            if column in numeric_fields:
                field_config = numeric_fields[column]
                converted_data[column] = self.data_converter.convert_column(
                    converted_data[column], field_config
                )
            else:
                # 智能类型转换
                converted_data[column] = self.data_converter.intelligent_convert(
                    converted_data[column], column
                )
        
        return converted_data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据"""
        validated_data = data.copy()
        
        # 基本验证规则
        validation_rules = [
            self._validate_required_columns,
            self._validate_data_types,
            self._validate_data_ranges,
            self._remove_empty_rows
        ]
        
        for rule in validation_rules:
            validated_data = rule(validated_data)
        
        return validated_data
    
    def _validate_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证必需列"""
        # 这里可以实现必需列的验证逻辑
        return data
    
    def _validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据类型"""
        # 这里可以实现数据类型的验证逻辑
        return data
    
    def _validate_data_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据范围"""
        # 这里可以实现数据范围的验证逻辑
        return data
    
    def _remove_empty_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """移除空行"""
        # 移除完全为空的行
        return data.dropna(how='all')
    
    def _generate_parse_report(self, file_path: str, sheet_name: str, 
                              header_info: Dict, data_region: Dict, 
                              final_data: pd.DataFrame) -> Dict[str, Any]:
        """生成解析报告"""
        return {
            'file_path': file_path,
            'sheet_name': sheet_name,
            'parse_timestamp': datetime.now(),
            'header_info': {
                'detected_row': header_info['header_row'],
                'column_count': header_info['column_count'],
                'confidence': header_info['confidence']
            },
            'data_info': {
                'start_row': data_region['start_row'],
                'end_row': data_region['end_row'],
                'total_rows': len(final_data),
                'total_columns': len(final_data.columns)
            },
            'data_quality': {
                'completeness': (1 - final_data.isnull().sum().sum() / final_data.size),
                'unique_rows': len(final_data.drop_duplicates()),
                'empty_rows_removed': len(final_data) - len(final_data.dropna(how='all'))
            }
        }
    
    def get_parse_statistics(self) -> Dict[str, Any]:
        """获取解析统计信息"""
        return self.parse_statistics.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.parse_statistics = {
            'total_files': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'error_logs': []
        }


# 使用示例
def enhanced_data_loading(file_paths: List[str]) -> pd.DataFrame:
    """增强的数据加载函数"""
    parser = ExcelIntelligentParser()
    all_data = []
    
    for file_path in file_paths:
        print(f"解析文件: {file_path}")
        result = parser.parse_excel_file(file_path)
        
        if result['success']:
            data = result['data']
            data['source_file'] = file_path
            data['source_sheet'] = result['file_info']['sheets'].get('selected_sheet', 'unknown')
            
            all_data.append(data)
            print(f"  ✓ 成功解析 {len(data)} 行数据")
        else:
            print(f"  ✗ 解析失败: {result['error']}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"总共合并 {len(combined_data)} 行数据")
        return combined_data
    else:
        print("警告: 没有成功解析任何数据")
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试代码
    parser = ExcelIntelligentParser()
    result = parser.parse_excel_file("test_file.xlsx")
    
    if result['success']:
        print("解析成功!")
        print(f"数据形状: {result['data'].shape}")
        print(f"列名: {list(result['data'].columns)}")
    else:
        print(f"解析失败: {result['error']}")
```

### 5.2 智能解析辅助类

```python
# -*- coding: utf-8 -*-
"""
智能解析辅助类集合
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime


class HeaderDetectionStrategy:
    """表头检测策略基类"""
    
    def detect_header(self, sheet_data, **kwargs) -> Dict[str, Any]:
        """检测表头信息"""
        raise NotImplementedError


class MultipleRowHeaderStrategy(HeaderDetectionStrategy):
    """多行表头检测策略"""
    
    def detect_header(self, sheet_data, **kwargs) -> Dict[str, Any]:
        """检测多行表头"""
        header_row = kwargs.get('header_row', 0)
        max_rows = kwargs.get('max_header_rows', 3)
        
        # 获取多行表头数据
        header_rows_data = []
        for i in range(header_row, min(header_row + max_rows, len(sheet_data))):
            row_data = sheet_data.iloc[i].dropna()
            if len(row_data) > 0:
                header_rows_data.append({
                    'row_index': i,
                    'data': list(row_data.values),
                    'columns': list(row_data.index)
                })
        
        # 合并多行表头
        merged_header = self._merge_multi_row_header(header_rows_data)
        
        return {
            'header_type': 'multiple_row',
            'header_rows': header_rows_data,
            'merged_header': merged_header,
            'header_row': header_row
        }
    
    def _merge_multi_row_header(self, header_rows_data: List[Dict]) -> List[str]:
        """合并多行表头"""
        if not header_rows_data:
            return []
        
        merged = []
        for i, row_info in enumerate(header_rows_data):
            for j, value in enumerate(row_info['data']):
                # 确保列表足够长
                while len(merged) <= j:
                    merged.append("")
                
                # 合并字段名
                if i == 0:
                    merged[j] = str(value)
                else:
                    if str(value) and str(value) != 'nan':
                        merged[j] = f"{merged[j]}_{str(value)}"
        
        return merged


class SkipRowsHeaderStrategy(HeaderDetectionStrategy):
    """跳行表头检测策略"""
    
    def detect_header(self, sheet_data, **kwargs) -> Dict[str, Any]:
        """检测跳行表头"""
        skip_rows = kwargs.get('skip_rows', [])
        
        # 寻找第一个有效表头行
        header_row = None
        for i in range(len(sheet_data)):
            if i in skip_rows:
                continue
            
            row_data = sheet_data.iloc[i].dropna()
            if len(row_data) > 0 and self._is_header_row(row_data.values):
                header_row = i
                break
        
        if header_row is None:
            raise Exception("无法找到有效表头")
        
        row_data = sheet_data.iloc[header_row].dropna()
        
        return {
            'header_type': 'skip_rows',
            'header_row': header_row,
            'skip_rows': skip_rows,
            'header_values': list(row_data.values),
            'confidence': self._calculate_header_confidence(row_data.values)
        }
    
    def _is_header_row(self, row_values: List[Any]) -> bool:
        """判断是否为表头行"""
        # 包含关键词的字段名比例
        keyword_count = 0
        total_count = 0
        
        for value in row_values:
            if pd.isna(value):
                continue
            
            total_count += 1
            value_str = str(value).strip()
            
            # 检查是否包含表头关键词
            if re.match(r'.*代码.*|.*名称.*|.*日期.*|.*率.*|.*比.*', value_str):
                keyword_count += 1
        
        return total_count > 0 and (keyword_count / total_count) > 0.3
    
    def _calculate_header_confidence(self, row_values: List[Any]) -> float:
        """计算表头置信度"""
        # 基于字段名模式和内容计算置信度
        confidence = 0.0
        
        for value in row_values:
            if pd.isna(value):
                continue
            
            value_str = str(value).strip()
            
            # 标准字段名匹配
            if re.match(r'^(股票代码|股票名称|信号日期|收益率)$', value_str):
                confidence += 0.3
            
            # 常见字段名模式
            elif re.match(r'.*代码.*|.*名称.*|.*日期.*|.*率.*', value_str):
                confidence += 0.1
        
        return min(confidence, 1.0)


class DataRegionLocator:
    """数据区域定位器"""
    
    def locate_data_region(self, sheet_data, header_info: Dict) -> Dict[str, Any]:
        """定位数据区域"""
        header_row = header_info.get('header_row', 0)
        
        # 寻找数据起始和结束位置
        data_start_row = self._find_data_start_row(sheet_data, header_row)
        data_end_row = self._find_data_end_row(sheet_data, data_start_row)
        data_end_col = self._find_data_end_col(sheet_data, header_row, data_end_row)
        
        return {
            'start_row': data_start_row,
            'end_row': data_end_row,
            'start_col': 0,
            'end_col': data_end_col,
            'header_row': header_row
        }
    
    def _find_data_start_row(self, sheet_data: pd.DataFrame, header_row: int) -> int:
        """寻找数据起始行"""
        for i in range(header_row + 1, len(sheet_data)):
            row_data = sheet_data.iloc[i]
            
            # 检查是否包含有效数据
            non_empty_count = sum(1 for val in row_data.values if pd.notna(val) and str(val).strip())
            
            if non_empty_count > 0:
                return i
        
        return header_row + 1
    
    def _find_data_end_row(self, sheet_data: pd.DataFrame, start_row: int) -> int:
        """寻找数据结束行"""
        last_data_row = start_row
        
        for i in range(start_row, len(sheet_data)):
            row_data = sheet_data.iloc[i]
            
            # 检查是否包含有效数据
            non_empty_count = sum(1 for val in row_data.values if pd.notna(val) and str(val).strip())
            
            if non_empty_count > 0:
                last_data_row = i
            else:
                # 如果连续多行为空，则认为数据结束
                if i - last_data_row > 2:
                    break
        
        return last_data_row + 1
    
    def _find_data_end_col(self, sheet_data: pd.DataFrame, header_row: int, end_row: int) -> int:
        """寻找数据结束列"""
        max_col = 0
        
        for i in range(header_row, min(end_row, len(sheet_data))):
            row_data = sheet_data.iloc[i]
            
            # 从右往左查找最后一个非空列
            for j in range(len(row_data) - 1, -1, -1):
                cell_value = row_data.iloc[j]
                
                if pd.notna(cell_value) and str(cell_value).strip():
                    max_col = max(max_col, j + 1)
                    break
        
        return max_col + 1


class IntelligentDataConverter:
    """智能数据转换器"""
    
    def __init__(self):
        # 定义各种数据类型的转换规则
        self.conversion_patterns = {
            'percentage': [
                (r'(\d+\.?\d*)%', lambda x: float(x.group(1)) / 100),
                (r'(\d+\.?\d*)', lambda x: float(x.group(1)) / 100)
            ],
            'currency': [
                (r'[¥￥]\s*(\d+\.?\d*)', lambda x: float(x.group(1))),
                (r'(\d+\.?\d*)\s*元', lambda x: float(x.group(1)))
            ],
            'number': [
                (r'(\d+\.?\d*)', lambda x: float(x.group(1)))
            ]
        }
    
    def convert_column(self, column_data: pd.Series, field_config: Dict) -> pd.Series:
        """转换单列数据"""
        data_type = field_config.get('type', 'auto')
        
        if data_type == 'percentage':
            return self._convert_percentage_column(column_data)
        elif data_type == 'currency':
            return self._convert_currency_column(column_data)
        elif data_type == 'float':
            return self._convert_numeric_column(column_data)
        else:
            return self.intelligent_convert(column_data, column_data.name)
    
    def intelligent_convert(self, column_data: pd.Series, column_name: str) -> pd.Series:
        """智能转换数据"""
        # 移除空值进行类型推断
        non_null_data = column_data.dropna()
        
        if len(non_null_data) == 0:
            return column_data
        
        # 转换为字符串进行分析
        str_data = non_null_data.astype(str)
        
        # 判断数据类型
        if self._is_percentage_column(str_data):
            return self._convert_percentage_column(column_data)
        elif self._is_date_column(str_data):
            return self._convert_date_column(column_data)
        elif self._is_numeric_column(str_data):
            return self._convert_numeric_column(column_data)
        else:
            return column_data  # 保持字符串类型
    
    def _is_percentage_column(self, str_data: pd.Series) -> bool:
        """判断是否为百分比列"""
        percentage_count = str_data.str.contains('%', na=False).sum()
        return percentage_count / len(str_data) > 0.5
    
    def _is_date_column(self, str_data: pd.Series) -> bool:
        """判断是否为日期列"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}-\d{2}',        # MM-DD
            r'\d{4}年\d{1,2}月\d{1,2}日'  # 中文日期
        ]
        
        for pattern in date_patterns:
            match_count = str_data.str.match(pattern, na=False).sum()
            if match_count / len(str_data) > 0.5:
                return True
        
        return False
    
    def _is_numeric_column(self, str_data: pd.Series) -> bool:
        """判断是否为数值列"""
        numeric_count = pd.to_numeric(str_data.str.replace(',', ''), errors='coerce').notna().sum()
        return numeric_count / len(str_data) > 0.7
    
    def _convert_percentage_column(self, column_data: pd.Series) -> pd.Series:
        """转换百分比列"""
        def convert_percentage(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # 处理百分比格式
            if '%' in value_str:
                # 移除%符号并转换为小数
                numeric_str = value_str.replace('%', '').strip()
                try:
                    return float(numeric_str) / 100
                except ValueError:
                    return np.nan
            else:
                # 假设是小数格式
                try:
                    return float(value_str)
                except ValueError:
                    return np.nan
        
        return column_data.apply(convert_percentage)
    
    def _convert_date_column(self, column_data: pd.Series) -> pd.Series:
        """转换日期列"""
        return pd.to_datetime(column_data, errors='coerce')
    
    def _convert_numeric_column(self, column_data: pd.Series) -> pd.Series:
        """转换数值列"""
        def convert_numeric(value):
            if pd.isna(value):
                return np.nan
            
            # 移除千分位逗号
            str_value = str(value).replace(',', '').strip()
            
            try:
                return float(str_value)
            except ValueError:
                return np.nan
        
        return column_data.apply(convert_numeric)
    
    def _convert_currency_column(self, column_data: pd.Series) -> pd.Series:
        """转换货币列"""
        def convert_currency(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # 移除货币符号
            cleaned_str = re.sub(r'[¥￥$]', '', value_str)
            
            # 移除千分位逗号
            cleaned_str = cleaned_str.replace(',', '')
            
            try:
                return float(cleaned_str)
            except ValueError:
                return np.nan
        
        return column_data.apply(convert_currency)


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "excel_parsing_rules.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'base_config': {
                'default_sheet_selection': 'first_sheet',
                'default_header_strategy': 'smart_detection',
                'data_region_strategy': 'density_analysis',
                'na_values': ["", " ", "NA", "N/A", "NULL", "null", "-", "—", "--"],
                'keep_default_na': True
            },
            'field_mappings': {
                'standard_fields': {
                    '股票代码': ['code', 'stock_code', '证券代码', '代码'],
                    '股票名称': ['name', 'stock_name', '证券名称', '名称'],
                    '信号日期': ['date', 'signal_date', '日期', '交易日期'],
                    '次日开盘买入持股两日收益率': ['return', '收益率', '收益', 'return_rate']
                },
                'numeric_fields': {
                    '当日回调': {
                        'type': 'float',
                        'required': True,
                        'validation_rules': [{'min_value': -1, 'max_value': 1}]
                    }
                }
            },
            'sheet_configs': {},
            'error_handling': {
                'tolerance_level': 'medium',
                'missing_data_strategy': 'use_available',
                'conversion_error_strategy': 'keep_original'
            }
        }
    
    def get_parsing_rules(self, sheet_name: str = None) -> Dict:
        """获取解析规则"""
        base_config = self.config.get('base_config', {})
        sheet_config = self.config.get('sheet_configs', {}).get(sheet_name, {})
        
        return {**base_config, **sheet_config}
    
    def get_field_mapping(self, column_name: str) -> str:
        """获取字段映射"""
        standard_fields = self.config.get('field_mappings', {}).get('standard_fields', {})
        
        for standard_name, aliases in standard_fields.items():
            if column_name in aliases or column_name == standard_name:
                return standard_name
        
        return column_name  # 如果没有找到映射，返回原名


# 实用工具函数
def detect_excel_structure(file_path: str) -> Dict[str, Any]:
    """检测EXCEL文件结构"""
    try:
        workbook = pd.ExcelFile(file_path)
        
        sheet_info = {}
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=10)
            
            # 分析工作表结构
            sheet_info[sheet_name] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'non_empty_rows': df.dropna(how='all').shape[0],
                'header_candidates': analyze_header_candidates(df),
                'data_preview': df.head(3).to_dict('records')
            }
        
        return {
            'file_path': file_path,
            'sheets': sheet_info,
            'recommended_sheet': recommend_sheet(sheet_info)
        }
        
    except Exception as e:
        return {'error': str(e)}


def analyze_header_candidates(df: pd.DataFrame) -> List[Dict]:
    """分析表头候选"""
    candidates = []
    
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        non_empty_count = row.notna().sum()
        
        if non_empty_count > 0:
            # 计算该行作为表头的可能性
            score = calculate_header_score(row.values)
            
            candidates.append({
                'row_index': i,
                'non_empty_count': non_empty_count,
                'score': score,
                'content': [str(val) for val in row.values if pd.notna(val)]
            })
    
    return sorted(candidates, key=lambda x: x['score'], reverse=True)


def calculate_header_score(row_values: List[Any]) -> float:
    """计算表头得分"""
    score = 0.0
    
    for value in row_values:
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        
        # 标准字段名匹配
        if re.match(r'^(股票代码|股票名称|信号日期|收益率)$', value_str):
            score += 10
        elif re.match(r'.*代码.*|.*名称.*|.*日期.*|.*率.*|.*比.*', value_str):
            score += 5
        elif value_str.replace('.', '').replace('-', '').isdigit():
            score -= 2  # 纯数字得分降低
    
    return score


def recommend_sheet(sheet_info: Dict[str, Any]) -> str:
    """推荐最佳工作表"""
    best_sheet = None
    best_score = -1
    
    for sheet_name, info in sheet_info.items():
        if 'error' in info:
            continue
        
        score = (
            info['non_empty_rows'] * 2 +  # 非空行数
            len(info['header_candidates']) * 5 +  # 表头候选数
            info['header_candidates'][0]['score'] if info['header_candidates'] else 0  # 最佳表头得分
        )
        
        if score > best_score:
            best_score = score
            best_sheet = sheet_name
    
    return best_sheet


if __name__ == "__main__":
    # 测试代码
    file_info = detect_excel_structure("test.xlsx")
    print(f"文件结构分析结果: {file_info}")
```

## 6. 与现有系统的集成

### 6.1 替换现有数据加载逻辑

```python
# 在 yinzifenxi_main.py 中替换原有加载逻辑

# 原有的加载逻辑
# def load_data(self):
#     # 现有简单的加载逻辑
#     pass

# 新的智能加载逻辑
def enhanced_load_data(self):
    """增强的数据加载方法"""
    from .fa_intelligent_excel_parser import ExcelIntelligentParser
    
    if self.data is not None:
        return True
    
    if not self.file_paths:
        print("[ERROR] 未找到任何数据文件路径，请检查配置")
        return False

    # 使用智能解析器
    parser = ExcelIntelligentParser()
    all_data_frames = []
    
    for file_path in self.file_paths:
        if not file_path:
            continue
            
        normalized_path = os.path.abspath(file_path)
        if not os.path.exists(normalized_path):
            print(f"[WARN] 数据文件不存在: {normalized_path}")
            continue
            
        print(f"[INFO] 智能解析文件: {os.path.basename(normalized_path)}")
        
        # 使用智能解析器解析文件
        parse_result = parser.parse_excel_file(normalized_path)
        
        if parse_result['success']:
            data = parse_result['data']
            
            # 添加数据来源信息
            data['source_file'] = normalized_path
            data['source_sheet'] = parse_result['file_info']['sheets'].get('selected_sheet', 'unknown')
            
            all_data_frames.append(data)
            
            print(f"[INFO] 智能解析成功: {len(data)} 行, {len(data.columns)} 列")
            print(f"  检测到字段: {list(data.columns)}")
            
            # 显示解析质量报告
            if 'parse_report' in parse_result:
                report = parse_result['parse_report']
                print(f"  数据质量: 完整性 {report['data_quality']['completeness']:.1%}")
                print(f"  表头置信度: {report['header_info']['confidence']:.1%}")
        else:
            print(f"[ERROR] 智能解析失败: {parse_result['error']}")
            
            # 回退到原有加载方法
            print("[INFO] 回退到原有加载方法...")
            try:
                if normalized_path.lower().endswith('.csv'):
                    df = pd.read_csv(normalized_path, encoding='utf-8-sig')
                elif normalized_path.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(normalized_path)
                else:
                    print(f"[WARN] 不支持的文件格式: {normalized_path}")
                    continue
                    
                if df is not None:
                    df['source_file'] = normalized_path
                    all_data_frames.append(df)
                    print(f"[INFO] 原有方法加载成功: {len(df)} 行")
            except Exception as e:
                print(f"[ERROR] 原有方法也加载失败: {e}")

    if not all_data_frames:
        print("[ERROR] 所有数据文件加载失败")
        return False

    # 合并所有数据
    if len(all_data_frames) == 1:
        self.data = all_data_frames[0]
    else:
        self.data = pd.concat(all_data_frames, ignore_index=True, sort=False)
        
    print(f"[INFO] 数据合并完成: {len(self.data)} 行, {len(self.data.columns)} 列 (合并 {len(all_data_frames)} 个文件)")
    
    # 显示解析统计
    stats = parser.get_parse_statistics()
    print(f"[INFO] 解析统计: 成功 {stats['successful_parses']}/{stats['total_files']}")
    
    return True
```

### 6.2 配置驱动的字段映射

```python
# 更新 fa_config.py 中的字段映射配置

# 原有配置
FACTOR_COLUMNS = [
    '当日回调',
    '机构持股比例(%)',
    # ...
]

# 新的智能字段映射
INTELLIGENT_FIELD_MAPPINGS = {
    # 主字段映射
    'primary_fields': {
        '股票代码': ['code', 'stock_code', '证券代码', '代码', 'stockcode'],
        '股票名称': ['name', 'stock_name', '证券名称', '名称', 'stockname'],
        '信号日期': ['date', 'signal_date', '日期', '交易日期', 'trading_date'],
        '次日开盘买入持股两日收益率': ['return', '收益率', '收益', 'return_rate', 'yield']
    },
    
    # 因子字段映射（支持多种命名）
    'factor_fields': {
        '当日回调': ['当日回调', '当日回撤', 'daily_drawdown', '回撤', 'drawdown'],
        '机构持股比例(%)': ['机构持股比例', '持股比例', 'institution_holding', '机构持股'],
        '流通市值(元)': ['流通市值', '市值', 'circulating_market_value', 'market_cap']
    },
    
    # 字段别名映射（支持拼写变体）
    'alias_mappings': {
        'stock_code': ['stockcode', 'stock_code', 'code'],
        'stock_name': ['stockname', 'stock_name', 'name'],
        'return_rate': ['return', 'returnrate', 'yield', '收益率', '收益']
    }
}

# 智能字段检测函数
def detect_fields_intelligently(df_columns):
    """智能检测字段"""
    detected_fields = {}
    
    # 构建所有可能的字段名映射
    all_mappings = {}
    
    # 添加主字段映射
    for primary, aliases in INTELLIGENT_FIELD_MAPPINGS['primary_fields'].items():
        all_mappings[primary] = aliases
    
    # 添加因子字段映射
    for primary, aliases in INTELLIGENT_FIELD_MAPPINGS['factor_fields'].items():
        all_mappings[primary] = aliases
    
    # 添加别名映射
    for alias, primaries in INTELLIGENT_FIELD_MAPPINGS['alias_mappings'].items():
        for primary in primaries:
            if primary not in all_mappings:
                all_mappings[primary] = [alias]
    
    # 智能匹配
    for col in df_columns:
        col_lower = str(col).strip().lower()
        
        for standard_name, variants in all_mappings.items():
            for variant in variants:
                variant_lower = str(variant).strip().lower()
                
                if (col_lower == variant_lower or 
                    col_lower in variant_lower or 
                    variant_lower in col_lower):
                    detected_fields[col] = standard_name
                    break
    
    return detected_fields
```

## 7. 扩展功能设计

### 7.1 插件化扩展机制

```python
# -*- coding: utf-8 -*-
"""
解析插件系统
支持自定义解析逻辑的动态加载
"""

import importlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ParsingPlugin(ABC):
    """解析插件基类"""
    
    @abstractmethod
    def can_handle(self, file_info: Dict[str, Any]) -> bool:
        """判断是否能处理该文件"""
        pass
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """解析文件"""
        pass
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, str]:
        """获取插件信息"""
        pass

class ExcelParsingPluginManager:
    """解析插件管理器"""
    
    def __init__(self):
        self.plugins: List[ParsingPlugin] = []
        self.load_builtin_plugins()
        self.load_custom_plugins()
    
    def load_builtin_plugins(self):
        """加载内置插件"""
        # 内置的Excel解析插件
        from .plugins.excel_standard import StandardExcelPlugin
        from .plugins.excel_advanced import AdvancedExcelPlugin
        from .plugins.excel_legacy import LegacyExcelPlugin
        
        self.register_plugin(StandardExcelPlugin())
        self.register_plugin(AdvancedExcelPlugin())
        self.register_plugin(LegacyExcelPlugin())
    
    def load_custom_plugins(self):
        """加载自定义插件"""
        # 从指定目录加载自定义插件
        plugin_dir = Path("plugins")
        if plugin_dir.exists():
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("plugin_"):
                    try:
                        module = importlib.import_module(f"plugins.{plugin_file.stem}")
                        if hasattr(module, 'Plugin'):
                            plugin = module.Plugin()
                            self.register_plugin(plugin)
                    except Exception as e:
                        print(f"加载插件失败 {plugin_file}: {e}")
    
    def register_plugin(self, plugin: ParsingPlugin):
        """注册插件"""
        self.plugins.append(plugin)
    
    def find_best_plugin(self, file_path: str) -> ParsingPlugin:
        """找到最佳适配插件"""
        # 简单的文件信息
        file_info = {
            'file_path': file_path,
            'file_extension': Path(file_path).suffix.lower(),
            'file_size': Path(file_path).stat().st_size
        }
        
        # 遍历插件，找到第一个可以处理的
        for plugin in self.plugins:
            try:
                if plugin.can_handle(file_info):
                    return plugin
            except Exception:
                continue
        
        # 如果没有找到特殊插件，返回标准插件
        return self.plugins[0] if self.plugins else None
    
    def parse_with_best_plugin(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """使用最佳插件解析"""
        plugin = self.find_best_plugin(file_path)
        
        if plugin:
            return plugin.parse(file_path, **kwargs)
        else:
            return {'success': False, 'error': '没有可用的解析插件'}
```

### 7.2 自适应学习机制

```python
# -*- coding: utf-8 -*-
"""
自适应学习模块
从解析历史中学习，优化解析策略
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

class AdaptiveLearningEngine:
    """自适应学习引擎"""
    
    def __init__(self, learning_data_path: str = "learning_data.json"):
        self.learning_data_path = Path(learning_data_path)
        self.learning_data = self.load_learning_data()
    
    def load_learning_data(self) -> Dict[str, Any]:
        """加载学习数据"""
        if self.learning_data_path.exists():
            try:
                with open(self.learning_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'successful_patterns': [],  # 成功的解析模式
            'failed_patterns': [],      # 失败的解析模式
            'field_mappings': {},       # 字段映射学习
            'header_patterns': {},      # 表头模式学习
            'optimization_stats': {
                'total_attempts': 0,
                'successful_optimizations': 0,
                'accuracy_improvement': 0.0
            }
        }
    
    def record_successful_parse(self, file_pattern: str, header_info: Dict, 
                               field_mappings: Dict, parse_quality: float):
        """记录成功解析"""
        pattern = {
            'file_pattern': file_pattern,
            'header_row': header_info.get('header_row'),
            'field_mappings': field_mappings,
            'parse_quality': parse_quality,
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_data['successful_patterns'].append(pattern)
        
        # 更新字段映射学习
        for column, field in field_mappings.items():
            if column not in self.learning_data['field_mappings']:
                self.learning_data['field_mappings'][column] = []
            
            self.learning_data['field_mappings'][column].append({
                'mapped_to': field,
                'success': True,
                'confidence': parse_quality
            })
        
        self.save_learning_data()
    
    def record_failed_parse(self, file_pattern: str, error_info: Dict):
        """记录失败解析"""
        pattern = {
            'file_pattern': file_pattern,
            'error_info': error_info,
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_data['failed_patterns'].append(pattern)
        self.save_learning_data()
    
    def get_learned_field_mapping(self, raw_column: str) -> str:
        """获取学习到的字段映射"""
        mappings = self.learning_data['field_mappings'].get(raw_column, [])
        
        if not mappings:
            return raw_column
        
        # 统计最常映射的字段
        field_counts = {}
        total_confidence = 0
        
        for mapping in mappings:
            field = mapping['mapped_to']
            confidence = mapping.get('confidence', 0.5)
            
            if field not in field_counts:
                field_counts[field] = {'count': 0, 'total_confidence': 0}
            
            field_counts[field]['count'] += 1
            field_counts[field]['total_confidence'] += confidence
            total_confidence += confidence
        
        if field_counts:
            # 选择成功率最高的映射
            best_field = max(field_counts.keys(), 
                           key=lambda x: field_counts[x]['total_confidence'])
            return best_field
        
        return raw_column
    
    def optimize_parsing_strategy(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """基于学习优化解析策略"""
        strategy = {}
        
        # 基于历史成功模式优化
        for pattern in self.learning_data['successful_patterns']:
            if self._pattern_matches_file(pattern, file_info):
                # 应用成功的策略
                if 'header_row' in pattern and pattern['header_row']:
                    strategy['optimized_header_row'] = pattern['header_row']
                
                if 'field_mappings' in pattern:
                    strategy['learned_mappings'] = pattern['field_mappings']
        
        # 更新学习统计
        self.learning_data['optimization_stats']['total_attempts'] += 1
        
        return strategy
    
    def _pattern_matches_file(self, pattern: Dict, file_info: Dict) -> bool:
        """判断模式是否匹配文件"""
        # 简单的模式匹配逻辑
        file_pattern = pattern.get('file_pattern', '')
        
        if not file_pattern:
            return False
        
        # 这里可以实现更复杂的模式匹配逻辑
        return file_pattern in str(file_info.get('file_path', ''))
    
    def save_learning_data(self):
        """保存学习数据"""
        try:
            with open(self.learning_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存学习数据失败: {e}")
```

## 8. 性能优化

### 8.1 大文件处理优化

```python
# -*- coding: utf-8 -*-
"""
性能优化模块
专门处理大文件和复杂Excel结构
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
    
    def parallel_parse_excel_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """并行解析Excel文件"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._parse_single_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'file_path': file_path,
                        'error': str(e)
                    })
        
        return results
    
    def _parse_single_file(self, file_path: str) -> Dict[str, Any]:
        """解析单个文件（线程安全）"""
        # 这里调用具体的解析逻辑
        parser = ExcelIntelligentParser()
        return parser.parse_excel_file(file_path)
    
    def chunked_data_processing(self, large_dataframe: pd.DataFrame, 
                               chunk_size: int = 10000) -> pd.DataFrame:
        """分块处理大数据"""
        chunks = []
        
        for start_idx in range(0, len(large_dataframe), chunk_size):
            end_idx = min(start_idx + chunk_size, len(large_dataframe))
            chunk = large_dataframe.iloc[start_idx:end_idx].copy()
            
            # 处理每个数据块
            processed_chunk = self._process_data_chunk(chunk)
            chunks.append(processed_chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _process_data_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """处理数据块"""
        # 这里实现具体的处理逻辑
        return chunk
    
    def memory_optimized_read(self, file_path: str, **kwargs) -> pd.DataFrame:
        """内存优化的文件读取"""
        # 使用pandas的迭代器读取大文件
        chunk_iter = pd.read_excel(file_path, chunksize=1000, **kwargs)
        
        chunks = []
        for chunk in chunk_iter:
            # 对每个块进行处理
            processed_chunk = self._process_data_chunk(chunk)
            chunks.append(processed_chunk)
            
            # 强制垃圾回收
            import gc
            gc.collect()
        
        return pd.concat(chunks, ignore_index=True)
```

## 9. 质量保证与测试

### 9.1 单元测试框架

```python
# -*- coding: utf-8 -*-
"""
EXCEL解析器测试套件
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path

class TestExcelIntelligentParser(unittest.TestCase):
    """Excel智能解析器测试"""
    
    def setUp(self):
        """测试准备"""
        self.parser = ExcelIntelligentParser()
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
    
    def create_test_excel(self, data: Dict, filename: str) -> str:
        """创建测试用的Excel文件"""
        file_path = self.test_data_dir / filename
        
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        
        return str(file_path)
    
    def test_basic_excel_parsing(self):
        """测试基本Excel解析"""
        data = {
            '股票代码': ['000001', '000002', '000003'],
            '股票名称': ['平安银行', '万科A', '中国石油'],
            '信号日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '次日开盘买入持股两日收益率': [0.02, 0.03, -0.01]
        }
        
        file_path = self.create_test_excel(data, 'basic_test.xlsx')
        result = self.parser.parse_excel_file(file_path)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['data']), 3)
        self.assertIn('股票代码', result['data'].columns)
    
    def test_complex_header_parsing(self):
        """测试复杂表头解析"""
        data = {
            'Unnamed: 0': ['基准数据'],
            'Unnamed: 1': ['股票代码', '股票名称', '信号日期', '收益率'],
            'Unnamed: 2': ['code', 'name', 'date', 'return'],
            '0': ['说明', '', '', ''],
            '1': ['000001', '平安银行', '2023-01-01', 0.02],
            '2': ['000002', '万科A', '2023-01-02', 0.03]
        }
        
        file_path = self.create_test_excel(data, 'complex_header.xlsx')
        result = self.parser.parse_excel_file(file_path)
        
        # 验证能够处理复杂表头
        self.assertTrue(result['success'])
        # 检查是否正确识别了表头
        self.assertIn('parse_report', result)
    
    def test_percentage_conversion(self):
        """测试百分比转换"""
        data = {
            '股票代码': ['000001', '000002'],
            '收益率': ['2.5%', '3.2%']
        }
        
        file_path = self.create_test_excel(data, 'percentage_test.xlsx')
        result = self.parser.parse_excel_file(file_path)
        
        self.assertTrue(result['success'])
        # 检查百分比是否正确转换为小数
        self.assertAlmostEqual(result['data']['收益率'].iloc[0], 0.025)
    
    def tearDown(self):
        """测试清理"""
        # 清理测试文件
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*.xlsx"):
                file.unlink()
            self.test_data_dir.rmdir()

class TestFieldMapping(unittest.TestCase):
    """字段映射测试"""
    
    def test_standard_field_detection(self):
        """测试标准字段检测"""
        from fa_intelligent_parsing_helpers import detect_fields_intelligently
        
        columns = ['code', 'stock_name', 'date', 'return_rate']
        mappings = detect_fields_intelligently(columns)
        
        expected = {
            'code': '股票代码',
            'stock_name': '股票名称',
            'date': '信号日期',
            'return_rate': '次日开盘买入持股两日收益率'
        }
        
        for col, expected_field in expected.items():
            self.assertEqual(mappings.get(col), expected_field)

if __name__ == '__main__':
    unittest.main()
```

## 10. 部署和配置

### 10.1 安装依赖

```bash
# 安装必要的依赖
pip install pandas openpyxl xlrd pyyaml

# 可选：安装性能优化相关依赖
pip install numba  # 用于数值计算加速
pip install dask   # 用于大数据处理
```

### 10.2 配置文件

创建 `excel_parsing_rules.yaml` 配置文件：

```yaml
# Excel智能解析规则配置

# 基础配置
base_config:
  # 默认工作表选择策略
  default_sheet_selection: "largest_sheet"  # first_sheet | largest_sheet | by_name
  
  # 默认表头检测策略
  default_header_strategy: "smart_detection"  # smart_detection | multiple_row | skip_rows
  
  # 数据区域定位策略
  data_region_strategy: "density_analysis"  # density_analysis | boundary_detection
  
  # 空值处理策略
  na_values: ["", " ", "NA", "N/A", "NULL", "null", "-", "—", "--"]
  keep_default_na: true
  
  # 性能配置
  max_workers: 4  # 并行处理工作线程数
  chunk_size: 10000  # 大数据块处理大小
  memory_limit_mb: 1024  # 内存限制(MB)

# 字段映射配置
field_mappings:
  # 标准字段映射
  standard_fields:
    "股票代码": 
      - "code"
      - "stock_code"
      - "证券代码"
      - "代码"
      - "stockcode"
      - "股票代码"
    
    "股票名称": 
      - "name"
      - "stock_name"
      - "证券名称"
      - "名称"
      - "stockname"
      - "股票名称"
    
    "信号日期": 
      - "date"
      - "signal_date"
      - "日期"
      - "交易日期"
      - "trading_date"
      - "signaldate"
    
    "次日开盘买入持股两日收益率": 
      - "return"
      - "收益率"
      - "收益"
      - "return_rate"
      - "yield"
      - "returnrate"

  # 因子字段映射
  factor_fields:
    "当日回调":
      - "当日回调"
      - "当日回撤"
      - "daily_drawdown"
      - "回撤"
      - "drawdown"
      - "当日最大回撤"
    
    "机构持股比例(%)":
      - "机构持股比例"
      - "持股比例"
      - "institution_holding"
      - "机构持股"
      - "持股比例(%)"
      - "inst_holding"
    
    "流通市值(元)":
      - "流通市值"
      - "市值"
      - "circulating_market_value"
      - "market_cap"
      - "流通市值(元)"
      - "marketvalue"

  # 数值字段配置
  numeric_fields:
    "当日回调":
      type: "float"
      required: true
      validation_rules:
        - min_value: -1
        - max_value: 1
        - treat_percentage: false
    
    "机构持股比例(%)":
      type: "percentage"
      required: false
      validation_rules:
        - min_value: 0
        - max_value: 1
    
    "流通市值(元)":
      type: "currency"
      required: false
      validation_rules:
        - min_value: 0
        - unit_conversion: "yuan_to_million"

# 工作表特定配置
sheet_configs:
  "Sheet1":
    header_row: 0
    data_start_row: 1
    header_strategy: "smart_detection"
    column_mapping:
      0: "股票代码"
      1: "股票名称"
      2: "信号日期"
      3: "次日开盘买入持股两日收益率"
  
  "详细数据":
    header_row: 1
    data_start_row: 3
    skip_rows: [0, 2]  # 跳过的行
    header_strategy: "multiple_row"
    special_handling:
      merged_cells: true
      multi_level_header: true
  
  "数据分析表":
    header_row: 0
    data_start_row: 2
    skip_rows: [1]  # 跳过的行
    header_strategy: "skip_rows"
    auto_detect_columns: true

# 错误处理配置
error_handling:
  # 容错级别
  tolerance_level: "medium"  # low | medium | high
  
  # 缺失数据处理
  missing_data_strategy: "use_available"  # use_available | skip_column | skip_row
  
  # 数据类型转换失败处理
  conversion_error_strategy: "keep_original"  # keep_original | skip_row | use_default
  
  # 自动修复策略
  auto_fixes:
    - fix_column_names: true
    - fix_data_types: true
    - fill_missing_dates: false
    - handle_merged_cells: true
  
  # 解析失败回退策略
  fallback_strategies:
    - strategy: "pandas_default"
      description: "使用pandas默认解析"
    - strategy: "openpyxl_manual"
      description: "使用openpyxl手动解析"
    - strategy: "skip_file"
      description: "跳过该文件继续处理"

# 性能优化配置
performance:
  # 并行处理
  enable_parallel: true
  max_workers: 4
  
  # 大文件处理
  chunk_processing:
    enabled: true
    chunk_size: 10000
    
  memory_management:
    garbage_collect_interval: 100  # 每100个文件强制垃圾回收
    memory_limit_mb: 2048
    
  cache_settings:
    enable_cache: true
    cache_ttl_hours: 24
    cache_dir: ".excel_parse_cache"

# 日志配置
logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR
  enable_file_logging: true
  log_file: "excel_parsing.log"
  max_log_size_mb: 10
  
  # 解析统计日志
  log_statistics: true
  statistics_file: "parsing_statistics.json"

# 质量保证配置
quality_assurance:
  # 数据质量检查
  enable_quality_checks: true
  
  # 必需的字段
  required_fields:
    - "股票代码"
    - "信号日期"
    - "次日开盘买入持股两日收益率"
  
  # 数据质量阈值
  quality_thresholds:
    min_completeness: 0.8  # 最小完整性要求
    max_duplication_ratio: 0.1  # 最大重复率
    min_data_rows: 10  # 最小数据行数
  
  # 数据验证规则
  validation_rules:
    - field: "股票代码"
      pattern: "^[0-9]{6}$"
      description: "股票代码应为6位数字"
    - field: "信号日期"
      format: "date"
      description: "日期格式应正确"
    - field: "次日开盘买入持股两日收益率"
      type: "numeric"
      min_value: -1
      max_value: 1
      description: "收益率应在合理范围内"

# 扩展配置
extensions:
  # 插件目录
  plugin_dir: "plugins"
  
  # 自定义处理器
  custom_processors:
    - name: "financial_data_processor"
      enabled: true
      config:
        handle_currency: true
        handle_percentage: true
        date_formats: ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"]
  
  # 机器学习增强
  ml_enhancements:
    enabled: false
    model_dir: "ml_models"
    auto_learning: false
```

### 10.3 使用示例

```python
# -*- coding: utf-8 -*-
"""
使用示例：如何在现有项目中集成智能EXCEL解析
"""

from fa_intelligent_excel_parser import ExcelIntelligentParser, PerformanceOptimizer

def enhanced_yinzifenxi_main():
    """增强版的主程序"""
    
    # 初始化智能解析器
    parser = ExcelIntelligentParser("excel_parsing_rules.yaml")
    optimizer = PerformanceOptimizer(max_workers=4)
    
    # 配置数据文件路径
    configured_files = list(DEFAULT_DATA_FILES) if DEFAULT_DATA_FILES else []
    if not configured_files and DEFAULT_DATA_FILE:
        configured_files = [DEFAULT_DATA_FILE]
    
    print("[INFO] 使用智能EXCEL解析器加载数据文件:")
    for path in configured_files:
        print(f"  - {path}")
    
    # 检查文件是否存在
    available_files = []
    for path in configured_files:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"[INFO] 数据文件路径: {abs_path}")
            available_files.append(path)
        else:
            print(f"[WARN] 数据文件不存在，已跳过: {abs_path}")
    
    if not available_files:
        print("[ERROR] 没有可用的数据文件，请检查配置后重试")
        return None
    
    # 使用智能解析器并行加载所有文件
    print(f"\n[INFO] 开始智能解析 {len(available_files)} 个文件...")
    
    # 并行解析
    parse_results = optimizer.parallel_parse_excel_files(available_files)
    
    # 收集成功解析的数据
    all_data_frames = []
    successful_parses = 0
    
    for result in parse_results:
        if result['success']:
            data = result['data']
            data['source_file'] = result['file_info']['file_path']
            data['source_sheet'] = result.get('selected_sheet', 'unknown')
            
            all_data_frames.append(data)
            successful_parses += 1
            
            # 显示解析质量信息
            if 'parse_report' in result:
                report = result['parse_report']
                print(f"  ✓ {os.path.basename(result['file_info']['file_path'])}: "
                      f"{len(data)}行, 质量{report['data_quality']['completeness']:.1%}")
        else:
            print(f"  ✗ {result.get('file_path', 'Unknown')}: {result['error']}")
    
    # 合并所有数据
    if all_data_frames:
        combined_data = pd.concat(all_data_frames, ignore_index=True, sort=False)
        print(f"\n[INFO] 数据合并完成:")
        print(f"  - 总行数: {len(combined_data)}")
        print(f"  - 总列数: {len(combined_data.columns)}")
        print(f"  - 成功解析: {successful_parses}/{len(available_files)} 个文件")
        
        # 显示数据质量报告
        completeness = 1 - combined_data.isnull().sum().sum() / combined_data.size
        print(f"  - 数据完整性: {completeness:.1%}")
        print(f"  - 字段信息: {list(combined_data.columns)}")
        
        return combined_data
    else:
        print("[ERROR] 所有文件解析失败")
        return None

# 使用方法
if __name__ == "__main__":
    data = enhanced_yinzifenxi_main()
    if data is not None:
        print("数据加载成功，可以继续进行因子分析...")
    else:
        print("数据加载失败，请检查文件格式和配置...")
```

## 11. 总结

这个EXCEL解析优化方案具有以下优势：

### 11.1 技术优势
- **智能化程度高**：自动检测表头、数据区域，无需人工配置
- **扩展性强**：插件化架构支持自定义解析逻辑
- **容错性好**：多层回退机制，确保解析成功率
- **性能优异**：并行处理、分块加载，优化大数据文件处理

### 11.2 业务优势
- **降低维护成本**：配置驱动的解析规则，无需修改代码
- **提高适配性**：支持各种复杂的Excel格式和布局
- **保证数据质量**：智能数据类型转换和数据验证
- **便于监控**：详细的解析统计和质量报告

### 11.3 实施建议
1. **分阶段实施**：先实现基础功能，再逐步增加高级特性
2. **充分测试**：使用各种真实Excel文件进行测试验证
3. **配置优化**：根据实际使用的Excel格式调整配置规则
4. **监控完善**：建立解析成功率和质量监控机制

通过这个优化方案，您的因子分析系统将能够智能地处理各种复杂的Excel文件，大大提高数据处理的效率和可靠性。
