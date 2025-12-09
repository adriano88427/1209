# -*- coding: utf-8 -*-
import os
from datetime import datetime

import pandas as pd


def main():
    table1_file = "创业板单日下跌14%详细交易日数据（清理后）1114_merged_20251128_144645_merged_20251128_144737_merged_20251128_144815_merged_20251128_144852.xlsx"
    table2_file = "股本分析结合收益2021.xlsx"
    output_file = None

    join_columns = [
        "流通市值(元)",
        "机构持股比例(%)",
        "十大流通机构大非",
        "企业大股东大非（包含国资）",
        "前10大流通股东持股比例合计",
    ]

    merger = ExcelDataMerger(
        table1_file=table1_file,
        table2_file=table2_file,
        join_columns=join_columns,
        output_file=output_file,
        year=2021,
    )
    merger.run()


class ExcelDataMerger:
    """Excel 数据合并工具：将表2的列按股票代码并入表1。"""

    def __init__(self, table1_file, table2_file, join_columns, output_file=None, year=None):
        self.table1_file = table1_file
        self.table2_file = table2_file
        self.join_columns = {col: col for col in join_columns}
        self._output_file_set_manually = output_file is not None
        self.output_file = output_file if output_file else self._generate_output_filename()
        self.year = year
        self.table1_data = None
        self.table2_data = None

    def _generate_output_filename(self):
        base_name, ext = os.path.splitext(self.table1_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_merged_{timestamp}{ext}"

    def load_data(self):
        """加载表1、表2数据并识别股票代码列。"""
        try:
            print(f"正在加载表1数据：{self.table1_file}")
            self.table1_data = pd.read_excel(self.table1_file)
            print(f"表1数据加载完成，共{len(self.table1_data)}行")

            print(f"正在加载表2数据：{self.table2_file}")
            self.table2_data = pd.read_excel(self.table2_file)
            print(f"表2数据加载完成，共{len(self.table2_data)}行")

            stock_code_columns = ['股票代码', 'stock_code', 'code']
            self.stock_code_col1 = None
            self.stock_code_col2 = None

            for col in stock_code_columns:
                if col in self.table1_data.columns:
                    self.stock_code_col1 = col
                    break

            if not self.stock_code_col1:
                raise ValueError("在表1中未找到股票代码列（股票代码、stock_code或code）")

            print(f"表1股票代码列：{self.stock_code_col1}")

            for col in stock_code_columns:
                if col in self.table2_data.columns:
                    self.stock_code_col2 = col
                    break

            if not self.stock_code_col2:
                raise ValueError("在表2中未找到股票代码列（股票代码、stock_code或code）")

            print(f"表2股票代码列：{self.stock_code_col2}")
            return True
        except Exception as e:
            print(f"加载数据失败：{str(e)}")
            return False

    def validate_join_columns(self):
        """验证要查询的列是否存在于表2中。"""
        missing_columns = []
        existing_columns = []

        for table2_col in self.join_columns.keys():
            if table2_col in self.table2_data.columns:
                existing_columns.append(table2_col)
            else:
                missing_columns.append(table2_col)

        if missing_columns:
            print(f"警告：表2中缺少以下列：{', '.join(missing_columns)}")

        if existing_columns:
            print(f"表2中存在的列：{', '.join(existing_columns)}")
            self.validated_columns = {col: self.join_columns[col] for col in existing_columns}
            return True
        else:
            print("错误：表2中没有找到任何指定的查询列")
            return False

    def query_and_merge(self):
        """执行查询与合并。"""
        if self.table1_data is None or self.table2_data is None:
            print("错误：请先加载数据")
            return False

        if not self.validate_join_columns():
            print("请修正查询列后重试")
            return False

        try:
            stock_mapping = {}
            for _, row in self.table2_data.iterrows():
                stock_code = str(row[self.stock_code_col2]).strip()
                stock_mapping.setdefault(stock_code, {})
                for table2_col, table1_col in self.validated_columns.items():
                    stock_mapping[stock_code][table1_col] = row.get(table2_col, None)

            filter_by_year = self.year is not None
            signal_date_col = "信号日期"
            has_signal_date = signal_date_col in self.table1_data.columns

            if filter_by_year:
                if not has_signal_date:
                    print(f"警告：表1中未找到'信号日期'列，将处理所有行")
                    filter_by_year = False
                else:
                    print(f"启用年份筛选，只处理信号日期在{self.year}年的行")
                    try:
                        self.table1_data[signal_date_col] = pd.to_datetime(self.table1_data[signal_date_col])
                    except Exception as e:
                        print(f"警告：无法将'信号日期'列转换为日期类型：{str(e)}")
                        print("将尝试从字符串中提取年份")

            matched_count = 0
            filtered_count = 0

            for idx, row in self.table1_data.iterrows():
                process_row = True

                if filter_by_year and has_signal_date:
                    signal_date = row[signal_date_col]
                    try:
                        if pd.api.types.is_datetime64_any_dtype(signal_date):
                            row_year = signal_date.year
                        else:
                            import re
                            date_str = str(signal_date)
                            year_match = re.search(r'\b(\d{4})\b', date_str)
                            if year_match:
                                row_year = int(year_match.group(1))
                            else:
                                process_row = False
                                continue

                        if row_year != self.year:
                            process_row = False
                            continue
                    except Exception as e:
                        print(f"处理行{idx}的信号日期时出错：{str(e)}")
                        process_row = False
                        continue

                    filtered_count += 1

                if process_row:
                    stock_code = str(row[self.stock_code_col1]).strip()
                    for table1_col in [col for col in self.validated_columns.values()]:
                        self.table1_data.at[idx, table1_col] = "无数据"

                    if stock_code in stock_mapping:
                        matched_count += 1
                        for table1_col, value in stock_mapping[stock_code].items():
                            if value is not None and str(value).strip() != '':
                                self.table1_data.at[idx, table1_col] = value

            print(f"查询完成，共匹配到{matched_count}条记录")
            if filter_by_year and has_signal_date:
                print(f"按年份{self.year}筛选后，处理了{filtered_count}行数据")
            return True

        except Exception as e:
            print(f"查询和合并过程中出错：{str(e)}")
            return False

    def save_result(self):
        try:
            print(f"正在保存结果到：{self.output_file}")
            self.table1_data.to_excel(self.output_file, index=False)
            print("结果保存成功")
            return True
        except Exception as e:
            print(f"保存结果失败：{str(e)}")
            return False

    def run(self):
        print("=== Excel数据合并器开始运行 ===")
        if not self.load_data():
            return False
        if not self.query_and_merge():
            return False
        if not self.save_result():
            return False
        print("\n=== Excel数据合并器运行完成 ===")
        return True


if __name__ == "__main__":
    main()
