# -*- coding: utf-8 -*-
import os
from datetime import datetime

import pandas as pd


def main():
    clean_table_file = "创业板单日下跌14%详细交易日数据（原始数据）1114.xlsx"
    output_file = None
    cleaner = ExcelDataCleaner(
        source_file=clean_table_file,
        output_file=output_file,
        limit_per_date=10,
    )
    cleaner.run()


class ExcelDataCleaner:
    """Excel 数据清洗工具，与原 qingxi.py 中的清洗逻辑保持一致。"""

    def __init__(self, source_file, output_file=None, limit_per_date=10):
        self.source_file = source_file
        self.limit_per_date = limit_per_date
        self._output_file_set_manually = output_file is not None
        self.output_file = output_file if output_file else self._generate_output_filename()
        self.table1_data = None

    def _generate_output_filename(self):
        base_name, ext = os.path.splitext(self.source_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_cleaned_{timestamp}{ext}"

    def load_data(self):
        try:
            print(f"正在加载专门的清洗表格数据：{self.source_file}")
            self.table1_data = pd.read_excel(self.source_file)
            print(f"清洗表格数据加载完成，共{len(self.table1_data)}行")
            stock_code_columns = ['股票代码', 'stock_code', 'code']
            self.stock_code_col1 = None

            for col in stock_code_columns:
                if col in self.table1_data.columns:
                    self.stock_code_col1 = col
                    break

            if not self.stock_code_col1:
                raise ValueError("在清洗表格中未找到股票代码列（股票代码、stock_code或code）")

            print(f"清洗表格股票代码列：{self.stock_code_col1}")
            return True
        except Exception as e:
            print(f"加载数据失败：{str(e)}")
            return False

    def clean_data(self):
        try:
            original_rows = len(self.table1_data)
            print(f"开始清洗表格数据，当前共有{original_rows}行")

            print("未执行查询合并，将检查整个表格中的'无数据'单元格")
            mask = ~self.table1_data.isin(["无数据"]).any(axis=1)
            self.table1_data = self.table1_data[mask]
            after_clean_rows = len(self.table1_data)
            print(f"步骤1：删除了{(original_rows - after_clean_rows)}行含有'无数据'的记录（全表检查）")
            original_rows = after_clean_rows

            signal_date_col = "信号日期"
            if signal_date_col in self.table1_data.columns:
                grouped = self.table1_data.groupby(signal_date_col)
                cleaned_rows = []
                rows_deleted = 0

                for _, group in grouped:
                    if len(group) > self.limit_per_date:
                        cleaned_group = group.sample(n=self.limit_per_date, random_state=42)
                        rows_deleted += (len(group) - self.limit_per_date)
                        cleaned_rows.append(cleaned_group)
                    else:
                        cleaned_rows.append(group)

                if rows_deleted > 0:
                    self.table1_data = pd.concat(cleaned_rows).reset_index(drop=True)
                    print(f"步骤2：每个日期最多保留{self.limit_per_date}行，共删除了{rows_deleted}行多余记录")

            final_rows = len(self.table1_data)
            print(f"表格清洗完成，最终剩余{final_rows}行数据")
            return True
        except Exception as e:
            print(f"表格清洗过程中出错：{str(e)}")
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
        print("=== Excel数据清洗器开始运行 ===")
        if not self.load_data():
            return False
        if not self.clean_data():
            return False
        if not self.save_result():
            return False
        print("\n=== Excel数据清洗器运行完成 ===")
        return True


if __name__ == "__main__":
    main()
