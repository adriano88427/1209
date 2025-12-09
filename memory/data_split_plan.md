# qingxi.py 模块拆分方案

> 目标：在**不改变现有逻辑/行为**的前提下，将 `shuju/qingxi.py` 拆分为“数据合并”与“清洗”两个独立模块，便于单独调用或组合运行。

## 1. 拆分思路
1. 不再保留入口脚本 `qingxi.py`，仅保留两个独立模块。
2. 新建模块/文件：
   - `data_merge.py`：包含 `ExcelDataQuerier` 的加载/合并相关方法（`load_data`、`validate_join_columns`、`query_and_merge`、`save_result` 的合并部分等），提供 `if __name__ == "__main__":` 入口以单独运行合并流程。
   - `data_clean.py`：包含与清洗相关的 `clean_data` 逻辑及独立运行入口（可以直接对指定表格执行清洗），并支持 `perform_merge="否" + clean_table_file` 的使用场景。
3. `ExcelDataQuerier` 类拆分为两个版本或两个职责清晰的函数：  
   - `MergeProcessor`（或保留原名，但驻留 `data_merge.py`）负责合并逻辑；  
   - `CleanProcessor` 负责清洗逻辑。若需要共享状态，可通过数据类/配置对象在两个脚本间传递。


## 2. 代码层面保持一致性的措施
- **类/函数签名不变**：`ExcelDataQuerier` 仍暴露与原来完全一致的 API（构造参数、`run()` 方法）。拆分后的两个模块只负责承载具体实现，不对调用方产生感知。
- **逻辑迁移**：`load_data`、`validate_join_columns`、`query_and_merge` 等按模块职责迁出；`clean_data` 及清洗相关辅助函数挪到清洗模块；`save_result` 中关于清洗的分支调用新的清洗函数。
- **共享状态**：通过将 `ExcelDataQuerier` 的实例自身传入两个模块（或传递需要的数据结构），保证各方法访问 `self.table1_data`、`self.table2_data` 等属性时无差异。
- **常量/默认参数**：保持原字符串（如“无数据”）和参数默认值不变。
- **日志与打印**：逐字保留原有 `print` 输出顺序与内容。

## 3. 文件结构调整
```
shuju/
 ├─ data_merge.py      # 独立可运行的合并脚本（含 ExcelDataQuerier / MergeProcessor）
 └─ data_clean.py      # 独立可运行的清洗脚本（含 CleanProcessor）
```

## 4. 迁移步骤
1. **复制现有类**到 `data_merge.py`，保留所有属性初始化与合并方法；清洗方法先保留。
2. 将 `clean_data` 及相关辅助处理移动到 `data_clean.py`，并暴露 `clean_data(self)` 函数，供 `ExcelDataQuerier` 调用。
3. 在 `data_merge.py` 中通过 `from .data_clean import clean_data`（或类方法注入）方式调用清洗逻辑。
4. 更新 `qingxi.py`：仅保留 `main()`、参数配置与 `ExcelDataQuerier` 导入，并确保 `if __name__ == "__main__": main()` 不变。
5. 删除旧 `qingxi.py`，确认无任何 import 依赖它；如需批处理，可通过 shell 脚本依次调用 `python data_merge.py ...` 与 `python data_clean.py ...`。

## 5. 验证策略
- 使用相同参数分别运行 `python data_merge.py`（合并场景）与 `python data_clean.py`（清洗场景），与拆分前 `qingxi.py` 的输出进行对比。
- 测试以下场景：
  1. `perform_merge="是"`、`clean_table="否"`（默认路径）。
  2. `perform_merge="是"`、`clean_table="是"`。
  3. `perform_merge="否"`、`clean_table="是"`（专用清洗表路径）。
- 通过 `git diff` 确认所有字符串、参数值、返回值保持不变，仅结构发生转移。

此方案执行后，`shuju` 目录具备可独立调用的数据清洗与合并模块，方便未来扩展或在其他脚本中复用，同时原入口 `qingxi.py` 的行为对用户完全透明。 
