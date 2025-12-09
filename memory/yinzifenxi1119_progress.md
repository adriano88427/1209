# yinzifenxi1119 Split / Refactor Progress Log

> This file is for the AI and for technical tracking only, so it is kept in **English only** to avoid any encoding issues on your system.

All functional messages, reports, and user-facing text in the actual Python code remain in Chinese and are **never changed** unless explicitly required.

---

## 2025-11-26 – Baseline and Phase 1

### Phase 0 – Baseline Outputs (original script)
- Script: `yinzifenxi1119.py` (the **golden** original, never modified).
- Actions:
  - Run: `python yinzifenxi1119.py`
  - Copy all generated TXT/CSV/log files into `baseline_outputs/`
- Purpose:
  - `baseline_outputs/` is the reference set used to check that all later refactor steps keep behavior and numeric results identical.

---

### Phase 1 – Logger Extraction (fa_logging.Logger)

#### 1.1 Create split entry script
- Created `yinzifenxi1119_split.py` as an exact copy of `yinzifenxi1119.py`.
- Goal:
  - All refactor experiments happen in `yinzifenxi1119_split.py` and under `daima/`.
  - `yinzifenxi1119.py` remains untouched and is still used for “official” runs.

#### 1.2 Create dedicated logging module
- New file: `daima/fa_logging.py`
- Implemented class `Logger` with behavior identical to the original inline `Logger`:
  - Creates `因子分析日志_YYYYMMDD_HHMMSS.txt`
  - Writes a header line with timestamp and separators
  - `write()` prints to both terminal and log file
  - `close()` writes a “log end” footer and restores `sys.stdout`
- At this point the split script still used its own local `Logger`, the module was just prepared.

#### 1.3 Wire split script to external Logger
- In `yinzifenxi1119_split.py`:
  - Added:
    ```python
    from daima.fa_logging import Logger as ExternalLogger
    ```
  - Switched `main()` to use:
    ```python
    logger = ExternalLogger()
    ```
  - (The old inline `class Logger` remained in the file but stopped being used.)

#### 1.4 Phase 1 verification
- Runs and outputs:
  - Ran `python yinzifenxi1119_split.py` and copied outputs to:
    - `stage1_logger_outputs/` (before switching to ExternalLogger)
    - `stage2_logger_outputs/` (after switching to ExternalLogger)
- Comparison vs `baseline_outputs/`:
  - All CSV files: byte-identical (same SHA256)
  - TXT reports and log files: differ only due to timestamps and run-time metadata, which is expected.
- Conclusion:
  - Extracting `Logger` into `fa_logging.Logger` and switching the split script to use it did **not** change any numeric results or CSV contents.

---

## 2025-11-26 – Phase 2: Statistical Utilities (fa_stat_utils)

### 2.1 Create fa_stat_utils with type-safety helpers
- New file: `daima/fa_stat_utils.py`
- Implemented (copied from original script):
  - `ensure_list`
  - `safe_len`
  - `safe_ensure_list`
- In `yinzifenxi1119_split.py`:
  - Added:
    ```python
    from daima.fa_stat_utils import ensure_list, safe_len, safe_ensure_list
    ```
  - For now, the original definitions remain in the split script to avoid any behavior change. The module is prepared for future use.

### 2.2 Move robustness and FDR helpers into fa_stat_utils (implementation only)
- In `daima/fa_stat_utils.py`:
  - Added placeholder versions (matching the “external helper removed” stubs in the script):
    - `kendall_tau_corr`, `robust_correlation`, `mann_whitney_u_test`
    - `bootstrap_confidence_interval`, `detect_outliers`, `sensitivity_analysis`
    - `false_discovery_control`, `rolling_window_analysis`
    - `temporal_stability_analysis`, `sample_sensitivity_analysis`
  - Added full implementation versions (with `_impl` suffix), copied from the rich implementations in the split script:
    - `kendall_tau_corr_impl`
    - `robust_correlation_impl`
    - `mann_whitney_u_test_impl`
    - `bootstrap_confidence_interval_impl`
    - `detect_outliers_impl`
    - `sensitivity_analysis_impl`
    - `false_discovery_control_impl`
- At this point:
  - The split script still uses its own internal implementations.
  - The `_impl` versions in `fa_stat_utils` are ready for gradual, controlled replacement.

### 2.3 Switch custom_spearman_corr to module implementation
- Chosen function: `custom_spearman_corr` (used in IC and robustness calculations).
- Steps:
  1. In `daima/fa_stat_utils.py`:
     - Added full implementation:
       ```python
       def custom_spearman_corr(x, y):
           ...
       ```
       copied from the original script.
  2. In `yinzifenxi1119_split.py`:
     - Added import:
       ```python
       from daima.fa_stat_utils import false_discovery_control_impl, custom_spearman_corr
       ```
     - Removed the local `custom_spearman_corr` function definition.
     - All calls to `custom_spearman_corr(...)` in the split script now use `fa_stat_utils.custom_spearman_corr`.

#### 2.3.1 Verification for custom_spearman_corr
- Run:
  - `python yinzifenxi1119_split.py`
  - Copy outputs to `stage_custom_spearman_outputs/`
- Compare vs `baseline_outputs/`:
  - CSV files (all factor/decile details and summary CSVs):
    - All pairs are byte-identical (same SHA256 hash).
  - TXT reports and logs:
    - Hashes differ due to timestamps / run-time metadata only (expected).
- Conclusion:
  - Switching `custom_spearman_corr` to the implementation in `fa_stat_utils` preserves all numeric behavior and CSV outputs exactly.

---

## 2025-11-26 – Phase 3: Prepare Non‑parametric Analysis Modules

### 3.1 Create fa_nonparam_analysis.py (passive copy of FactorAnalysis)
- New file: `daima/fa_nonparam_analysis.py`
- Contents:
  - English header explaining this is a non‑parametric factor analysis module,
    currently a passive copy.
  - Imports:
    ```python
    import numpy as np
    import pandas as pd
    from datetime import datetime
    ```
  - A verbatim copy of `class FactorAnalysis` taken from `yinzifenxi1119_split.py`.
- Important:
  - This module is **not yet wired** into the main workflow.
  - The active `FactorAnalysis` used by the program is still the one defined
    inside `yinzifenxi1119_split.py`.
  - Therefore, creating this file does **not** change any behavior or outputs.

### 3.2 Create fa_nonparam_report.py (placeholder for future report code)
- New file: `daima/fa_nonparam_report.py`
- Contents:
  - English header describing it as a future home for non‑parametric factor
    analysis report generation.
  - A note that it is not yet used by the main workflow.
- Purpose:
  - Provide a clear place to move `FactorAnalysis` report methods into
    (e.g. `generate_summary_report`, `generate_factor_analysis_report`, etc.)
    in later steps.

### 3.3 Add thin report wrappers (not yet used)
- Updated `daima/fa_nonparam_report.py` to add thin wrapper functions:
  - `_fa_generate_summary_report(self)`
  - `_fa_generate_factor_analysis_report(self, ...)`
  - `_fa_generate_positive_factors_analysis(self)`
  - `_fa_generate_negative_factors_analysis(self)`
- Current behavior:
  - Each wrapper simply delegates to the existing methods on `self`, e.g.:
    - `_fa_generate_summary_report(self)` calls `self.generate_summary_report()`.
  - These wrappers are not yet imported or used by `yinzifenxi1119_split.py`.
  - No logic has been changed; this is only preparing a clean separation layer
    for future refactors.

### 3.3 Status after Phase 3 preparation
- All changes in Phase 3 so far:
  - Only **add new modules** under `daima/`.
  - No changes to `yinzifenxi1119.py` or the active logic in
    `yinzifenxi1119_split.py`.
- Behavior check:
  - Running `python yinzifenxi1119.py` or `python yinzifenxi1119_split.py`
    still uses the original in‑script `FactorAnalysis` implementation.
  - No need for a new CSV/TXT comparison at this step, because the new modules
    are not yet imported or used.

---

## Notes / Guidelines Going Forward

- `yinzifenxi1119.py` remains the **golden reference**:
  - Only used for official runs and baseline generation.
  - Never modified during refactor/split work.

- All refactor work happens in:
  - `yinzifenxi1119_split.py`
  - `daima/fa_*.py` modules

- When switching any function from inline implementation to a module implementation:
  1. Copy the original implementation into a module (e.g. `fa_stat_utils`) verbatim.
  2. Update the split script to import and call the module function.
  3. Run `yinzifenxi1119_split.py`, capture outputs to `stageX_*_outputs/`.
  4. Compare all CSVs (and, if needed, logs and TXT reports) against `baseline_outputs/`.
  5. Only proceed if CSVs are byte-identical and differences in TXT/log files are explained by timestamps.


## 2025-11-26 – Phase 3.4: Wire non-parametric report wrappers (no behavior change)

- Updated `yinzifenxi1119_split.py` to call `_fa_generate_summary_report` and `_fa_generate_factor_analysis_report`
  instead of invoking the class methods directly so we could later move the implementations safely.
- Verification:
  - Command: `python yinzifenxi1119_split.py`
  - Outputs archived in `stage_nonparam_report_wrappers/`
  - Every CSV in the stage folder matches its counterpart under `baseline_outputs/` (SHA256 identical). TXT/log files
    only differ by timestamps.


## 2025-11-26 – Phase 3.5: Copy FactorAnalysis report implementations into fa_nonparam_report

- Cloned the bodies of the four report methods into the `_fa_*` helpers inside `daima/fa_nonparam_report.py` (including
  numpy/pandas/datetime imports) while keeping the original methods in the class for the moment.
- Verification:
  - Command: `python yinzifenxi1119_split.py`
  - Outputs archived in `stage_nonparam_report_impl/`
  - All CSV hashes equal the baseline hashes; TXT/log drift stems only from timestamps.


## 2025-11-26 – Phase 3.6: Delegate FactorAnalysis report methods to fa_nonparam_report

- Replaced the four report methods in `FactorAnalysis` with thin wrappers that call the helper functions, eliminating
  the duplicated logic in the class.
- Verification:
  - Command: `python yinzifenxi1119_split.py`
  - Outputs archived in `stage_nonparam_report_wiring/`
  - All CSVs remain byte-identical to baseline; TXT/log differences are timestamp-only.


## 2025-11-26 – Phase 3.7a: Prepare fa_nonparam_analysis for real usage

- Brought `daima/fa_nonparam_analysis.py` up to parity with the split script by importing `DEFAULT_DATA_FILE`,
  statistical helpers, and the plotting/scipy guards so the module became self-contained.
- Verification:
  - Command: `python yinzifenxi1119_split.py`
  - Outputs stored in `stage_nonparam_analysis_imports/`
  - CSV hashes still match baseline; TXT/log differences remain timestamp-only.


## 2025-11-26 – Phase 3.7b: Extract classification helpers

- Added `daima/fa_nonparam_helpers.py` with `_fa_classify_factors_by_ic`,
  `_fa_generate_factor_classification_overview`, and `_fa_get_suggested_weight`, then rewired the class methods to call
  those helpers.
- Verification:
  - Command: `python yinzifenxi1119_split.py`
  - Outputs archived in `stage_nonparam_helpers/`
  - CSV hashes equal the baseline set; TXT/log drift is timestamp-only.


## 2025-11-26 – Phase 3.8: Helper delegation for scoring/overview

- Added `_fa_get_scoring_standards` to the helper module and routed the scoring/overview paths through the helper to
  finish trimming duplicated text from `FactorAnalysis`.
- Outputs for this verification live in `stage_nonparam_helpers_delegation/`, and every CSV hash matches the baseline.


## 2025-11-27 – Progress audit & fixes

- Findings from the audit:
  1. `daima/fa_nonparam_report.py` **is already imported and used** inside `yinzifenxi1119_split.py`, so the header comment saying “尚未被引用” was outdated.
  2. The module missed `from daima.fa_config import DEFAULT_DATA_FILE` and referenced an undefined `analyzer` variable, which would raise `NameError` once `generate_factor_analysis_report()` writes the TXT.
  3. `daima/fa_nonparam_analysis.py` still contained full `_fa_*` helper implementations even though the helpers already live under `fa_nonparam_helpers`, risking divergence when we eventually activate this module.
- Fixes completed:
  - Updated `fa_nonparam_report.py` header + imports, and replaced all lingering `analyzer` usages with `self` (the helpers now truly act as module functions).
  - Ran `python yinzifenxi1119_split.py`, moved the outputs to `stage_nonparam_report_fix/`, and compared every CSV vs `baseline_outputs/` using SHA256 hashes — all matched byte-for-byte.
  - Removed the duplicated `_fa_*` helper definitions from `fa_nonparam_analysis.py`; the file now relies solely on the shared helpers it already imports.

## 2025-11-27 – Phase 3.9: FactorAnalysis wiring to module

- Refreshed `daima/fa_nonparam_analysis.py` so it now contains the up-to-date `FactorAnalysis` implementation from `yinzifenxi1119_split.py`, including the `_fa_*` helper delegations and the annual-return helper functions (`calculate_standard_annual_return`, `safe_calculate_annual_return`, `validate_annual_return_calculation`).
- Added the missing `_fa_generate_*` imports (classification/report helpers) plus a direct `import scipy` so the module has all dependencies in one place.
- In `yinzifenxi1119_split.py`:
  - Added `from daima.fa_nonparam_analysis import FactorAnalysis`.
  - Removed the local `class FactorAnalysis` block (replaced with a short comment noting it now lives in the module). ParameterizedFactorAnalyzer remains local for now.
- Verification:
  1. First two trial runs exposed missing helper definitions inside the module; outputs from those attempts were archived under `stage_nonparam_analysis_wiring_failed/` for reference.
  2. After inserting the helpers, re-ran `python yinzifenxi1119_split.py`, archived the successful artifacts in `stage_nonparam_analysis_wiring/`, and compared every CSV against `baseline_outputs/` via SHA256 hashes — all entries report `Match`.
- Result: the split entry script now consumes the module-defined `FactorAnalysis` while keeping behavior identical to the baseline.


## 2025-11-27 – Phase 4.0 Prep: Parameterized analyzer status check

- `ParameterizedFactorAnalyzer` (class + reporting logic) still resides inside `yinzifenxi1119_split.py`. None of its methods have been modularized yet, and the class depends on globals such as `np`, `pd`, and the inline constants.
- Goal for Phase 4: mirror the non-parametric refactor by introducing `daima/fa_param_analysis.py` (class + helpers) and, if needed, `daima/fa_param_report.py`, then wire the split script to import the module version.
- Validation requirements remain unchanged:
  1. Copy the class into the new module verbatim, wire imports, and remove the inline definition from the split script.
  2. Run `python yinzifenxi1119_split.py`, archive outputs (tentatively `stage_param_analysis_wiring/`), and compare every CSV against `baseline_outputs/` using SHA256.
  3. Update this progress log with the results plus any follow-up tasks (e.g., consolidating shared helpers into `fa_stat_utils`).


## 2025-11-27 – Phase 4.1: Parameterized analyzer wiring (成功)

- 新模块：创建 `daima/fa_param_analysis.py` 并将 `class ParameterizedFactorAnalyzer` 整体搬入其中，保留所有评分/报告逻辑，新增模块说明并使用 `DEFAULT_DATA_FILE` 生成报告抬头。
- 主脚本：在 `yinzifenxi1119_split.py` 中新增 `from daima.fa_param_analysis import ParameterizedFactorAnalyzer`，删除原类定义并留下注释，主流程创建带参数分析器时直接调用模块版本。
- 运行与产物：
  - 首次尝试因超时被截断，产物已归档在 `stage_param_analysis_wiring_failed/` 供审计。
  - 正式验证命令：`python yinzifenxi1119_split.py`（输出存入 `stage_param_analysis_wiring/`）。
  - CSV 哈希比对（vs `baseline_outputs/`）全部 `Match`，包括 `因子分析汇总_*`, `带参数因子分析数据_*` 及 7 个 “带参数因子详细分析_*” 文件；TXT/日志只因时间戳差异而不同。
- 结论：带参数分析器成功模块化，拆分脚本保持与黄金脚本一致的行为，可继续推进后续公共函数迁移与报告解耦工作。


## 2025-11-27 – Phase 4.2: Parameterized 报告模块化 & 哈希脚本扩展

- 报告拆分：
  - 新建 `daima/fa_param_report.py`，把 `generate_parameterized_report` 的正文移入 `_fa_generate_parameterized_report()`，`ParameterizedFactorAnalyzer` 仅保留薄薄的委托方法。
  - `calculate_ic()` 现在改用 `fa_stat_utils.custom_spearman_corr` 计算相关系数，避免 pandas 版本差异并与非参数模块共享实现。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出目录：`stage_param_report_module/`
  - 对比：`python tools/compare_stage_outputs.py stage_param_report_module`。脚本默认比较所有 CSV，并在哈希前忽略 TXT 中的 `生成时间` 行，从而验证 `带参数因子综合分析报告_*` 内容未发生实质性变化。
- 新工具：`tools/compare_stage_outputs.py`
  - 作用：统一比较阶段目录与 `baseline_outputs/` 的 SHA256，新增对“带参数”系列 TXT 的检测。
  - 使用：`python tools/compare_stage_outputs.py <stage_dir> [--baseline other_dir]`


## 2025-11-27 – Phase 4.3: Parameterized scoring helper & split 脚本清理

- 把 `ParameterizedFactorAnalyzer.score_factors` 迁入 `daima/fa_param_helpers.py`，主类只剩薄封装，更易后续复用或测试。
- 清理 `yinzifenxi1119_split.py` 中残留的全局工具（早期拷贝的 `Logger`、`ensure_list`/`safe_len` 等），现在全部依赖 `daima` 模块提供的实现，入口脚本进一步瘦身。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 产物：`stage_param_helpers_refactor/`
  - 对比：`python tools/compare_stage_outputs.py stage_param_helpers_refactor`，所有 CSV 以及带参数 TXT 文件均与 `baseline_outputs/` 匹配（TXT 仍在忽略时间戳后哈希）。


## 2025-11-27 – Phase 4.4: Annualization utilities consolidation

- 将 `calculate_standard_annual_return` / `safe_calculate_annual_return` / `validate_annual_return_calculation` 剥离到 `daima/fa_stat_utils.py`，`FactorAnalysis` 和入口脚本不再各自维护副本，参数化模块也可以按需重用。
- `daima/fa_nonparam_analysis.py` 改为直接 `from daima.fa_stat_utils import ...`；`yinzifenxi1119_split.py` 仅保留 orchestrator 逻辑（重新整理 main，保持调用流程与之前一致）。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出保存至 `stage_annual_utils_refactor/`
  - 对比：`python tools/compare_stage_outputs.py stage_annual_utils_refactor`，所有 CSV 与带参数 TXT 再次和 `baseline_outputs/` 匹配（借助新脚本自动忽略 TXT 中的时间戳行）。


## 2025-11-27 – Phase 4.5: Split script helper purge

- 清空 `yinzifenxi1119_split.py` 中遗留的稳健统计/敏感性全局函数（Kendall Tau、稳健相关、Mann-Whitney、bootstrap CI、异常检测、rolling/sample 分析等），入口脚本现在完全依赖 `daima/fa_stat_utils` 里的一份实现，避免再次漂移。
- 顺带移除了 `false_discovery_control_impl`、`ensure_list` 等未再使用的导入，让 split 脚本真正退化成纯 orchestrator。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出目录：`stage_stat_helpers_cleanup/`
  - 比对：`python tools/compare_stage_outputs.py stage_stat_helpers_cleanup`，全部 CSV 与带参数 TXT 对应 `baseline_outputs/` 报告 `Match`，日志差异仅为时间戳。


## 2025-11-27 – Phase 4.6: fa_stat_utils 公共函数升格

- 将 `fa_stat_utils` 里的占位函数全部改成对 `_impl` 的正式封装，并把 `rolling_window_analysis` / `temporal_stability_analysis` / `sample_sensitivity_analysis` 的完整实现从黄金脚本迁入，保证后续模块有单一来源。
- `daima/fa_nonparam_analysis.py` 改为直接导入这些公共函数（不再通过 `_impl` 取别名），避免再次出现重复实现。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出目录：`stage_stat_helpers_exports/`
  - 对比：`python tools/compare_stage_outputs.py stage_stat_helpers_exports`，所有 CSV/TXT 与 `baseline_outputs/` 保持 `Match`。


## 2025-11-27 – Phase 4.7: Rolling/Sensitivity wrappers

- 在 `FactorAnalysis` 与 `ParameterizedFactorAnalyzer` 内新增 `analyze_rolling_ic`、`analyze_temporal_stability`、`analyze_sample_sensitivity` 三个便捷入口，直接委托给 `fa_stat_utils` 封装好的实现，未来若需要在主流程或人工调试中调用这些分析，就不必复制逻辑。
- `daima/fa_param_analysis.py` 与 `daima/fa_nonparam_analysis.py` 同步补充导入，彻底杜绝 `_impl` 别名或多份实现的可能性。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出目录：`stage_stat_helpers_wrappers/`
  - 对比：`python tools/compare_stage_outputs.py stage_stat_helpers_wrappers`，全部 CSV 及带参数 TXT 与 `baseline_outputs/` 一致（TXT 仅时间戳差异）。


## 2025-11-27 – Phase 4.8: 辅助分析报告接入

- 功能扩展：
  - `daima/fa_nonparam_analysis.py` 内 `generate_auxiliary_analysis_report` 升级为真正的统计汇总：逐因子收集滚动 IC（含窗口样本数、衰减与稳定性指标）、按窗口包装 `temporal_stability_analysis` 结果，并输出样本敏感性（含成功率、IQR、整体稳健性）到新的 `辅助分析报告_*.txt`，同时生成结构化 `辅助分析报告摘要_*.csv` 方便下游分析。
  - `yinzifenxi1119_split.py` 在主流程中于非参数 & 带参数报告之间调用该辅助分析生成器，保持 run_full 流程自动出报告。
  - `tools/compare_stage_outputs.py` 增加了对“辅助分析报告”系列 TXT/CSV 的匹配与哈希过滤（同样忽略 `生成时间:` 行），并将首个基线样本拷贝至 `baseline_outputs/` 便于后续回归。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 产物存放：`stage_auxiliary_report_csv/`，包含所有既有 CSV/TXT 以及新增 TXT/CSV 辅助分析文件。
  - 对比：`python tools/compare_stage_outputs.py stage_auxiliary_report_csv`，所有受控 CSV、辅助 TXT 及“带参数”TXT 与 `baseline_outputs/` 同步（TXT 比对继续忽略时间戳行）。


## 2025-11-27 – Phase 4.9: 辅助稳健性评分融合

- 主流程顺序调整：`yinzifenxi1119_split.py` 先调用 `generate_auxiliary_analysis_report` 并打印“整合辅助稳健性分析数据…”，确保 `self.analysis_results` 在生成主报告前就挂载 `auxiliary_stats`/`integrated_scores`，不再额外提示“生成辅助分析报告…”。  
- `daima/fa_nonparam_analysis.py` 将辅助统计 dict 与 `integrated_scores` 同步写回 `analysis_results[factor]['auxiliary_stats']`，方便报告层直接读取，无需重复计算。  
- `daima/fa_nonparam_report.py` 新增格式化工具，正/负因子段落会输出“评分融合：基础/滚动/时序/样本 → 综合”以及“稳健性分析”三大子段（滚动 CV & 半衰期、时序自相关/换向、样本敏感性与跨样本方差），真正把辅助报告整合进 `因子分析详情_精简版_*`.  
- 副产品：`baogao/` 只保留结构化 `辅助分析报告摘要_*.csv`，不再生成独立 TXT；主报告内容同步扩容以覆盖原辅助信息。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 产物：`baogao/因子分析详情_精简版_20251127_131839.txt`（含稳健性段落）、`baogao/辅助分析报告摘要_20251127_131830.csv`
  - 对比：`python tools/compare_stage_outputs.py baogao` → 失败，原因：`baseline_outputs/` 目录缺失，待补齐基线后再做哈希校验


## 2025-11-27 – Phase 5.0: 摘要模式 & 稳健性评分 CSV

- CLI：`yinzifenxi1119_split.py` 增加 `--summary-report` 开关，启用后 `_fa_generate_factor_analysis_report`/正负向分析均走“评分融合 + 稳健性摘要”三行展示，正文头部提示当前为精简模式，方便移动端或快速复盘。
- 报告：`daima/fa_nonparam_report.py` 新增 `_fa_format_stability_summary`/`_fa_render_factor_summary`，摘要模式下聚合 CV/半衰期/Lag1/样本Std/成功率，默认模式仍输出完整段落。
- 数据：`daima/fa_nonparam_analysis.py` 的 `generate_auxiliary_analysis_report` 现在在 `summary_df` 中附带 `base_score/rolling_score/.../rating` 与 `rolling_cv_avg` 等关键统计，同时额外落地 `辅助分析稳健性评分_<timestamp>.csv`，方便 notebook/BI 直接引用。
- 工具：`baseline_outputs/` 重新以当前 `baogao/` 内容为基线，`tools/compare_stage_outputs.py` 先匹配同名文件再退回去时间戳前缀，确保多时间戳文件也能稳定对比。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`（默认）与 `python yinzifenxi1119_split.py --summary-report`
  - 产物：`baogao/因子分析详情_精简版_20251127_140437.txt`（完整版）、`因子分析详情_精简版_20251127_140509.txt`（摘要版）、`辅助分析稳健性评分_20251127_140437.csv`
  - 对比：`python tools/compare_stage_outputs.py baogao` → 全部 `Match`


## 2025-11-27 – Phase 5.1: compare 工具扩展 & 基线刷新（15:19 run）

- 工具增强：
  - `tools/compare_stage_outputs.py` 的 TXT 覆盖范围扩展到 `因子分析详情_精简版` 与辅助报告，新增 `_detect_variant()` 根据文件内容区分“完整版/摘要版”，确保摘要模式输出与基线逐一对比。
  - TXT 哈希时继续剔除 `生成时间:` 行，另外发现 `因子分析日志_*` 中嵌入大量时间戳与文件名，已从自动回归列表中剔除，避免无意义的 `Mismatch`。
- 基线重建：
  - 清空旧的 `baseline_outputs/` 并拷贝当前 `baogao/` 快照（包含 12:20~15:07 的完整版/摘要版 + 辅助 CSV），形成新的黄金参考集。
  - 之后重新运行 `python yinzifenxi1119_split.py`（默认）与 `python yinzifenxi1119_split.py --summary-report`，生成 15:19 批次的报告/CSV（如 `因子分析详情_精简版_20251127_151929.txt`、`因子分析详情_精简版_20251127_152007.txt`、`带参数因子综合分析报告_20251127_152016.txt` 等）。
- 回归验证：
  - 命令：`python tools/compare_stage_outputs.py baogao`
  - 结果：所有 CSV、带参数 TXT、精简版报告 TXT 以及辅助 CSV 均报 `Match`，证明新版 compare 脚本 + 新基线可以稳定捕捉后续改动；日志文件因动态内容被排除在回归范围之外。


## 2025-11-27 – Phase 5.2: Overall IC 维度纳入评分体系

- 新的评分维度：
  - 在 `daima/fa_nonparam_helpers.py` 引入 `_score_sample_size_component`、`_extract_overall_metrics`、`_compute_overall_score`，读取 `extra_stats['overall_metrics']` 中的整体 IC / IR / p 值 / 样本量，并通过 `overall_score`（0–100）反映整段样本表现。
  - 综合权重调整为：基础 45% + 整体 15% + 滚动 15% + 时序 12.5% + 样本 12.5%，评级映射保持不变。
  - `compute_integrated_factor_scores`、分类与报告模块均增加 “整体得分” 字段，评分文案改写为 “基础 / 整体 / 滚动 … → 综合”。
- 报告与 CSV：
  - `generate_auxiliary_analysis_report` 在 `summary_df` 及 `score_df` 中写入 `overall_ic/overall_ir/overall_sample_size/overall_score`，`辅助分析稳健性评分_*` CSV 现在含有整体表现列。
  - `因子分析详情_精简版_*` 在正/负向摘要及详细段落都新增 “整体表现参考：IC…样本…” 文案，并在优秀因子分析中列出 “整体表现得分”。
- 验证流程：
  - 命令：`python yinzifenxi1119_split.py`（产出 `因子分析详情_精简版_20251127_162309.txt`、`辅助分析稳健性评分_20251127_162259.csv` 等）。
  - `python tools/compare_stage_outputs.py baogao` → 新文件与基线存在结构性差异（新增整体列/文案），已记录 `Mismatch` 作为预期结果；待确认后将把 16:23 批次替换进 `baseline_outputs/`。


## 2025-11-27 – Phase 5.3: 评分标签释义 & 正向详细分析补强

- 报告可读性：
  - `评分融合` 文案改为“基础表现45%（IC/IR/统计显著性）/ 整体样本表现15% / 滚动稳定度15% / 时序一致性12.5% / 样本稳健性12.5%”，直接反映各维度含义及权重。
  - 正向因子如未出现 A 级，也会按照综合得分排序输出 “核心正向因子” 详细分析，同步展示基础/整体/稳健得分、整体表现参考与 IC 计算摘要，避免出现仅有策略建议的“空心”段落。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`
  - 输出：`因子分析详情_精简版_20251127_163049.txt` 等新报告显示完整的正向详解与新的评分标签。
  - `compare_stage_outputs` 与基线相比会出现文案差异（预期），待下一轮基线更新后统一纳入回归。


## 2025-11-27 – Phase 5.4: 日度权重降级 & 整体权重加严

- 综合权重：将基础（日度）权重从 45% 降至 35%，整体样本权重升至 25%，滚动/时序/样本继续为 15%/12.5%/12.5%，以弱化高噪声日度窗口、突出整体表现。
- 阈值收紧：
  - `sample_size_component` 只有样本≥500 才满分，200~400 递减，避免 800+ 样本自动 100 分；
  - `sample_std` 只有 std≤0.005 才拿 100 分，0.005~0.02 区间分段下降至 60，>0.02 再降到 30。
- 报告同步：`评分融合` 标签更新为 “基础表现35% / 整体样本表现25% …”，保证文案与新权重一致。
- 影响：整体评分不再轻易接近 100，更能体现“整体样本强但日度噪音大”的结构，对比 `因子分析详情_精简版_20251127_163608.txt` 可见整体得分明显回调。


## 2025-11-27 – Phase 5.5: 动态可靠性加权 + 报告透明化

- 可靠性输入：
  - `calculate_ic` 记录 `daily_sample_cv`、Top5 日期集中度、缺样比例、有效日占比、窗口策略说明等字段，`extra_stats` 还同步整体 IC 置信区间宽度。
  - 辅助分析阶段把滚动/时序/样本统计聚合到 `metric_summary`，CSV 加入 `ic_mode`、`daily_points`、`reliability_*`、`weight_*` 列。
- 动态权重：
  - `compute_integrated_factor_scores` 调用 `_derive_reliability_scores`，依据日度/整体/滚动/时序/样本五维可靠性（0.3–1.0）调整权重；使用 `bounded_normalize` 保证权重落在 5%~55% 区间。
  - `integrated_scores` 公开 `component_weights`、`reliability_scores`、`reliability_labels`、`weight_notes`，供报告与 CSV 使用。
  - `_score_sample_size_component` 与 `_score_sample_std` 再次收紧，整体得分即使样本足量也需满足更严苛的波动阈值。
- 报告增强：
  - `评分融合` 行展示“基础表现（权重18%，可靠性=低）…”等动态标签；
  - 新增 “评分权重调整” 与 “数据可靠性评估” 段落，正/负向摘要与详细分析都列出低样本提示，确保所有因子获得同等详尽说明；
  - 辅助 CSV/摘要寫入 `weight_*` 与 `reliability_*`，方便 compare 工具检查。
- 验证：
  - 命令：`python yinzifenxi1119_split.py`（产出 17:26 批次，如 `因子分析详情_精简版_20251127_172621.txt`、`辅助分析稳健性评分_20251127_172611.csv`）。
  - 命令：`python tools/compare_stage_outputs.py baogao` → 所有新报告/CSV 因字段扩列出现 `Mismatch`，属预期，需要在验收后刷新 `baseline_outputs/`。
