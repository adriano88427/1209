# yinzifenxi1119.py Split Plan (v5, English Version)

> This plan replaces the previous mixed‑encoding version.  
> It is kept **entirely in English** to avoid encoding issues, while the actual
> Python code and reports remain in Chinese and must not change their text.

---

## 1. Overall Goals and Hard Constraints

1. **Behavioral equivalence**
   - All refactors must keep:
     - Business logic and formulas unchanged.
     - Numerical results identical (IC, IR, returns, etc.).
     - All generated CSV/TXT report contents identical, byte‑for‑byte, except for
       timestamps or run‑time metadata which are expected to differ.

2. **Golden script**
   - `yinzifenxi1119.py` is the **golden reference**:
     - Used for official runs and generating `baseline_outputs/`.
     - Must **never be modified** during refactor/split work.

3. **Refactor plane**
   - All refactor work happens in:
     - `yinzifenxi1119_split.py` (split/experimental entry script).
     - `daima/fa_*.py` modules (logging, config, utilities, analysis, reports).

4. **External interface compatibility**
   - Keep these usage patterns working:
     - `python yinzifenxi1119.py` — original behavior.
     - `python yinzifenxi1119_split.py` — refactored behavior, expected to be
       equivalent to the original.
     - `from yinzifenxi1119 import FactorAnalysis, ParameterizedFactorAnalyzer`
       — external imports must remain valid.

5. **Safety on code changes**
   - Do not introduce new side‑effects at module import time:
     - No top‑level `print` / file writes / network calls in newly created
       modules (except for logic explicitly kept from the original script).
   - Avoid tools that rewrite entire `.py` files and re‑encode text:
     - No direct PowerShell `Set-Content` on `.py` with Chinese text.
     - Use small, focused patches (`apply_patch`) or copy‑paste within the IDE.

---

## 2. File and Module Structure After Split

### 2.1 Core scripts
- `yinzifenxi1119.py`
  - Original, unchanged script.
  - Single source of truth for current behavior and outputs.

- `yinzifenxi1119_split.py`
  - Experimental split script, initially a copy of the original.
  - All refactor steps are applied here.
  - Must remain runnable at each stage and stay behaviorally equivalent.

### 2.2 Modules under `daima/`

- `daima/fa_logging.py`
  - Contains `Logger` class.
  - Behavior matches the original inline `Logger`:
    - Creates `因子分析日志_YYYYMMDD_HHMMSS.txt`.
    - Writes start/end headers.
    - Mirrors `print` output to both terminal and log file.

- `daima/fa_config.py`
  - Contains configuration constants (currently only `DEFAULT_DATA_FILE`).
  - No top‑level printing or environment checks.

- `daima/fa_stat_utils.py`
  - General statistical utilities and helpers:
    - Type‑safety functions: `ensure_list`, `safe_len`, `safe_ensure_list`.
    - Robust statistics: Kendall’s Tau, robust correlation, Mann–Whitney U.
    - Bootstrap CI, outlier detection, sensitivity analysis, FDR control.
    - Rolling window IC analysis, temporal stability, sample sensitivity.
    - Custom Spearman correlation, annual return calculations.
  - Pattern:
    - “Stub” versions keep function names consistent with the original script.
    - `_impl` versions hold the full implementation copied from the script.
    - The split script gradually switches to these module functions.

- Planned (later phases, not yet implemented):
  - `daima/fa_nonparam_analysis.py`
    - Core logic for non‑parameterized factor analysis (`FactorAnalysis`).
  - `daima/fa_nonparam_report.py`
    - TXT/CSV report generation for `FactorAnalysis`.
  - `daima/fa_param_analysis.py`
    - Core logic for parameterized factor analysis (`ParameterizedFactorAnalyzer`).
  - `daima/fa_param_report.py`
    - TXT/CSV report generation for parameterized factors.

---

## 3. Phase Plan (High‑Level)

### Phase 0 – Baseline
- Run `python yinzifenxi1119.py`.
- Copy all generated TXT/CSV/log files into `baseline_outputs/`.
- Use these files as the golden reference for all future comparisons.

### Phase 1 – Logger extraction and switch
1. Create `yinzifenxi1119_split.py` as a copy of `yinzifenxi1119.py`.
2. Create `daima/fa_logging.py` with `Logger` copied from the original.
3. In `yinzifenxi1119_split.py`:
   - Add `from daima.fa_logging import Logger as ExternalLogger`.
   - Change `logger = Logger()` to `logger = ExternalLogger()`.
4. Run `yinzifenxi1119_split.py`, store outputs in `stage1_logger_outputs/`
   and `stage2_logger_outputs/` as needed.
5. Compare CSVs (and optionally TXT/logs) vs `baseline_outputs/` using hash
   or `fc /b`. Expect CSVs to be identical.

### Phase 2 – Statistical utilities (fa_stat_utils)
1. Create `daima/fa_stat_utils.py`:
   - Add `ensure_list`, `safe_len`, `safe_ensure_list`.
   - Add stub versions of robust stats and FDR functions.
   - Add full `_impl` versions for:
     - `kendall_tau_corr_impl`, `robust_correlation_impl`,
       `mann_whitney_u_test_impl`,
       `bootstrap_confidence_interval_impl`, `detect_outliers_impl`,
       `sensitivity_analysis_impl`, `false_discovery_control_impl`,
       `custom_spearman_corr`, and annual return helpers.
2. In `yinzifenxi1119_split.py`:
   - Import from `fa_stat_utils`:
     ```python
     from daima.fa_stat_utils import (
         ensure_list, safe_len, safe_ensure_list,
         false_discovery_control_impl, custom_spearman_corr,
     )
     ```
   - Gradually replace local implementations with module calls:
     - First replacement successfully validated:
       - `custom_spearman_corr` — local function removed and script now uses
         `fa_stat_utils.custom_spearman_corr`.
       - Outputs in `stage_custom_spearman_outputs/` have CSVs identical to
         `baseline_outputs/`.
   - For each future function replacement:
     1. Copy implementation into `fa_stat_utils` (if not already there).
     2. Change the split script to import and call the module function.
     3. Run split script, store outputs (e.g. in `stageX_outputs/`).
     4. Compare CSVs vs `baseline_outputs/`.
     5. Only accept the change if CSVs are identical and TXT/log differences
        are due only to timestamps.

### Phase 3 – FactorAnalysis split (non‑param analysis / reports)
> Planned, not yet executed.

1. Create `fa_nonparam_analysis.py`:
   - Move `class FactorAnalysis` core analytical methods here (data loading,
     preprocessing, IC calculation, grouping, annualization, scoring).
   - Keep `generate_*report` methods as thin wrappers calling report functions.

2. Create `fa_nonparam_report.py`:
   - Implement report generation functions:
     - `_fa_generate_summary_report(self)`
     - `_fa_generate_factor_analysis_report(self, ...)`
     - `_fa_generate_positive_factors_analysis(self)`
     - `_fa_generate_negative_factors_analysis(self)`
   - Each function body is copied verbatim from the original methods in
     `FactorAnalysis`.

3. In `yinzifenxi1119_split.py`:
   - Import `FactorAnalysis` from `fa_nonparam_analysis`.
   - Keep the external interface unchanged.
   - Run, compare outputs vs `baseline_outputs/`.

### Phase 4 – ParameterizedFactorAnalyzer split (param analysis / reports)
> Completed on 2025-11-27 (see progress log Phase 4.1–4.4).

1. Created `fa_param_analysis.py` and `fa_param_report.py`, migrating the full class + report logic out of the split script.
2. `yinzifenxi1119_split.py` now instantiates the module versions, keeping the public interface intact.
3. Verification runs (e.g., `stage_param_analysis_wiring/`, `stage_param_report_module/`) all matched `baseline_outputs/` per `tools/compare_stage_outputs.py`.

### Phase 5 – Auxiliary analytics & validation tooling

- Extend the `FactorAnalysis` auxiliary diagnostics so rolling IC / temporal stability / sample sensitivity are exported both as human-readable TXT and as a structured CSV (`辅助分析报告摘要_*`) for downstream analysis.
- Update `tools/compare_stage_outputs.py` to hash-check the auxiliary TXT/CSV (filtering `生成时间:` lines just like the parameterized reports) so the new artifacts participate in every regression run.
- Continue to route any new statistical helpers through `daima/fa_stat_utils.py` first, then reference them from `daima/fa_*` modules to keep implementations single-sourced.

---

## 4. Encoding and Modification Rules

1. **Python code with Chinese text**
   - All user‑facing messages and report strings in `.py` files remain in Chinese.
   - When editing such code:
     - Avoid automated tools that rewrite entire files or re‑encode content.
     - Prefer:
       - Small, precise patches using `apply_patch`.
       - Copy‑and‑paste within the IDE where the encoding is known to be stable.

2. **English‑only documentation**
   - All internal planning and progress files under `memory/` are kept in
     English only:
     - `memory/yinzifenxi1119_split_plan.md`
     - `memory/yinzifenxi1119_progress.md`
   - This avoids encoding issues and makes it easier for tools to parse and
     edit these documents safely.

3. **Rollback policy**
   - If any refactor step causes:
     - Syntax errors,
     - Unexpected exceptions,
     - Differences in CSV contents vs `baseline_outputs/`,
   - Then:
     - Immediately revert to the previous known‑good state (using git or a
       backup copy).
     - Revisit the patch and adjust it instead of patching on top of a broken
       state.
