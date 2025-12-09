# Codex Project Memory – yinzifenxi 1129

## Snapshot
- **Primary goal**: Run the repaired因子分析 pipeline (non-parametric + 带参数) against Excel sources in `shuju/biaoge`, produce HTML/CSV/XLSX reports under `baogao`, and log the process for auditing.
- **Key entry**: `python yinzifenxi1119_split.py [--summary-report]` → wraps `yinzifenxi/yinzifenxi_main.py:main`.
- **Data expectation**: Excel files such as `shuju/biaoge/回测详细数据结合股本分析：创业板2025.xlsx` configured in `yinzifenxi/fa_config.py`.

## Directory Map (top level)
- `yinzifenxi/`: Python package with the full analysis pipeline (parsers, configs, stats, reporting). This is where most edits happen.
- `shuju/`: Data + AKShare helpers. Contains raw tables (`biaoge`), cache (`ak_cache`), AKShare client scripts (`akshare/`), plus doc notes under `shuju/MEMORY`.
- `memory/`: Planning / progress docs plus this project memory file.
- `tools/compare_stage_outputs.py`: Hash-based regression checker for generated CSV/TXT reports.
- `mcp-filesystem/`: Node-based MCP server exposing whitelisted filesystem tooling (list/read/search/write). Built with `@modelcontextprotocol/sdk`.
- `codex_cli.py`, `codex_example.py`, `codex.bat`: Token counting CLI helpers unrelated to the factor analysis flow but shipped with the repo.
- `yinzifenxi1119_split.py`: Legacy-compatible launcher for the package entry point.

## yinzifenxi Package Overview

### Configuration & Logging
- `fa_config.py`
  - Defines data/report directories (`shuju`, `baogao`) and ensures they exist.
  - Centralizes `DEFAULT_DATA_FILES`, `FACTOR_COLUMNS`, `RETURN_COLUMN`, parsing aliases/types, market segment rules, and reliability weighting bounds.
  - `build_report_path` maps filenames to the report output folder.
- `fa_logging.py`
  - `Logger` redirects `stdout` to a timestamped log file (default `因子分析日志_*.txt` in `baogao`). Only ERROR/INFO (and optionally WARN) lines are persisted unless `FA_LOG_VERBOSE=1`.

### Data ingestion
- `excel_parser.py`
  - Multi-engine Excel/CSV reader (`pyarrow`, `openpyxl`, fallback `pandas`) with header detection, alias normalization, type conversion, and diagnostics capture.
  - `load_excel_sources()` merges multiple files and returns data + column-level failure stats + missing column tracking to help `FactorAnalysis` mark unavailable factors.

### Non-parametric pipeline (`fa_nonparam_analysis.py` + helpers/reports)
- `FactorAnalysis` class responsibilities:
  - Manage Excel paths (`DEFAULT_DATA_FILES`), load via `excel_parser`, track missing columns and parse diagnostics.
  - `preprocess_data()` performs numeric casting, percent stripping, winsorization/standardization (configurable), anomaly tracking (missing/outliers/unique ratio), and date normalization.
  - `calculate_ic()` and `_compute_rolling_ic` variants compute IC/IR/t-statistics (Spearman default, optional Pearson) with rolling-window stats.
  - `calculate_group_returns()` sorts factor values, splits into `DEFAULT_GROUP_COUNT` bins, and reports average returns, p-values, confidence diagnostics, plus win-rate/drawdown/Sharpe per bucket.
  - `run_factor_analysis()` orchestrates preprocessing, IC evaluation, auxiliary stats (rolling/temporal/sample stability via `fa_stat_utils`), and feeds output to report generators.
  - Attributes like `analysis_results`, `auxiliary_stats`, `segment_overview`, and `anomaly_stats` are the canonical caches consumed by helper/report modules.
- `fa_nonparam_helpers.py`
  - Computes ratings, suggested weights, integration of multi-dimensional scores (基础/整体/滚动/时序/样本 + 动态可靠性 from `RELIABILITY_CONFIG`).
  - `compute_integrated_factor_scores` (near bottom) attaches these metrics back onto `FactorAnalysis.analysis_results`.
  - Provides Markdown builders for strategy notes (`_fa_generate_factor_classification_overview`, `_fa_generate_positive_factors_analysis`, etc.).
- `fa_nonparam_report.py`
  - Builds summary CSV + HTML detail report via `HTMLReportBuilder` (from `fa_report_utils.py`).
  - Sections include classification overview, positive/negative factor cards, textual diagnoses, scoring standards, and optional summary-only mode (`--summary-report` flag).
  - Saves reports as `因子分析详情_精简版_*.html` and `因子分析汇总_*.csv` inside `baogao`.
- `fa_report_utils.py`
  - Provides the HTML scaffold (CSS, metric cards, tables, alerts, etc.) shared by both non-param and parameterized reports.

### Parameterized pipeline
- `fa_param_analysis.py`
  - `ParameterizedFactorAnalyzer` accepts the already loaded/preprocessed DataFrame, ensures numeric factor columns, and runs per-factor qcut groupings.
  - Computes per-bin averages, win rates, max drawdown, adaptive-but-currently simplified annualization (252-day assumption), Sharpe/Sortino, and long-short stats.
  - Reuses shared stats helpers (`rolling_window_analysis`, `sample_sensitivity_analysis`) for advanced diagnostics.
- `fa_param_helpers.py`
  - Scores each parameter bucket (wins, return, Sharpe, volatility, drawdown) and smooths the final rating by blending neighbors to prevent jagged rankings.
- `fa_param_report.py`
  - Generates `带参数因子综合分析报告_*.html` plus `带参数因子分析数据_*.xlsx` with conditional formatting (top/bottom quantile highlights via `openpyxl`).
  - Sections: overview cards, positive/negative leaderboards, TOP5 recommendations, detailed cards per bucket (with direction tags), strategy and risk bullet lists.

### Statistical utilities (`fa_stat_utils.py`)
- Houses reusable math helpers: correlation variants (Spearman, Kendall, robust median/trimmed), bootstrap confidence intervals, Mann-Whitney test, outlier detection, rolling IC analysis, temporal stability, sample sensitivity, and reliability weight normalization.
- Designed so other modules import light wrappers (`kendall_tau_corr`, `rolling_window_analysis`, etc.) without worrying about scipy availability (fallbacks exist).

### Excel knowledge base
- `yinzifenxi/MEMORY/Excel数据处理...md` contains prior context about Excel quirks; useful when adjusting `excel_parser`.

## Data & AKShare Utilities (`shuju/`)
- `akshare/akshare_client.py`: High-level AkShare client with rate limiting, anti-crawl headers/proxies, persistent cache (`ak_cache`), and performance telemetry for pulling holdings, management, shareholder info, etc.
- `akshare/akshare_config.py`: Dataclass for tuning client behavior via env vars (intervals, proxies, cache directory).
- Additional AKShare memory docs (`shuju/MEMORY/*.md`) describe previous optimization attempts (Qwen/GLM/Cline notes).
- Excel data tables stored under `shuju/biaoge/` (currently the创业板2025回测明细). Scripts like `BAOSTOCK...py`, `数据清洗...py` are standalone prep utilities for specific vendor exports.

## Supporting Tooling
- `tools/compare_stage_outputs.py`
  - Usage: `python tools/compare_stage_outputs.py stage_dir --baseline baseline_outputs`
  - Hashes CSV/selected TXT outputs, ignores timestamp suffixes and volatile header lines, and reports Match/Mismatch per file. Ideal for regression testing after refactors.
- `mcp-filesystem/`
  - Node/TS project providing Model Context Protocol filesystem tooling via stdio. Offers `list_directory`, `read_text_file`, `search_files`, `write_file`, etc., and constrains access to allowed directories supplied on startup.
- Codex CLI scripts (`codex_cli.py`, `.bat`, `codex_example.py`) are standalone helpers for token counting/encoding using `tiktoken` + `openai`.

## Typical Workflow
1. Place the latest Excel(s) under `shuju/biaoge/` and update `DEFAULT_DATA_FILES` in `fa_config.py` if needed.
2. (Optional) Fetch/refresh raw data via AKShare scripts, then join into the master Excel.
3. Run `python yinzifenxi1119_split.py` for full HTML + CSV outputs, or add `--summary-report` to limit HTML sections.
4. Inspect logs in `baogao/因子分析日志_*.txt`. Reports include:
   - `因子分析详情_精简版_*.html`
   - `因子分析汇总_*.csv`
   - `带参数因子综合分析报告_*.html`
   - `带参数因子分析数据_*.xlsx` (with highlighted top/bottom quantiles)
5. Use `tools/compare_stage_outputs.py` to diff against `baseline_outputs/` when validating refactors.

## Key External Dependencies
- Python: `pandas`, `numpy`, `scipy` (optional but enables richer stats), `matplotlib`/`seaborn` (plots if available), `pyarrow`, `openpyxl`.
- Data: Excel files with factor columns listed in `FACTOR_COLUMNS`, containing at least `股票代码/名称/信号日期/次日开盘买入持股两日收益率`.
- Optional: `akshare`, `requests` for data acquisition scripts; `openpyxl` for Excel highlighting; fonts supporting Chinese for charts (`SimHei` configured in `yinzifenxi_main.py`).

## Operational Notes
- Reliability weighting is controlled in `fa_config.RELIABILITY_CONFIG` – base weights + allowed scaling + drop threshold (0.35). Changing this impacts helper scoring logic.
- Reports assume the `baogao/` directory exists (auto-created). Keep filenames ASCII-safe when possible to avoid Windows path issues.
- Many helper strings are Chinese; keep files saved in UTF-8 to prevent mojibake when editing.
- When extending factor coverage, update `FACTOR_COLUMNS`, `DATA_PARSE_CONFIG['column_aliases']`, and consider alias entries for `excel_parser`.
- `FactorAnalysis` automatically filters `unavailable_factors` from parsing diagnostics; check `self.unavailable_factors` if certain columns vanish.
- Parameterized analysis expects the same factor list; ensure preprocess step keeps enough rows (>10) per factor to avoid empty stats.

## Fast Pointers
- Entry: `yinzifenxi/yinzifenxi_main.py:main` (CLI args defined near top).
- Logs: `baogao/因子分析日志_*.txt`.
- Non-param summary builder: `fa_nonparam_report._fa_generate_summary_report`.
- Parameterized report builder: `fa_param_report._fa_generate_parameterized_report`.
- Data parser configs: `yinzifenxi/fa_config.py` and `yinzifenxi/excel_parser.py`.
- AKShare client metrics: `AkShareClient.metrics()` for telemetry if requests start failing.

Keep this file updated after major structural changes so future sessions can jump directly to the right modules/config knobs.
