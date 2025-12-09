# Repository Guidelines

## Global Codex Rules (must follow)
### 1. Roles
- Architect: plan system architecture and technology path; ensure sensible technical choices.
- Full-stack engineer: build frontend, backend, database, and API integrations.
- Test engineer: design test cases and CI flow; own product quality.
- Docs engineer: maintain API docs and developer guides; keep docs consistent with code.

### 2. Collaboration
- Replies, comments, and docs should be in Chinese; when using English terms, add concise Chinese explanations. Keep logic consistent and structure clear.
- Keep code/docs concise and developer-friendly; prefer official SDKs, modern syntax, and stable dependencies.
- Format code with `eslint` / `prettier`; store documentation under `/docs/`.
- Prefer responses that include code which runs as-is.

### 3. Coding & Files
- Use UTF-8 (no BOM) for all new/edited files; if another encoding is encountered, convert and notify.
- Logs and error messages should be in Chinese; add Chinese explanations when needed without breaking existing format.
- Comments should be concise and in Chinese, focusing on key logic.

### 4. Windows Environment (highest priority; if conflicts arise, this wins)
- Prefer PowerShell (`pwsh.exe`). In WSL2, call Windows scripts like: `pwsh.exe -c "pnpm typecheck"`.
- Never auto-install dependencies; always get user confirmation before any install.
- Avoid Unix tools (e.g., `sed`); use native PowerShell commands instead.

### 5. Tool Priority
- File names: `fd`
- Text content: `rg`
- Code structure (imports, JSX, etc.): `sg`

### 6. Additional Operating Rules
- Project documents (plans, reports) must be Markdown files saved under `memory/`.
- 文档命名：使用“文档正常名称-生成该文档的AI名称-具体时间”（示例：`数据质量审查-Codex-20251206-203406.md`）。
- 当生成新方案/分析时，默认同时在仓库根的 `memory/` 目录写入对应的 Markdown 文档，遵循上述命名规范。
- 默认长时间任务运行方式：在仓库根目录使用 nohup 无超时后台执行（示例：命令 `python3 -m yinzifenxi.yinzifenxi_main --summary-report`，位置 `/mnt/c/Users/NINGMEI/Documents/trae_projects/1206/adriano88427-1206`，输出重定向 `run.log`，记录进程 PID）。
- Do not modify code unless I explicitly instruct you to; ask for permission before any code change.
- When my request is unclear, or options/assumptions are needed, ask me immediately; raise concerns or objections early.
- After any code change, self-check for correctness, requirement fit, and alignment with the agreed plan; if issues are found, fix them before handing off.
- Safety first: if a change might break project structure, slow performance significantly, affect other modules or outputs, or reduce accuracy/reliability, pause and report to me for discussion.
- After each program run, review logs and generated artifacts to detect anomalies, verify whether the last changes met expectations, and surface any further optimization opportunities; report the findings back to me.
- Communicate with me in Chinese at all times.
- 当需要用户在多条执行路线中选择时，先提供专业评估后再征询选择：逐条写优点/风险/适用场景，并给出推荐（示例如容忍阈值 vs 复刻逻辑的对比）。
- 用户是编程/数学/统计/金融的新手：对话、选项、CONFIG 注释必须更详细，避免术语堆砌，必要时给出具体评估或操作指南，遇到专业概念需额外解释。

## Project Structure & Module Organization
- `yinzifenxi/`: Core factor-analysis code. `yinzifenxi_main.py` orchestrates the pipeline; `fa_nonparam_analysis.py` and `fa_param_analysis.py` handle single-factor logic; `fa_dual_*` modules cover dual-factor flows; helpers live in `fa_stat_utils.py`, `fa_nonparam_helpers.py`, and reporting utilities (HTML/CSV/Excel) are in `fa_*_report.py`.
- `shuju/`: Data inputs. Place Excel sources under `shuju/biaoge/`; akshare caches in `shuju/ak_cache/`; intermediate dumps may go under `shuju/MEMORY/`.
- `baogao/`: Generated outputs (CSV/Excel/HTML) and run log (`baogao/log_metrics.csv`). Keep writable; avoid committing large artifacts unless required.
- `tools/`: One-off utilities (column fixes, diffing outputs).
- `memory/`: Design notes, plans, prior analyses—read-only reference.

## Build, Test, and Development Commands
- Create env and install deps (Python 3.9+ recommended):
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate
  pip install pandas numpy scipy matplotlib seaborn openpyxl pyarrow
  ```
- Run main pipeline from repo root (uses `fa_config.py` paths):
  ```powershell
  python -m yinzifenxi.yinzifenxi_main --summary-report
  ```
- Enable debug logging and capture raw logs:
  ```powershell
  python -m yinzifenxi.yinzifenxi_main --debug --dump-debug-log baogao/debug.log
  ```
- IC 验证默认顺序（不改主报告结构）：
  1) 先跑纯 pandas 基准对照（仅验证）：  
     `python -m yinzifenxi.fa_ic_validation --pandas-benchmark-only --data baogao/benchmark_input_long.csv --return-col 次日开盘买入持股两日收益率 --target baogao/因子分析汇总_最新.csv --output-dir baogao --weighted`  
     容忍阈值建议 |差值|≤0.006；超出再做后续验证。
  2) 如需基础验证：`python -m yinzifenxi.yinzifenxi_main --summary-report --run-ic-validation`（默认开关均关，只生成差异/审计）。
  3) 如需自校验/CI：在 `fa_config.py` 开启 `IC_DUMP_DAILY_ENABLED=True` 跑主流程，再执行 `python -m yinzifenxi.ic_selfcheck ... --tolerance 0.0005`；CI 通过 `fa_ic_validation` 的 `--enable-ci` 运行。

## Coding Style & Naming Conventions
- Python, 4-space indent, `snake_case` for functions/variables, `PascalCase` for classes; prefer f-strings and type hints.
- Centralize paths/switches in `fa_config.py`; avoid hard-coded file names—use `DATA_DIR`, `DATA_TABLE_DIR`, `ANALYSIS_SWITCHES`.
- Use `fa_logging.Logger` instead of plain `print` for anything that should reach `baogao` logs.
- Reuse helpers in `fa_nonparam_helpers.py` / `fa_stat_utils.py` for metrics instead of duplicating logic.
- IC 配置基线：`ENABLE_COVERAGE_HINT=True`，`ENABLE_IC_VALIDATION/BENCHMARK/CI=False`，`IC_DUMP_DAILY_ENABLED=False`，`FA_AUX_ENABLED=False`，`ZERO_PRICE_ACTION=off`；改动前请确认开关含义。
- 带参数双因子排行榜：协同增益需 >20% 方可入榜；样本下限 500；样本计分恢复为按样本分位线性插值（与其他指标一致，不对 <500 额外扣分）；展示行数为 `max_rank_display` 的两倍（默认 40 行）。

## Testing Guidelines
- No formal automated tests yet; validate by running the pipeline on a small Excel sample in `shuju/biaoge/`, then confirm new rows in `baogao/log_metrics.csv` and generated HTML/CSV/Excel reports.
- For numerical changes, spot-check IC/IR outputs and classification summaries; compare to prior artifacts in `baogao/`.
- If adding new helpers, consider lightweight `assert` checks or a small `pytest` module under `yinzifenxi/` for regressions.

## Commit & Pull Request Guidelines
- Use clear, action-oriented commits (e.g., `feat: add dual-factor report guardrails`, `fix: repair percent column parsing`); keep related changes together.
- PRs should describe scope, data sources touched, and verification steps (commands run, sample files, key outputs). Link relevant design notes from `memory/` when useful.
- Avoid committing large data/report artifacts unless explicitly requested; treat `shuju/biaoge` inputs as local-only unless sanitized. Never commit secrets.
