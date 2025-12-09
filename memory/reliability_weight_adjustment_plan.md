# Dynamic Reliability-Weighted Scoring Plan (Refined)

> Language: English for plan/code comments; runtime日志/报告保持中文。

---

## 1. Objectives & Requirements Recap

1. **Trading-profile-aware reliability**  
   Measure the reliability of each统计维度（基础日度、整体样本、滚动稳定、时序一致、样本稳健）并把交易系统的“日均交易次数、波动、集中度、合格交易日占比”等特征纳入判断，明确噪音和可信范围。
2. **Dynamic weight fusion**  
   Replace the fixed `基础35%/整体25%/滚动15%/时序12.5%/样本12.5%` with reliability-aware weights so that noisy components自动降权、可信组件上调。明确满足以下讨论结论：  
   - 日均样本不足时，基础（日度 IC/IR）权重下降，整体样本权重上升；  
   - 整体 IC/IR/P 值仍作为评分一项，但 `_score_*` 阈值更严格，避免整体样本一口气拿 99 分；  
   - 日度与整体两套计算互补，报告需指明“该因子主要由 X 维度驱动”。
3. **Transparent reporting**  
   All TXT/CSV outputs must展示“可靠性等级 + 实际权重 + 动态调整原因”，正向与负向因子输出一致的详细段落（不再只列“优质因子”）。
4. **Logging & monitoring**  
   运行日志需记录每个因子的可靠性分、权重调整、IC 计算模式（daily/daily_window/overall），方便排查。

---

## 2. Reliability Inputs (Expanded)

| Dimension | Metrics | Notes |
| --- | --- | --- |
| 基础（日度窗口） | 平均/中位每日样本、样本 CV、缺样比例、有效日度点数、窗口合并模式、min_samples_per_day 决策、Top5 日期样本占比、平均/最大“交易次数波动” | 反映日度噪音、交易集中度 |
| 整体样本 | 总样本数、Spearman/Pearson 一致性、p 值或 bootstrap CI 宽度、unique 标的数、样本在时间上的覆盖率 | 样本越多越稳定，但若集中在少数日子需降权 |
| 滚动稳定度 | 各窗口样本数、CV、MAIC、半衰期、窗口覆盖比例（有效窗口/总窗口） | 识别滚动统计是否抖动过大 |
| 时序一致性 | Lag1 自相关、趋势相关系数、换向次数、均值回归度、排名波动指标 | 自相关太高（>0.95）或换向频繁加重惩罚 |
| 样本稳健性 | 抽样成功率、跨样本 IC 方差/IQR、最差样本表现、样本集中度（最大样本占比）、成功样本天数占比 | 反映 resampling、样本敏感性质量 |

每个维度单独产生 0.30–1.00 的 reliability score。允许多个子指标合成：  
`score_base = 0.40 * f_avg_samples + 0.25 * f_missing_ratio + 0.20 * f_effective_days + 0.15 * f_concentration`.

---

## 3. Adaptive Weighting Model

1. **Base weights**: `[0.35, 0.25, 0.15, 0.125, 0.125]` 对应 `基础/整体/滚动/时序/样本`.
2. **Reliability multiplier**: `adj_weight_i = base_weight_i * clamp(reliability_i, 0.4, 1.6)`.  
   - reliability < 0.35 ⇒ `adj_weight_i = 0`（该维度不参与评分）。
3. **Normalization & clipping**:  
   - Normalize `adj_weight_i`;  
   - Clamp to `[0.05, 0.55]`；若 clipping 导致总和 ≠ 1，再次归一化；  
   - Limit “单维最高不超过 0.60” 以防整体权重过度集中。
4. **Trade-profile hints**:  
   - 若 `avg_daily_samples < 2` 且 `missing_ratio > 0.4`，强制 `weight_base ≤ 0.20`；  
   - 若整体样本 ≥ 800 且 bootstrap CI ≤ 0.015，则 `weight_overall ≥ 0.30`，但 `_score_overall` 中将评分阈值收紧（例如 `IC≥0.04` 才≥90）。  
5. **Outputs**:  
   - `reliability_scores`: raw 0–1 value + label (“高/中/低”);  
   - `final_weights`: normalized percentages；  
   - `weight_notes`: textual reason (“日均样本不足→基础降权；整体样本充分→整体升权”).

---

## 4. Threshold Mapping & Formulas

| Metric | Score≈1.0 | Score≈0.7 | Score≈0.4 |
| --- | --- | --- | --- |
| Avg daily samples | ≥8 | 4–8 | <4 |
| Daily sample CV | ≤0.30 | 0.30–0.70 | >0.70 |
| Missing ratio | ≤5% | 5%–20% | >20% |
| Effective daily points | ≥80 | 30–80 | <30 |
| Top5 day share | ≤45% | 45%–65% | >65% |
| Overall sample size | ≥800 | 400–800 | <400 |
| Bootstrap CI width / |IC| | ≤0.015 | 0.015–0.035 | >0.035 |
| Rolling MAIC | ≥0.05 | 0.02–0.05 | <0.02 |
| Rolling CV | ≤0.50 | 0.50–1.0 | >1.0 |
| Lag1 autocorr | 0.40–0.85 | 0.20–0.40 or 0.85–0.95 | <0.20 or >0.95 |
| Trend corr | |r| ≤ 0.2 | 0.2–0.5 | >0.5 |
| Direction changes | 1–3 次 | 4–6 次 | ≥7 次 |
| Resample success | ≥0.95 | 0.80–0.95 | <0.80 |
| Cross-sample std | ≤0.01 | 0.01–0.02 | >0.02 |

所有阈值放在 `fa_config.RELIABILITY_CONFIG`。另新增 `trade_profile_thresholds`（用来描述交易系统平均交易次数、波动、集中度等）便于未来按策略类型切换。

---

## 5. Implementation Tasks

### 5.1 `fa_nonparam_analysis.py`
1. **Daily metrics capture**:  
   - Extend `calculate_ic` to output `daily_avg_samples`, `daily_median_samples`, `daily_sample_cv`, `daily_missing_ratio`, `daily_effective_points`, `ic_mode`（daily/daily_window/overall）、`window_span_days`, `min_samples_per_day_applied`, `top5_day_share`, `avg_trades_per_day`, `trade_count_cv`.  
   - Record `skipped_reasons`（样本不足/方差为零/异常值），用于 reliability 文案。
2. **Auxiliary stats enrichment**:  
   - Ensure rolling、temporal、sample-sensitivity metrics（MAIC、CV、半衰期、Lag1、trend corr、direction_changes、resample_success、cross_sample_std、success_days_pct）都写入 `auxiliary_stats[factor]`。  
   - Add `overall_ci_width`, `overall_sample_size`, `overall_unique_symbols`.
3. **Reliability payload**:  
   - Build `reliability_inputs = { 'base': {...}, 'overall': {...}, ... }`;  
   - Call helper（§5.2）得到 `reliability_scores` + `reliability_labels` + `final_weights`;  
   - Attach to `analysis_results[factor]` + `auxiliary_stats[factor]`.
4. **CSV writer updates**:  
   - `辅助分析稳健性评分_*.csv` 增加列：  
     `ic_mode, daily_points, reliability_base, reliability_overall, ..., weight_base, weight_overall, ...`.  
   - Include textual note column `reliability_note`.
5. **Logging**:  
   - Logger prints `[INFO] Factor=xxx mode=daily_window avg=2.1 cv=1.2 missing=38% reliability(base=0.42→low) weight(base=0.19 overall=0.34 rolling=0.20 …)`。

### 5.2 `fa_nonparam_helpers.py`
1. **Reliability computation**:  
   - Implement `_score_interval(value, th_high, th_mid, th_low)` returning 0.3–1.0.  
   - Implement `_blend_scores(score_dict)` to combine metrics per dimension.  
   - `_derive_reliability_scores(reliability_inputs, config)` 输出 `{scores, labels}`.
2. **Weight adjustment**:  
   - Modify `compute_integrated_factor_scores` to accept reliability info; produce `final_weights` (list + dict) and `weight_notes`.  
   - Provide `_normalize_weights(weights, clip_min, clip_max)` with fallback to base weights if reliability missing.
3. **Score tightening**:  
   - Update `_score_overall_component`, `_score_ic_component`, `_score_sample_std`, `_score_sample_size_component` thresholds to align with“整体样本多但得分不过满”的要求（例如 `sample_size >= 800 → 90` 而非 100；`std <= 0.01 → 90`，`std <= 0.005 → 100`）。

### 5.3 `fa_nonparam_report.py`
1. **Unified detail sections**: iterate over所有正向因子 + 负向因子，输出同等详尽的段落。  
2. **评分融合表述**：  
   - `评分融合：基础表现（日度IC/IR，权重18%，可靠性=低）55.0 / 整体样本表现（整体IC/IR/P值，权重32%，可靠性=高）66.0 / 滚动稳定度（CV/半衰期，权重20%，可靠性=中）72.0 / … → 综合 64.5 (B级)`  
   - 追加 `实际权重(%)` 与 `reliability labels`。  
3. **可靠性说明段**：  
   - `数据可靠性评估：日均样本 2.1，缺样 35%，Top5 日期占 70%；判定为低 → 建议更多依赖整体样本统计。`  
   - 若 `reliability_base < 0.4`，添加警示语。  
4. **整体表现引用**：在正/负向段落中同步展示“整体 IC/IR/P 值/样本数/方法”避免只有负向段落才显示。

### 5.4 `fa_stat_utils.py`
1. Provide `bounded_normalize(weights, min_ratio, max_ratio)` helper reused by scoring;  
2. Optionally expose reliability helpers if parameterized模块想复用；  
3. Confirm `ensure_list`/`normalize_sequence` 仍然满足 IC 计算鲁棒性需求。

### 5.5 `fa_config.py`
1. Add `RELIABILITY_CONFIG = { thresholds, base_weights, clamp_min, clamp_max, drop_threshold, trade_profile_thresholds }`.  
2. Document entries for quick tuning (`avg_daily_samples_high=8`, etc.).  
3. Reserve profile presets (`'balanced'`, `'robust'`) though CLI flag未开放。

---

## 6. Validation Checklist

1. Run `python yinzifenxi1119_split.py`（完整 & 摘要模式）：  
   - 确认日志出现 reliability 行；  
   - 报告中所有因子都带“评分融合 + 可靠性说明”；  
   - CSV 新增列写入正确、各权重之和=1。
2. `python tools/compare_stage_outputs.py baogao`：  
   - 预期 diff 较多（新行、新列）；记录在 `memory/yinzifenxi1119_progress.md`。  
   - 若 compare 工具难以自动忽略“实际权重”行，可扩展匹配规则或更新 baseline。
3. Spot check：  
   - 低样本因子应显示基础权重大幅下降 + 警示语；  
   - 大样本因子显示整体权重提升但得分不会直接 99/100；  
   - 正向/负向段落均完整呈现。
4. Refresh `baseline_outputs/` after确认报告正确，并附运行命令/时间在进度文件记录。

---

## 7. Future Ideas

- CLI flag `--fixed-score-weights` for regression;  
- 把可靠性指标输出到 BI/Notebook，追踪随时间变化；  
- 同步 reliability engine 至参数化/机器学习模块；  
- 扩展为“交易系统画像”驱动的多模板权重（稳健型/进取型/高频型），让策略配置可定制。
