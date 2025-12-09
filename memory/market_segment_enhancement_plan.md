# Market Segment Enhancement Plan

## 1. Objectives
1. Embed board/segment awareness (创业板、主板、科创板等) into every stage of the factor workflow so that score reliability reflects the market regime differences (20cm vs 10cm limits, volatility, liquidity).
2. Provide segment-specific statistics (IC、IR、样本量、胜率) and warnings when samples concentrate in a single board.
3. Surface these insights transparently inside the TXT/CSV reports so portfolio users understand where a factor is effective.

---

## 2. Data Preparation & Detection
1. **Segment rules（固定映射，无需配置文件）**  
   - `SZ300/301` → 创业板；`SZ000/001/002/003` → 深圳主板；  
     `SH600/601/603/605` → 沪主板；`SH688` → 科创板；  
     `BJ` 开头或 `830~839` → 北交所；其它统一归 “其他/未知”。  
2. **Preprocessing hook** (`FactorAnalysis._attach_market_segment`)  
   - 新增工具函数根据证券代码写入 `processed_data['market_segment']`；  
   - 统计 `segment_counts`、`segment_ratio` 存入 `self.segment_overview`，并在 `analysis_results[factor]['extra_stats']['segment_counts']` 挂载。

---

## 3. Segment-level Calculations
### 3.1 IC Calculation (`calculate_ic`)
1. After computing overall IC，新增 `segment_metrics = {}`，遍历 segments：
   - `df_seg = df[df.market_segment == seg]`；若总样本 ≥30 且日均≥2，则运行 `_compute_segment_ic(df_seg)`，返回 `daily_points/avg_samples/overall_ic/overall_ir/p_value`；  
   - 记入 `extra_stats['segment_metrics'][seg]`；否则 `segment_warning.append(f"{seg} 样本不足（{len(df_seg)}）")` 并存储占比。  
2. `extra_stats` 还需新增 `segment_primary`（样本最大板块）、`segment_primary_ratio`（%）、`segment_secondary_ic` 等字段，便于 CSV 和报告直接引用。

### 3.2 分组收益 (`calculate_group_returns`)
1. 默认仍执行全样本 10 等分。完成后新增一个摘要 `segment_summary = {seg: {'avg_return': …, 'win_rate': …, 'count': …}}`；  
2. 若未来需要按板块单独排序，只需复用 `segment_summary` 中的数据；当前阶段仅展示统计，不改变主流程排序。  
3. `group_results` 写入 `segment_summary`、`segment_recommendation`（若主板收益为0，则提示“限制于创业板”）。

---

## 4. Reliability & Scoring Adjustments (`fa_nonparam_helpers.py`)
1. `_compute_base_reliability` 加入 `segment_concentration` 与 `segment_coverage`（有有效 IC 的板块数量/总板块数）。可靠性映射建议：  
   - 集中度 ≤60% → 1.0；60~80% → 0.75；≥80% → 0.45；  
   - 若有效板块 ≤1（即只有一个板块算出 IC），再乘以 0.85 作为惩罚。  
2. 若 `segment_metrics` 表明“主板 IC ≈ 0、创业板 IC 明显”，在 `weight_notes` 中明确说明权重被下调。  
3. `segment_adjusted_base_score` 采用 `sum(segment_ic_score * segment_ratio)` 作为基础得分的补充，遇到单板块样本时仍可 fallback 到原 score。

---

## 5. Reporting Enhancements (`fa_nonparam_report.py`)
1. **板块覆盖段落**  
   - 格式：“板块覆盖：创业板 420 条(68%，IC 0.04)，主板 200 条(32%，IC 0.01)，科创板 0 条；提示：主板样本不足”。  
2. **权重说明**  
   - 若 `segment_concentration > 0.8`，在 `评分权重调整` 后附加一句：“因子样本高度集中在创业板，基础权重受限”。  
3. **策略建议**  
   - 当 `segment_summary` 显示收益差异 <0，则输出 “建议限定在主板/创业板使用”。  
4. **CSV**  
   - `因子分析汇总`: 新增 `segment_primary`, `segment_primary_ratio`, `segment_warning`；  
   - `辅助分析稳健性评分`: 新增 `segment_primary`, `segment_primary_ic`, `segment_secondary_ic`；  
   - 摘要 CSV 在表头加入 `segment_*` 字段供 compare 工具识别。

---

## 6. CLI & Config
1. 无需新增配置文件；在 `fa_config.py` 内提供常量 `SEGMENT_MIN_SAMPLES = 30`、`SEGMENT_MIN_DAILY = 2` 即可。  
2. CLI 默认启用板块统计，仅在日志中增加 `[INFO] Segment stats: ...`，暂不暴露开关以减少复杂度；若未来需要可再加 `--disable-segment-analysis`。

---

## 7. Validation Plan
1. Unit tests / synthetic data verifying segment detection & fallback logic (low sample segments produce warnings but no crash).
2. Integration: run `python yinzifenxi1119_split.py` full + 摘要模式; confirm:
   - 报告出现“板块覆盖”段落。
   - `weight_notes` 提到板块集中度。
   - CSV 新列含 segment 信息。
3. Regression: `python tools/compare_stage_outputs.py baogao` 将出现新列导致的 diff，评估无误后更新 `baseline_outputs/` 或调整 compare 忽略规则。
