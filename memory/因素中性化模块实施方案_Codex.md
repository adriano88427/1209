# 因素中性化模块实施方案（Codex）

## 1. 需求概述
- **目标**：在单因子、参数化因子及双因子分析流程中加入市值与行业中性化，使报告输出符合中国 A 股市场主流的中性化处理方式。
- **范围**：`fa_config.py`（配置）、新模块 `fa_neutralizer.py`、`fa_nonparam_analysis.py`、`fa_param_analysis.py`。后续 QA/监控功能也围绕该模块展开。
- **数据依赖**：`信号日期`、`流通市值(元)`、`所属同花顺行业`，必须在预处理阶段被正确解析。

## 2. 模块设计
1. **新增文件 `yinzifenxi/fa_neutralizer.py`**
- `FactorNeutralizer`：接收配置与可选 `ExternalLogger`。
   - 核心方法：
     - `_prepare_cache(df)`：基于 `信号日期` 分组，同时生成 log 市值序列和行业序列，缺列则发出 `[NEUTRAL][WARN]`。
     - `_apply_market_cap`：逐日执行 `y ~ α + β*log(mcap)` OLS；若样本数或方差不足则退化成当日去均值。
     - `_apply_industry`：行业内去均值，样本小于阈值时退化为使用当日整体均值。行业字段默认拆分为“前两级（大类-中类）”，如 `医药生物-化学制药`；支持通过配置调整到一级或三级，以确保样本充足且行业相似。
     - `_apply_to_factor`：按照配置模式（`market_cap`/`industry`/`both`/`none`）组合调用，覆盖原列并在 DataFrame 中追加 `<factor>__raw` 备份。
   - **结果追踪**：返回 `NeutralizationStats`（覆盖率、模式、样本数、备注），并在日志中输出 `[NEUTRAL][INFO/WARN/ERROR]`。

2. **配置扩展（`fa_config.py`）**
   - `FACTOR_NEUTRALIZATION_RULES`：为 20 个因子逐一指定模式。
   - `NEUTRALIZATION_CONFIG`：集中开关与阈值（`enabled`、`signal_date_column`、`market_cap_column`、`industry_column`、`industry_level=2` 默认取“前两级行业”、`min_cross_section=8`、`min_industry_group=4`、`store_raw_suffix='__raw'`、`default_method='none'`、`factor_rules`）。特别强调：所有中性化前的预处理步骤沿用现有逻辑，不修改其他模块的行为，以保证中性化关闭时数据完全与源代码一致。
   - **可选增强**（配置灵活度）：允许通过环境变量或命令行覆盖 `factor_rules` 中的部分因子，使研究员可快速试验不同中性化策略；配置异常时在日志中给出回退与提示。

## 3. 与主流程的耦合
- `FactorAnalysis` / `ParameterizedFactorAnalyzer`
  1. 在 `__init__` 中新增 `self.neutralization_summary=[]`。
  2. 在 `preprocess_data` 中，所有因子数值化与 winsorize 完成后调用 `_apply_factor_neutralization`，注入 `FactorNeutralizer` 并保存 summary。
  3. `_apply_factor_neutralization` 接受配置、因子清单以及可选 logger；若配置关闭或无可用因子则直接返回。
  4. 后续的收益率处理、覆盖率校验、报告生成全部基于中性化后的列。
- 通过 `<factor>__raw` 可在导出 Excel/TXT 时保留原值（默认报告仍只展示处理结果）。

## 4. 日志 & QA
- **日志规范**：
  - `[NEUTRAL][INFO] 当日回调 - 模式 both 已完成，覆盖率 93.2% (市值: 15280 行, 行业: 15280 行)`
  - `[NEUTRAL][WARN] 缺少市值列流通市值(元)，市值中性化降级为行业去均值`
  - `[NEUTRAL][ERROR] 因子 当日最高涨幅 中性化失败: <异常>`
- **异常监控增强**：当 `NeutralizationStats.coverage` 低于阈值（例如 0.6），或样本退化次数过多时，自动在日志中添加 `[NEUTRAL][WARN] 因子 X 覆盖率 45%，建议检查数据质量`，并将该信息落盘到 QA CSV，方便后续统一巡检。
- **结果诊断增强**：基于 summary 计算中性化前后与 `log(mcap)`、行业哑变量的相关系数变化、均值/标准差变化，并在调试日志或附加的诊断表中展示。必要时在 HTML 报告里生成“中性化效果对比”小节（例如相关系数热图或散点缩略图），仅在 `--debug`/`--summary-report` 模式启用，控制体积。
- **性能回归验证**：提供一个辅助脚本/命令行参数，用相同数据分别运行“启用”与“禁用”中性化并比较核心指标（IC、IR、胜率等）。若差值超出阈值（例如 IC 变化>0.05），触发 `[QA][WARN]` 提示，帮助快速定位因子被异常放大的情况。

## 5. 实施步骤
1. **配置阶段**：
   - 在 `fa_config.py` 增加 `FACTOR_NEUTRALIZATION_RULES` 与 `NEUTRALIZATION_CONFIG`。
   - 针对每个因子确认模式，并预留 `default_method` 与可覆盖入口。
2. **新模块开发**：
   - 编写 `fa_neutralizer.py`，包含所有回归、行业去均值、日志、统计逻辑。
3. **主流程接入**：
   - 修改 `fa_nonparam_analysis.py` 和 `fa_param_analysis.py` 调用中性化流程。
4. **QA & 日志**：
   - 新增 summary 输出与异常监控；必要时扩展 `ExternalLogger` 以支持 `[NEUTRAL]` 级别。
5. **手工验证**：
   - 使用 2025 样例数据运行一次完整流程，检查 `[NEUTRAL]` 日志、`__raw` 列是否存在、覆盖率是否合理。
   - 可加一份对比脚本（启用/禁用中性化）确认核心指标的差异。

## 6. 风险与缓解
- **缺列风险**：通过 WARN 日志和退化处理保证流程继续执行；同时在 QA 阶段提示因子覆盖率。
- **性能**：基于 groupby 的 OLS 计算量较大，但属于单次 per factor per day OLS，可接受；如后续发现瓶颈，可采用矩阵化或缓存残差策略。
- **配置风险**：提供默认 `none`，并在配置异常时回退，防止因 mis-configuration 导致因子无输出。
- **一致性**：中性化后覆盖原列，因此报告与下游逻辑无需改动；若需回滚，只需关闭配置开关。

## 7. 后续优化路线（按优先级）
1. **结果诊断可视化**：完成基础功能后，将诊断图/表嵌入报告或导出独立 HTML，辅助研究员快速判断效果。
2. **异常监控报表**：把覆盖率、退化情况输出到 `baogao/QA_*` CSV，配合现有日志实现自动巡检。
3. **性能回归验证脚本**：制作 `tools/check_neutralization_effect.py`（示例），对关键因子进行启停对比并生成差异报告。
4. **配置灵活度增强**：允许研究员通过环境变量或 CLI 指定单个因子模式，方便试验。
5. **批量 OLS/向量化优化**：在因子数量增多或性能成为瓶颈时优先实现。
6. **动态样本阈值/行业映射/缓存残差**：视后续需求逐步引入，保持模块简洁可维护。
