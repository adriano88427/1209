# Code Modification Plan – Integrated Auxiliary Analysis

## 1. Objectives
- 取消独立的 `辅助分析报告_*.txt`，将辅助分析内容合并到 `因子分析详情_精简版_*.txt`。
- 让因子评级/综合得分在评估基础表现的同时，纳入动态稳健性指标（滚动 IC、时序一致性、样本敏感性）。
- 保留结构化的辅助数据（CSV 摘要）供外部 notebook/可视化使用。

## 2. Scope
| 模块/文件 | 变更内容 |
| --- | --- |
| `daima/fa_nonparam_analysis.py` | 将 `generate_auxiliary_analysis_report()` 改为“计算 + 结构化返回 + CSV 摘要”，不再写独立 TXT；把结果存入 `self.auxiliary_stats` 供报告和评分调用。 |
| `daima/fa_nonparam_report.py` | 在 `_fa_generate_factor_analysis_report()` 中获取 `auxiliary_stats`，在每个因子段落中插入“稳健性分析”小节，同时根据新评分格式输出综合评价与建议。 |
| `daima/fa_nonparam_helpers.py` 或新增 helper | 承载评分与权重逻辑：引入 `rolling_score / temporal_score / sample_score` 子项，并提供 `compute_final_factor_score()` 等函数。 |
| `tools/compare_stage_outputs.py` | 不需要调整（辅助 TXT 已取消），但可继续使用来校验 CSV/带参数 TXT；如需校验精简报告，可扩展过滤规则。 |

## 3. Detailed Steps
1. **辅助分析计算与缓存**
   - 修改 `generate_auxiliary_analysis_report()` 逻辑：  
     - 返回结构化 dict `{factor: {rolling: {...}, temporal: {...}, sample: {...}}}`。  
     - 写 CSV 摘要仍保留，路径仍在 `baogao/`。  
     - 将 dict 挂到 `self.auxiliary_stats`（若无，则创建）。  
   - 主流程不再打印“生成辅助分析报告…”，改为“辅助分析数据已整合至主报告”。

2. **评分融合（增强版）**
   - **维度拆分**  
     - “基础表现”（IC、IR、t、胜率、分组收益）保留为主维度。  
     - 新增“动态稳健性”维度，包含 3 个子项：  
       1. **滚动稳定度**：窗口 CV、IC 半衰期、均值绝对 IC（MAIC）。  
       2. **时序一致性**：lag1 自相关、趋势相关、换向次数、均值回归度。  
       3. **样本稳健性**：各抽样比例的 IC 均值/标准差/IQR 及成功率。  
   - **子项评分（0–100）**  
     - `base_score`：沿用现有映射，如 IC≥0.03 → 90、IC≈0.015 → 60，再由 IR/胜率等补充加权。  
     - `rolling_score`：  
       - CV ≤0.3 → 100，0.3–0.6 之间线性递减，>0.8 → 20。  
       - 半衰期 ≥15 窗口 +20 分，5–15 线性递减。  
       - MAIC ≥0.05 → 100，0.02 → 60。对三个指标做加权平均。  
     - `temporal_score`：  
       - lag1 自相关 0.4–0.8 得高分（>0.9 视为过拟合需扣分）。  
       - 趋势相关 |r|<0.2 加分，换向次数在 1–3 之间最佳，均值回归度越高越加分。  
     - `sample_score`：  
       - 各样本比例的 std <0.01 → 100，0.02 → 70，>0.04 → 30。  
       - 有效迭代成功率 ≥0.95 再加 10 分；IC 均值跨样本波动 ≤0.01 再加分。  
   - **综合权重**  
     - `final_score = base_score * 0.6 + rolling_score * 0.15 + temporal_score * 0.125 + sample_score * 0.125`。  
     - 可根据“稳健/进取”策略调整权重（如稳健型把动态权重提升到 50%）。  
   - **评级映射与文案**  
     - A+≥85，A 75–85，B+ 65–75，B 55–65，C<55。  
     - 文案示例：“基础表现得分 78 / 动态稳健性 82 → 综合 80（A级）；因滚动稳定度高、样本波动低，建议权重 20%。”  
     - 负向因子同理，文本中强调“反向 IC 绝对值 + 稳定性”如何影响对冲价值。  
   - **实现要点**  
     - 在 `generate_auxiliary_analysis_report()` 之后，使用 helper 函数计算 `rolling_score / temporal_score / sample_score` 并存入 `self.auxiliary_stats[factor]['scores']`。  
     - `_fa_generate_factor_analysis_report()` 先展示基础统计，再展示“稳健性子项得分”，最后输出综合得分/评级/权重建议。  
     - 可在 CSV 摘要里附加子项得分列，便于筛选稳定且收益佳的因子。

3. **报告合并**
   - `_fa_generate_factor_analysis_report()`：  
     - 在每个因子段落中插入“稳健性分析”小节，展示滚动窗口统计、半衰期、时序指标、样本敏感性结果。  
     - 文案模板：  
       ```
       稳健性分析：
         • 滚动IC：30D 均值0.02 (CV0.35, 半衰期12)，60D 均值...
         • 时序一致性：自相关0.6，趋势相关0.1，换向2次...
         • 样本敏感性：80% 0.06±0.016，成功率100%...
       综合评价：基础得分 78，稳健得分 82 → 综合得分 80（A级），建议权重 20%-25%。
       ```
   - 删除“辅助报告已生成”相关输出，确保用户只看到主报告路径。

4. **数据结构扩展**
   - `analysis_results[factor]` 中可新增 `auxiliary_scores` 字段，方便后续 CSV 导出或 API 调用。
   - 更新 CSV 摘要（若需要），在 `summary_df` 中附加 `rolling_score / temporal_score / sample_score / final_score` 列。

5. **验证**
   - 运行 `python yinzifenxi1119_split.py`，确认 `baogao/因子分析详情_精简版_*` 包含新内容，未再生成 `辅助分析报告_*` TXT。  
   - `辅助分析报告摘要_*` CSV 仍生成，供后续使用。  
   - `python tools/compare_stage_outputs.py <stage_dir>` 校验 CSV 与带参数报告，记录新的 stage 结果，并在 `memory/yinzifenxi1119_progress.md` 添加阶段说明。

## 4. Risks & Mitigations
| 风险 | 缓解措施 |
| --- | --- |
| 文本合并后报告体积增大，阅读成本上升 | 采用折叠式标题或子小节，保持段落层次清晰；必要时提供“仅评分摘要”选项。 |
| 辅助评分权重与旧评分差异导致历史比较困难 | 在报告中说明“新增稳健性评分模块”并保留 `base_score` 明细，便于用户对比。 |
| 缺失 `baseline_outputs/` 导致 compare 工具无法运行 | 事先恢复或重建 baseline；或临时指定 `--baseline` 为新的参考目录。 |

## 5. Deliverables
- 更新后的 `fa_nonparam_analysis.py` / `fa_nonparam_report.py` / helper 模块代码。
- 新版 `因子分析详情_精简版_*` 报告（包含辅助统计）。
- `辅助分析报告摘要_*` CSV（保持结构，可扩充列）。  
- 进度文件 `memory/yinzifenxi1119_progress.md` 新增“Phase 4.9” 记录。

## 6. 后续扩展建议
1. **精简/摘要模式**：在 `_fa_generate_factor_analysis_report()` 中追加“摘要模式”渲染选项，仅保留评分融合 + 核心稳健性指标（例如每因子 3–4 行），以 CLI 参数或配置开关控制，满足快速阅览或移动端查看需求。
2. **稳健性 CSV 导出**：在 `generate_auxiliary_analysis_report()` 返回的 `summary_df` 上扩列（或新增 `辅助分析报告摘要_*_评分.csv`），输出 `base_score / rolling_score / temporal_score / sample_score / final_score` 以及关键滚动/样本指标，方便 notebook/BI 对稳定性维度做二次筛选。
3. **compare 工具拓展**：若启用摘要模式或新增 CSV，可更新 `TXT_TIMESTAMP_PREFIXES`/匹配规则，将 `因子分析详情_精简版` 或新的摘要文件也纳入自动校验，确保报告文本演进仍能被回归检测覆盖。

## 7. 执行进展（2025-11-27 15:20 更新）
- 按照第 6 条建议完成 compare 工具扩展：`tools/compare_stage_outputs.py` 现可识别 `因子分析详情_精简版` 的“完整版/摘要版”，在计算哈希前自动剔除 `生成时间:` 行，并缓存 variant 区分结果，避免不同模式混比。日志文件因包含大量带时间戳的路径提示而被排除在自动对比范围之外。
- `baseline_outputs/` 已清空并用当前 `baogao/` 快照重建，后续用 `python yinzifenxi1119_split.py` + `python yinzifenxi1119_split.py --summary-report` 重跑一次，生成 15:19 批次输出，再以 `python tools/compare_stage_outputs.py baogao` 验证所有 CSV/TXT/辅助统计均与基线一致（忽略时间戳后哈希）。这套流程将作为今后每轮重构/修复的标准验证步骤。
