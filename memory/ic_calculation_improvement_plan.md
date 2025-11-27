# IC 计算鲁棒性优化方案（细化版）

## 关键痛点复盘
1. **daily_ics 类型安全不足**：a_stat_utils.ensure_list 遇到非 list 直接返回空列表；a_nonparam_analysis.calculate_ic 中也多次调用 ensure_list，导致数据被清空，引发 len() of unsized object。
2. **低样本日处理粗糙**：只要某日 len(valid_data) < min_samples_per_day 就跳过，导致大量有效信息丢失，最终只能 fallback 到整体 IC。
3. **阈值与日志不一致**：min_samples_per_day 仅根据平均样本数切换 3 档，实际样本分布、滑动窗口、日志输出之间缺乏联动，也无法在报告中说明“低样本模式”。

## 具体优化项
### A. 工具层（fa_stat_utils.py）
1. **ensure_list 重写**
   - 若对象已是 list，原样返回。
   - 若为 tuple/pd.Series/np.ndarray，eturn list(obj)。
   - 若为标量（int/float/np.number），包装成 [value]。
   - 保留原数据引用并打印一次警告，防止 silently reset。
2. **新增 
ormalize_sequence(obj, allow_scalar=True)**
   - 集中处理 numpy 标量、pandas Series、空值等场景，在 calculate_ic 及其他需要 list 的地方调用。
3. **safe_len 输出上下文**
   - 当类型不支持 len() 时记录调用栈/对象内容，方便后续排查。

### B. 日度 IC 计算链条（fa_nonparam_analysis.calculate_ic）
1. **逐日循环的异常隔离**
   - 将 or date, group in df.groupby('信号日期') 中每次迭代包裹 	ry/except，一旦单日出错仅记录该日，不再 reak 整段。
2. **动态阈值与窗口**
   - 依据 vg_daily_samples / median_daily_samples / p25/p75 选择：
     | 场景 | 平均样本 | min_samples_per_day | 窗口 | 说明 |
     | --- | --- | --- | --- | --- |
     | 高频 | ≥8 | 5 | 1 天 | 保持当前逻辑 |
     | 中频 | 4-8 | 3 | 2 天滑动合并 | 两天样本合一，减少 ±1 |
     | 低频 | <4 | 2 | 3 天滑动或“事件周” | 输出频率降到周级 |
   - 当窗口合并时，日志记录“使用 3 天滑动窗口，起点 xxx”。
3. **稳健统计**
   - 对每个日度 IC 进行 
p.clip(ic, -0.99, 0.99) 或 winsorize，防止因样本极少导致 t-stat 爆炸。
   - 可选：对 daily_ics 使用 statsmodels.robust.scale.mad 来衡量波动，而不仅依赖标准差。
4. **整体回退条件**
   - 仅当 alid_daily_ics < min_overall_points（默认 10）时才 fallback，fallback 时输出 overall_ic、daily_points、skipped_dates 等信息。
   - 若 fallback，仍给 extra_stats['daily_points']=n，供报告使用。
5. **缺样原因统计**
   - self.anomaly_stats['ic_calculation'][factor]['skipped_reasons'] 记录 {'date': xxx, 'reason': '样本不足/方差为零/异常值'}。

### C. 报告与摘要联动
1. **日志增强**
   - 输出：平均样本、窗口策略、有效日度 IC 数、缺样日比例、是否 fallback。
2. **报告展示**
   - 在正/负向摘要中加入：“日度 IC 有效点数 42/398，min_samples=3，窗口=2 天；缺样日 64（样本不足）”。
   - 如果 fallback，注明“因子处于低样本模式，报告中的 IC 指标基于整体样本计算”。
3. **辅助 CSV 扩列**
   - 辅助分析稳健性评分_*.csv 增加 daily_ic_points, skip_ratio, ic_mode（e.g. daily, daily_window, overall）。

### D. 验证策略
1. **单元测试**：构造多种日度样本（全为空、单样本、滑动窗口）验证 calculate_ic 不再抛出 len() 异常，返回结构完整。
2. **集成测试**：默认/摘要模式各 run 一次，确认日志中 len() of unsized object 消失，报告/CSV 新字段正确。
3. **回归**：更新 aseline_outputs/ 并 python tools/compare_stage_outputs.py baogao，若新增字段需要 compare 调整过滤逻辑（忽略“日报模式”提示行）。

## 产出与后续
- 代码：a_stat_utils.py、a_nonparam_analysis.py、a_nonparam_report.py 的更新。
- 文档：memory/yinzifenxi1119_progress.md 增加“IC 计算鲁棒性”阶段记录。
- 数据：新版 因子分析日志_*、辅助分析稳健性评分_*（含日度 IC 统计），以及刷新后的 aseline_outputs/。

上述改动完成后，逐日 IC 将能在低样本场景下稳定计算，日志与报告也能清楚说明计算模式，避免误把 fallback 结果当成日度表现。
