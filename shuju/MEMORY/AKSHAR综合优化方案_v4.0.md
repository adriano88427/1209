# AKSHAR股本数据访问综合优化方案 v4.0

> 综合吸收 `AKSHAR优化方案.md`（缓存+集中限流）、`Cline_精细化AKSHARE优化方案_v2.0.md`（零重复获取、反爬、监控）、`AKShare股本数据访问优化方案-GLM-4.6-v3.0.md`（请求控制与代理策略）和 `AKShare优化方案-Qwen-3-Coder.md`（缓存组件化、批量预取）的思路，形成一份更细化、可落地的实施方案。

## 1. 优化目标
1. **访问去重**：所有“全市场返回”的接口仅请求一次，其余通过缓存/映射复用。
2. **速率与反爬治理**：统一限流、随机 UA、可选代理、自适应延迟，避免瞬时高频触发封禁。
3. **弹性缓存体系**：内存 + 本地文件 +（可选）长期缓存，支持断点续跑与多次执行共享数据。
4. **稳健重试与恢复**：指数退避、错误分类恢复、被封时自动降速或切换代理。
5. **可监控可扩展**：实时记录请求量/命中率/失败率，分阶段实施，既兼顾快速缓解又保留进阶优化空间。

## 2. 目标架构概览
```
┌───────────────────────────────────────────┐
│                Orchestrator               │
│  - 主流程 (AKSHAR股本数据.py)                │
│  - 负责加载股票清单 / 控制循环 / 输出         │
├──────────────┬───────────────┬───────────────┤
│ AkShareClient│ AntiCrawlMgr │ CacheLayer     │
│ 统一请求入口  │ UA/代理/延迟   │ L1/L2/L3缓存        │
├──────────────┴──────┬────────┴──────────────┤
│ Shared Data Prefetch │ RateLimiter & Monitor │
└──────────────────────┴───────────────────────┘
```

## 3. 核心组件设计
### 3.1 `AkShareClient`（结合四份方案的缓存+限流思想）
- **职责**：包装 akshare 的所有请求，接入统一的 `RateLimiter`、`AntiCrawlManager`、`CacheLayer`。
- **缓存键设计**：
  - `stock_yzxdr_em(date)` → 常量键；
  - `stock_report_fund_hold(hold_type, date)` → `(hold_type,date)`；
  - `stock_individual_info_em(symbol)`、`stock_zh_a_hist(symbol,start,end)`、`stock_gdfx_*` 等按参数组合。
- **请求流程**：
  1. 查 L1/L2/L3 缓存（参考 Cline 的三级缓存 + Qwen 的持久化策略）。
  2. 若缓存未命中 ⇒ 调用 `RateLimiter.wait()` + `AntiCrawlManager.prepare_request()`（设置 UA/代理/Referer）。
  3. 请求失败时进入 `ErrorRecoveryManager.handle(error)`，执行指数退避、降速或代理切换。
  4. 成功后写入缓存，并向 `PerformanceMonitor` 汇报请求耗时与结果。

### 3.2 `RateLimiter`（融合 GLM/Qwen/Cline 的节流）
- 维护最近一分钟的请求时间戳队列（GLM/Qwen思路），确保 `calls_per_minute` 上限；
- 增加“最小间隔 + 抖动”机制（GLM/Cline），并在高压时由 `AntiCrawlManager` 提升延迟；
- 暴露 `set_mode('normal'|'slow')`，在封禁/频率提示时自动切到慢速模式。

### 3.3 `AntiCrawlManager`
- **请求头池**（Cline方案）+ `fake_useragent`，每次请求随机选择并附带常见 Referer；
- **代理池（可选）**：支持轮换或与外部代理 API 对接（GLM/Cline 提及）；
- **自适应延迟**：维护 5 分钟内请求数，超过阈值自动乘以放大系数（Cline）；
- **封禁检测**：HTTP 403/429 或 akshare 抛错信息触发 `ErrorRecoveryManager`。

### 3.4 `CacheLayer`（统一吸收 Qwen/Cline 的分层思路）
- **L1**：运行时内存字典（命中最快）；
- **L2**：当日本地文件缓存（如 `cache_l2_YYYYMMDD/`），适合短期重复执行；
- **L3**：长期缓存（`cache_l3/`），存放不常更新的数据（如上市信息、历史十大股东）；
- **TTL**：按数据类型配置（Cline）：`stock_info` 24h、`fund_hold` 6h、`price_hist` 1h 等；
- **接口**：`get(cache_key, cache_type)` / `set(..., cache_type)`，其中 `cache_key` = `hash(api_name+params)`。

### 3.5 `SharedDataPrefetch`
- 执行顺序按照 Qwen/GLM 的建议：
  1. `stock_yzxdr_em(date)` → `yzxdr_map[stock_code_prefix] = set(名字)`；
  2. `stock_report_fund_hold(hold_type, date)` → `fund_hold_map[hold_type][stock_code] = row`；
  3. 可进一步预取 `stock_gdfx_free_top_10_em`（按日期）及历史十大股东（只要 `OLDDATE` 集合有限则批量获取）；
  4. 如若某些数据依旧按股票调用，也通过 `AkShareClient` 的缓存减少重复。

### 3.6 `ErrorRecoveryManager` + `PerformanceMonitor`
- **错误分类**（Cline方案）：403/429 ⇒ 降速 + 代理轮换；timeout ⇒ 降低并发/延迟；
- **恢复动作**：等待、切代理、刷新 UA、写告警；
- **监控**：统计成功数、失败数、封禁数、平均响应时间、缓存命中率，若 `success_rate < 90%` 或 `block_rate > 5%` 时打印警告（Cline/GLM）。

## 4. 主流程改造要点
1. **初始化阶段**：
   - 读取股票清单 → 规范化代码（补零、前缀）；
   - 构建 `AkShareClient`, `SharedDataPrefetch`, `PerformanceMonitor`；
   - 预取共享数据，构造映射。
2. **股票循环**：
   - 获取流通市值/上市时间 ⇒ 直接从 `stock_info_cache` 读取；若缺失则 `client.stock_individual_info_em`；
   - 行情数据 `stock_zh_a_hist` 仅在缓存 miss 时调用，`get_circ_mv_on_date` 仅接收 `client` 结果；
   - 历史/流通十大股东，先尝试 `client.top10_*` 缓存；`stock_management_change_ths` 同理；
   - 机构持仓直接查 `fund_hold_map`，若没有记录则填“0%/N/A”；
   - 主循环不再包含 `for attempt`，失败由 `client` 抛异常时标记并继续下一个股票。
3. **输出/监控**：
   - 每处理 N 只股票输出 `PerformanceMonitor` 的摘要（成功率/封禁率/平均响应时间/缓存命中率）；
   - 结果数据结构保持不变，确保输出 Excel 与旧版一致。

## 5. 实施路线（结合三阶段理念）
### Phase 1（1-2 天，快速止血）
- 落地 `AkShareClient`、`RateLimiter`、`SharedDataPrefetch`；
- 移除循环内对 `stock_yzxdr_em`、`stock_report_fund_hold` 的重复调用；
- 简单内存缓存 + 每分钟请求上限，验证结果一致性。

### Phase 2（3-5 天，稳健运行）
- 引入 L2/L3 缓存与文件持久化；
- 增加请求头池、可选代理、自适应延迟；
- 接入 `ErrorRecoveryManager` 与 `PerformanceMonitor`，监控成功率、封禁率；
- 在 AkShareClient 中实现指数退避 + 缓存命中日志。

### Phase 3（1-2 周，进阶优化）
- 扩展到异步/批处理（参考 Cline 的 `AsyncDataFetcher` 思路或多线程线程池），在保持限流的前提下提升吞吐；
- 部署更细的监控（Prometheus/Grafana 或日志分析），评估请求耗时与API命中率；
- 若业务需要，加入可配置的代理池/付费代理支持，以及对缓存 TTL 的动态调节；
- 视需求接入报警（success rate、block rate、cache miss）和自动降速策略。

## 6. 预期收益
| 维度 | 优化前 | 优化后预期 |
| --- | --- | --- |
| `stock_yzxdr_em` 调用次数 | O(股票数) | 1 次 + 缓存 |
| `stock_report_fund_hold` | O(股票数×6) | 6 次 + 缓存 |
| `stock_individual_info_em` | 每股 2 次 | 每股 ≤1 次（命中缓存则 0 次） |
| 请求成功率 | 80% 左右 | ≥95%（有监控与恢复） |
| 被封风险 | 高频触发 | 速率受控 + 反爬伪装 + 备援代理 |
| 执行时间 | 长、易超时 | 预计缩短 50%+，且可持续运行 |

## 7. 注意事项
1. **缓存一致性**：定期清理过期文件，保证当日数据及时更新；
2. **代理安全**：若开启代理模式，需确保来源可靠并记录；
3. **合规性**：严格控制请求频率，遵守数据源使用条款；
4. **验证**：每阶段上线前，用小样本对比新旧结果，确认数值不变；
5. **回退计划**：保留原始脚本可配置开关（如 `USE_OPTIMIZED_CLIENT`），遇紧急问题可切换回旧逻辑。

通过整合四个方案的优势，可以在不改变输出结果的前提下，大幅降低对东方财富的请求压力，增强脚本的鲁棒性及可运维性。
