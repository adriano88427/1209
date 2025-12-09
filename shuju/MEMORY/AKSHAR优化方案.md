# AKSHAR股本数据访问优化方案（加强版）

> 备注：尝试访问 https://www.akshare.xyz/data/stock/stock.html 以查阅官方接口说明，但本机网络无法连通（curl 超时）。以下方案结合已知 AKShare 文档与现有代码逻辑整理。

## 1. 当前瓶颈再梳理
| 模块 | 现状 | 影响 |
| --- | --- | --- |
| 一致行动人 `stock_yzxdr_em` | 在股票循环里每只股票重试 3 次；该接口本身返回“指定日期的所有股票” | 请求次数 = 股票数 × 3，易触发东财限流 |
| 六类机构持仓 `stock_report_fund_hold` | 对每只股票、每个 `hold_type` 都重新抓取整张表 | 请求次数 = 股票数 × 6（一次调用约 200ms，放大为几分钟） |
| 上市信息/流通股本 `stock_individual_info_em` | 在 `get_circ_mv_on_date` 与“上市时间”逻辑分别调用 | 同一股票两次命中东方财富接口 |
| 十大股东系列 `stock_gdfx_free_top_10_em` / `stock_gdfx_top_10_em` | 无缓存，失败会重新发起请求 | 高频接口缺乏速率控制 |
| 单请求重试策略 | 各模块各自 `for attempt in range(3)`，没有限速、没有共享缓存 | 瞬时并发高，失败时反复触发限流 |

## 2. 优化目标
1. **共享数据一次拉取**：把“按日期返回全量数据”的接口移出股票循环，只保留本地过滤。
2. **缓存 + 复用**：对按股票查询但可复用的数据统一缓存，杜绝重复打点。
3. **全局节流 + 统一重试**：实现集中式速率限制器，限制对东方财富的每秒请求数，并采用指数退避。
4. **可配置、可监控**：所有阈值写入配置，记录缓存命中率与失败次数，便于后续根据 IP 封锁风险调优。

## 3. 详细方案
### 3.1 建立 `AkShareClient` 封装
```python
class AkShareClient:
    def __init__(self, max_calls_per_minute=60):
        self.lock = threading.Lock()
        self.call_timestamps = collections.deque()
        self.max_calls = max_calls_per_minute
        self.caches = {
            'stock_info': {},
            'fund_hold': {},
            'yzxdr': None,
            'management_change': {},
            'top10_history': {},
            'top10_free': {},
            'price_hist': {}
        }
```
- `request(api_name, loader)`：统一入口，先检查缓存，再做速率限制（保证相邻请求 ≥ 0.3s 或每分钟 ≤ max_calls），失败则指数退避（1s → 2s → 4s），超过 `MAX_RETRIES` 直接返回 `None`。
- 将所有 akshare 函数包装为 `client.stock_yzxdr(date)`、`client.stock_report_fund_hold(hold_type, date)` 等方法，在内部维护缓存。

### 3.2 把“全量接口”移出循环
1. **一致行动人**：
   ```python
   yzxdr_df = client.stock_yzxdr(date)
   yzxdr_map = defaultdict(set)
   for _, row in yzxdr_df.iterrows():
       yzxdr_map[row['股票代码']].update(row['一致行动人'].split(','))
   ```
   主循环中仅做 `insider_names.update(yzxdr_map.get(code_with_prefix, set()))`，完全消除 O(N) 请求。
2. **机构持仓**：对 `hold_types` 预拉一次 `client.stock_report_fund_hold(hold_type, report_date)`，返回 `DataFrame`；构建 `fund_hold_map[hold_type][stock_code] = row`。主循环只读取字典。

### 3.3 合并 `stock_individual_info_em` 与 `stock_zh_a_hist`
- `client.stock_info(stock_code)` 在首次调用时获得包含“上市时间”“流通股”等字段，缓存后供：
  - `get_circ_mv_on_date` 直接使用 `流通股`，避免再次调用；
  - “获取上市时间”逻辑直接从缓存读 `info_dict`，若缺失再触发一次请求。
- `client.price_hist(stock_code, start_date, end_date)` 缓存最近一次行情 DataFrame，若重试或多模块需要同区间，可复用。

### 3.4 十大股东接口缓存
- `client.top10_free(symbol, date)` 与 `client.top10_history(symbol, date)` 使用 `(symbol, date)` 作为键缓存 DataFrame，避免失败后重下。
- 对长时间不再访问的键可设置 LRU（如 `functools.lru_cache(maxsize=500)`）。

### 3.5 全局速率限制 & 批处理
- 设定 `MAX_CALLS_PER_MINUTE`（如 60）和 `BURST_SIZE`（如 5）。`request` 每次调用前检查队列 `call_timestamps`，若在最近 60 秒内已达上限，则 sleep 到最早一次出窗口为止。
- 对必需但量大的接口（十大股东、历史股东等）可额外 `time.sleep(0.2)`，进一步平滑流量。

### 3.6 失败策略
- 避免在主循环里嵌套 `for attempt`：统一由 `client.request` 控制重试次数和退避时间，调用方只判断返回值是否 `None`。
- 对不可或缺的数据（如十大流通股东）在失败时标记为“待补采”并写日志，避免程序直接崩溃。

### 3.7 监控与可配置化
- 增加 `config.py` 或顶部配置块，包含 `MAX_CALLS_PER_MINUTE`, `CACHE_TTL`, `ENABLE_PARALLELISM` 等参数。
- 统计输出：
  - 缓存命中率（每个接口：命中次数 / 请求次数）。
  - 实际请求总数，便于与优化前对比。
  - 失败/重试次数，帮助定位限流点。

## 4. 修改步骤
1. **新增 `akshare_client.py`**：实现上述 `AkShareClient`、缓存和节流逻辑，并在脚本开头实例化。
2. **主脚本调整输入阶段**：
   - `df_stocks` 读取后立即标准化股票代码（去前缀、补零）。
   - 调用 `client.stock_yzxdr(date)`、`client.stock_report_fund_hold` 生成映射字典。
3. **重构 `get_circ_mv_on_date`/上市时间段落**：改为使用 `client.stock_info` 返回的缓存数据；行情接口也通过 `client.price_hist` 访问。
4. **十大股东/高管变动**：调用 `client.management_change(stock_code)`、`client.top10_history(symbol, date)` 等缓存方法，移除旧的 `for attempt`。
5. **添加速率配置与日志**：在脚本入口打印当前 `MAX_CALLS_PER_MINUTE`、缓存命中率，便于确认优化生效。
6. **验证**：以 10、50、100 只股票跑一遍，记录请求数量与运行时间，确保相比原版明显下降并无异常。

## 5. 预期收益
- `stock_yzxdr_em`、`stock_report_fund_hold` 等接口请求次数从“与股票数成正比”下降为“常数级（1~6 次）”。
- `stock_individual_info_em` 从每股 2 次下降为 1 次；若运行多次，可进一步落地磁盘缓存。
- 全局速率限制显著减小瞬时访问峰值，降低东方财富封 IP 风险。
- 统一日志和配置让后续根据封锁策略及时调整调用频次。
