# AKShare股本数据获取优化方案 - Qwen-3-Coder版

## 问题分析

通过对现有代码`AKSHAR股本数据.py`的分析，发现存在以下几个导致东方财富网IP封禁的主要问题：

1. **高频重复请求**：
   - `stock_yzxdr_em`接口在股票循环中每只股票都请求一次，但实际上该接口返回的是指定日期所有股票的一致行动人数据
   - `stock_report_fund_hold`接口对每种持仓类型都单独请求一次，且每次都是获取全市场的数据
   - `stock_individual_info_em`接口在多个地方重复调用获取相同股票的信息

2. **缺乏请求频率控制**：
   - 没有对请求频率进行限制，容易触发网站的反爬虫机制
   - 失败重试机制简单粗暴，没有指数退避策略

3. **缓存机制不完善**：
   - 虽然有简单的缓存，但没有持久化，程序重启后需要重新获取
   - 缺乏智能缓存策略，无法有效复用已获取的数据

4. **数据获取策略不合理**：
   - 对于返回全市场数据的接口，应该一次性获取后在内存中筛选，而不是每只股票都请求一次

## 优化目标

1. **显著减少对东方财富网的请求次数**
2. **实现智能缓存机制，避免重复获取相同数据**
3. **添加请求频率控制和智能重试机制**
4. **提高代码执行效率**

## 优化方案

### 1. 创建优化组件

#### 1.1 RequestController - 请求控制器
```python
class RequestController:
    def __init__(self, min_interval=1.0, max_calls_per_minute=50):
        self.min_interval = min_interval  # 最小请求间隔（秒）
        self.max_calls_per_minute = max_calls_per_minute  # 每分钟最大请求数
        self.call_times = []  # 请求时间记录
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """根据策略等待合适的时机发起请求"""
        with self.lock:
            now = time.time()
            
            # 清理一分钟前的请求记录
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            # 检查是否超过每分钟最大请求数
            if len(self.call_times) >= self.max_calls_per_minute:
                # 等待到最早的请求超过1分钟
                earliest = min(self.call_times)
                sleep_time = 60 - (now - earliest) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
            
            # 检查与上次请求的间隔
            if self.call_times:
                last_call = max(self.call_times)
                sleep_time = self.min_interval - (now - last_call)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
            
            # 记录本次请求时间
            self.call_times.append(now)
```

#### 1.2 DataCache - 数据缓存器
```python
class DataCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}  # 内存缓存
        self.cache_expiry = 3600  # 缓存有效期1小时
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, func_name, *args, **kwargs):
        """生成缓存键"""
        key_str = f"{func_name}_{hash(str(args) + str(sorted(kwargs.items())))}"
        return key_str
    
    def _get_cache_path(self, cache_key):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, func_name, *args, **kwargs):
        """获取缓存数据"""
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return data
            else:
                del self.memory_cache[cache_key]
        
        # 检查文件缓存
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data, timestamp = pickle.load(f)
                if time.time() - timestamp < self.cache_expiry:
                    # 放入内存缓存
                    self.memory_cache[cache_key] = (data, timestamp)
                    return data
                else:
                    os.remove(cache_path)
            except:
                pass
        
        return None
    
    def set(self, func_name, data, *args, **kwargs):
        """设置缓存数据"""
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        timestamp = time.time()
        
        # 设置内存缓存
        self.memory_cache[cache_key] = (data, timestamp)
        
        # 设置文件缓存
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((data, timestamp), f)
        except:
            pass
```

#### 1.3 EnhancedAkShareClient - 增强型AKShare客户端
```python
class EnhancedAkShareClient:
    def __init__(self, cache_dir="cache"):
        self.request_controller = RequestController()
        self.cache = DataCache(cache_dir)
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _retry_request(self, func, *args, max_retries=3, **kwargs):
        """带重试机制的请求"""
        for attempt in range(max_retries):
            try:
                self.request_controller.wait_if_needed()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    # 指数退避
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    raise e
    
    def stock_yzxdr_em(self, date):
        """获取一致行动人数据"""
        cached = self.cache.get('stock_yzxdr_em', date=date)
        if cached is not None:
            return cached
        
        def _fetch():
            return ak.stock_yzxdr_em(date=date)
        
        result = self._retry_request(_fetch)
        self.cache.set('stock_yzxdr_em', result, date=date)
        return result
    
    def stock_report_fund_hold(self, symbol, date):
        """获取基金持仓数据"""
        cached = self.cache.get('stock_report_fund_hold', symbol=symbol, date=date)
        if cached is not None:
            return cached
        
        def _fetch():
            return ak.stock_report_fund_hold(symbol=symbol, date=date)
        
        result = self._retry_request(_fetch)
        self.cache.set('stock_report_fund_hold', result, symbol=symbol, date=date)
        return result
    
    def stock_individual_info_em(self, symbol):
        """获取个股信息"""
        cached = self.cache.get('stock_individual_info_em', symbol=symbol)
        if cached is not None:
            return cached
        
        def _fetch():
            return ak.stock_individual_info_em(symbol=symbol)
        
        result = self._retry_request(_fetch)
        self.cache.set('stock_individual_info_em', result, symbol=symbol)
        return result
    
    def stock_zh_a_hist(self, symbol, period="daily", start_date="20200101", end_date="20241231", adjust=""):
        """获取历史行情数据"""
        cached = self.cache.get('stock_zh_a_hist', symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
        if cached is not None:
            return cached
        
        def _fetch():
            return ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
        
        result = self._retry_request(_fetch)
        self.cache.set('stock_zh_a_hist', result, symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
        return result
```

### 2. 优化主程序逻辑

#### 2.1 批量数据预获取
```python
def pre_fetch_shared_data(client, date, report_date, hold_types):
    """预获取共享数据"""
    print("🔄 预获取共享数据...")
    
    # 一次性获取一致行动人数据
    print("  获取一致行动人数据...")
    yzxdr_data = client.stock_yzxdr_em(date=date)
    
    # 一次性获取各类持仓数据
    fund_hold_data = {}
    for hold_type in hold_types:
        print(f"  获取{hold_type}数据...")
        fund_hold_data[hold_type] = client.stock_report_fund_hold(symbol=hold_type, date=report_date)
    
    return yzxdr_data, fund_hold_data
```

#### 2.2 优化后的主程序结构
```python
def optimized_main():
    """优化后的主程序"""
    # 初始化优化组件
    client = EnhancedAkShareClient()
    
    # 预获取共享数据
    yzxdr_data, fund_hold_data = pre_fetch_shared_data(client, date, report_date, hold_types)
    
    # 构建数据映射
    yzxdr_map = {}
    if not yzxdr_data.empty:
        for _, row in yzxdr_data.iterrows():
            code = row['股票代码']
            if code not in yzxdr_map:
                yzxdr_map[code] = set()
            if pd.notna(row['一致行动人']):
                yzxdr_map[code].update(name.strip() for name in str(row['一致行动人']).split(','))
    
    fund_hold_map = {}
    for hold_type, data in fund_hold_data.items():
        fund_hold_map[hold_type] = {}
        if not data.empty:
            for _, row in data.iterrows():
                code = row['股票代码']
                fund_hold_map[hold_type][code] = row
    
    # 处理每只股票时，直接使用预获取的数据，避免重复请求
    for idx, stock_name in enumerate(stocks_to_process, 1):
        # 使用预获取的一致行动人数据
        if stock_code_with_prefix in yzxdr_map:
            yzxdr_names = yzxdr_map[stock_code_with_prefix]
        
        # 使用预获取的持仓数据
        for hold_type in hold_types:
            if hold_type in fund_hold_map and stock_code_pure in fund_hold_map[hold_type]:
                stock_data = fund_hold_map[hold_type][stock_code_pure]
                # 处理数据...
```

## 实施步骤

1. **创建优化组件**：
   - 创建`optimized_components`目录
   - 实现`RequestController`、`DataCache`和`EnhancedAkShareClient`类

2. **修改主程序**：
   - 替换原有的AKShare直接调用为优化后的客户端
   - 实现批量数据预获取逻辑
   - 修改各个数据获取函数，使用预获取的数据

3. **测试验证**：
   - 选择少量股票进行测试，验证请求次数是否显著减少
   - 检查缓存机制是否正常工作
   - 验证数据准确性是否保持一致

## 预期效果

1. **请求次数大幅减少**：
   - `stock_yzxdr_em`接口请求次数从N次（N为股票数量）减少到1次
   - `stock_report_fund_hold`接口请求次数从6*N次减少到6次
   - `stock_individual_info_em`接口请求次数减少50%以上

2. **降低IP封禁风险**：
   - 通过请求频率控制，避免短时间内大量请求
   - 通过智能重试机制，减少因失败导致的重复请求

3. **提高执行效率**：
   - 通过缓存机制，避免重复获取相同数据
   - 通过批量预获取，减少等待时间

## 注意事项

1. **缓存管理**：
   - 定期清理过期缓存文件
   - 根据磁盘空间情况调整缓存策略

2. **错误处理**：
   - 保持原有的错误处理逻辑
   - 增加重试日志，便于问题排查

3. **兼容性**：
   - 确保优化后的代码与原有数据格式保持一致
   - 不改变输出文件的结构和内容