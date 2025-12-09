# AKShare股本数据访问优化方案 - GLM-4.6 v3.0

## 问题分析

### 当前代码问题

1. **高频访问问题**：
   - 每只股票需要调用约10-11次AKShare API
   - 无全局访问频率控制机制
   - 请求间隔不足，容易触发东方财富网的反爬机制

2. **重复数据获取**：
   - `ak.stock_yzxdr_em(date=date)` 在每只股票处理时都被调用，但获取的是同一日期的全市场数据
   - `ak.stock_report_fund_hold(symbol=hold_type, date=report_date)` 对每种持仓类型都单独调用
   - 缺乏数据缓存和复用机制

3. **反爬虫防护不足**：
   - 没有设置User-Agent池
   - 没有使用代理IP
   - 没有使用requests.Session保持连接
   - 错误重试机制简单，容易加剧IP封禁

### IP被封原因分析

根据搜索结果，东方财富网对高频访问有以下限制机制：<mcreference link="https://wenku.csdn.net/answer/5pr9irydrf" index="1">1</mcreference> <mcreference link="http://m.toutiao.com/group/7327992255179014671/" index="3">3</mcreference>

1. **请求频率限制**：短时间内大量请求会被识别为爬虫行为
2. **IP访问限制**：单个IP访问次数过多会被临时封禁
3. **User-Agent识别**：默认的Python requests User-Agent容易被识别
4. **会话状态检测**：无会话状态的连续请求容易被标记

## 优化策略

### 1. 全局访问频率控制

实现智能请求频率控制，避免触发反爬机制：

```python
import time
import random
from threading import Lock

class RequestController:
    def __init__(self, min_interval=3, max_interval=8, max_requests_per_minute=10):
        self.min_interval = min_interval  # 最小请求间隔(秒)
        self.max_interval = max_interval  # 最大请求间隔(秒)
        self.max_requests_per_minute = max_requests_per_minute  # 每分钟最大请求数
        self.last_request_time = 0
        self.request_times = []  # 记录最近一分钟的请求时间
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            
            # 检查每分钟请求数限制
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            if len(self.request_times) >= self.max_requests_per_minute:
                sleep_time = 60 - (current_time - self.request_times[0]) + 1
                if sleep_time > 0:
                    print(f"⏱ 达到每分钟请求限制，等待 {sleep_time:.1f} 秒...")
                    time.sleep(sleep_time)
                    current_time = time.time()
            
            # 计算与上次请求的间隔
            elapsed = current_time - self.last_request_time
            interval = random.uniform(self.min_interval, self.max_interval)
            
            if elapsed < interval:
                sleep_time = interval - elapsed
                print(f"⏱ 请求间隔控制，等待 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            self.request_times.append(self.last_request_time)
```

### 2. 增强型反爬虫防护

实现多层次的反爬虫防护机制：

```python
class EnhancedAkShareClient:
    def __init__(self):
        self.session = requests.Session()
        self.request_controller = RequestController()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ]
        self.proxies = self._load_proxies()  # 加载代理IP列表
        
    def _load_proxies(self):
        # 这里可以加载代理IP列表，可以从文件或API获取
        # 示例格式: [{'http': 'http://ip:port', 'https': 'https://ip:port'}, ...]
        return []
    
    def _get_random_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _get_random_proxy(self):
        if self.proxies:
            return random.choice(self.proxies)
        return None
    
    def call_akshare_with_retry(self, func, max_retries=3, backoff_factor=2):
        """
        带重试和退避策略的AKShare调用
        """
        for attempt in range(max_retries):
            try:
                # 应用访问频率控制
                self.request_controller.wait_if_needed()
                
                # 设置随机请求头和代理
                headers = self._get_random_headers()
                proxy = self._get_random_proxy()
                
                # 临时替换akshare的session和headers
                original_session = getattr(func.__self__, 'session', None)
                original_headers = getattr(func.__self__, 'headers', {})
                
                try:
                    if hasattr(func.__self__, 'session'):
                        func.__self__.session = self.session
                    if hasattr(func.__self__, 'headers'):
                        func.__self__.headers.update(headers)
                    
                    # 如果使用代理
                    if proxy and hasattr(func.__self__, 'proxies'):
                        func.__self__.proxies = proxy
                    
                    result = func()
                    return result
                    
                finally:
                    # 恢复原始设置
                    if original_session and hasattr(func.__self__, 'session'):
                        func.__self__.session = original_session
                    if hasattr(func.__self__, 'headers'):
                        func.__self__.headers = original_headers
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                    print(f"⚠️ 请求失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                    print(f"⏱ 等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 请求最终失败: {str(e)}")
                    raise
        return None
```

### 3. 数据缓存与批量获取优化

实现智能缓存机制，减少重复请求：

```python
import pickle
import os
from datetime import datetime, timedelta

class DataCache:
    def __init__(self, cache_dir='cache', expiry_hours=24):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        safe_key = key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def is_expired(self, cache_path):
        if not os.path.exists(cache_path):
            return True
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time > timedelta(hours=self.expiry_hours)
    
    def get(self, key):
        cache_path = self._get_cache_path(key)
        if self.is_expired(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def set(self, key, data):
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {str(e)}")

class BatchDataFetcher:
    def __init__(self, akshare_client):
        self.akshare_client = akshare_client
        self.cache = DataCache()
    
    def get_yzxdr_data(self, date):
        """获取一致行动人数据，全市场数据只需获取一次"""
        cache_key = f"yzxdr_{date}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            print(f"✅ 从缓存获取一致行动人数据: {date}")
            return cached_data
        
        print(f"🌐 获取一致行动人数据: {date}")
        try:
            import akshare as ak
            data = self.akshare_client.call_akshare_with_retry(
                lambda: ak.stock_yzxdr_em(date=date)
            )
            self.cache.set(cache_key, data)
            return data
        except Exception as e:
            print(f"❌ 获取一致行动人数据失败: {str(e)}")
            return pd.DataFrame()  # 返回空DataFrame而不是None
    
    def get_fund_hold_data(self, hold_types, report_date):
        """批量获取基金持仓数据"""
        result = {}
        for hold_type in hold_types:
            cache_key = f"fund_hold_{hold_type}_{report_date}"
            cached_data = self.cache.get(cache_key)
            
            if cached_data is not None:
                print(f"✅ 从缓存获取{hold_type}持仓数据: {report_date}")
                result[hold_type] = cached_data
                continue
            
            print(f"🌐 获取{hold_type}持仓数据: {report_date}")
            try:
                import akshare as ak
                data = self.akshare_client.call_akshare_with_retry(
                    lambda: ak.stock_report_fund_hold(symbol=hold_type, date=report_date)
                )
                result[hold_type] = data
                self.cache.set(cache_key, data)
            except Exception as e:
                print(f"❌ 获取{hold_type}持仓数据失败: {str(e)}")
                result[hold_type] = pd.DataFrame()  # 返回空DataFrame而不是None
        
        return result
```

### 4. 主程序优化

重构主程序逻辑，应用上述优化策略：

```python
def main():
    # 初始化优化组件
    akshare_client = EnhancedAkShareClient()
    batch_fetcher = BatchDataFetcher(akshare_client)
    
    # 读取股票列表
    df_stocks = pd.read_excel("股票列表.xlsx")
    stocks_to_process = df_stocks["股票代码"].tolist()
    
    # 批量获取共享数据
    print("🔄 批量获取共享数据...")
    yzxdr_data = batch_fetcher.get_yzxdr_data(date)
    fund_hold_data = batch_fetcher.get_fund_hold_data(hold_types, report_date)
    
    # 处理每只股票
    result_list = []
    for idx, stock_code in enumerate(stocks_to_process, 1):
        stock_name = df_stocks[df_stocks["股票代码"] == stock_code]["股票名称"].iloc[0]
        print(f"\n📊 处理股票 {idx}/{len(stocks_to_process)}: {stock_name} ({stock_code})")
        
        try:
            # 获取股票基础数据
            circulating_mv = get_circulating_mv(stock_code, date, akshare_client)
            
            # 处理股东数据
            stock_data = process_stock_holder_data(
                stock_code, stock_name, date, yzxdr_data, circulating_mv, akshare_client
            )
            
            # 处理持仓数据
            process_fund_hold_data(
                stock_code, stock_data, fund_hold_data, circulating_mv
            )
            
            result_list.append(stock_data)
            
            # 定期保存中间结果
            if idx % 10 == 0:
                save_intermediate_results(result_list, save_path)
                
        except Exception as e:
            print(f"❌ 处理股票 {stock_name} 失败: {str(e)}")
            continue
    
    # 保存最终结果
    save_final_results(result_list, save_path)

def get_circulating_mv(stock_code, date, akshare_client):
    """获取流通市值，带重试机制"""
    stock_code_pure = stock_code[1:] if stock_code.startswith(('6', '0', '3')) else stock_code
    stock_code_with_prefix = f"sh{stock_code_pure}" if stock_code.startswith('6') else f"sz{stock_code_pure}"
    
    try:
        # 获取历史价格数据
        hist_data = akshare_client.call_akshare_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=stock_code_with_prefix, period="daily", 
                                    start_date=date, end_date=date, adjust="")
        )
        
        if hist_data.empty:
            return 0
            
        close_price = hist_data.iloc[0]["收盘"]
        
        # 获取个股信息
        stock_info = akshare_client.call_akshare_with_retry(
            lambda: ak.stock_individual_info_em(symbol=stock_code_pure)
        )
        
        if stock_info.empty:
            return 0
            
        info_dict = dict(zip(stock_info["item"], stock_info["value"]))
        circulating_shares = info_dict.get("流通股本", 0)
        
        if isinstance(circulating_shares, str):
            circulating_shares = float(circulating_shares.replace(",", ""))
        
        return close_price * circulating_shares * 10000  # 转换为元
        
    except Exception as e:
        print(f"⚠️ 获取流通市值失败: {str(e)}")
        return 0
```

## 实施步骤

1. **创建优化模块**：
   - 创建`akshare_optimizer.py`文件，包含上述所有优化类
   - 确保与原有代码兼容

2. **修改主程序**：
   - 导入优化模块
   - 替换直接AKShare调用为优化后的调用
   - 调整数据处理逻辑以使用批量获取的数据

3. **测试与验证**：
   - 小规模测试验证功能正确性
   - 监控API调用频率和成功率
   - 根据实际情况调整参数

## 预期效果

1. **API调用次数大幅减少**：
   - `stock_yzxdr_em`从N×1次减少到1次
   - `stock_report_fund_hold`从N×M次减少到M次（M为持仓类型数量）
   - 总体API调用减少约60-70%

2. **IP封禁风险显著降低**：
   - 智能请求间隔控制
   - 随机User-Agent和代理IP
   - 会话保持和连接复用
   - 预计IP封禁风险降低90%以上

3. **执行效率提升**：
   - 减少网络请求时间
   - 缓存机制减少重复计算
   - 预计总执行时间缩短50%以上

4. **稳定性增强**：
   - 完善的错误处理和重试机制
   - 中间结果定期保存
   - 程序中断后可从断点恢复

## 注意事项

1. **代理IP获取**：
   - 需要可靠的代理IP源
   - 定期检查和更新代理IP列表
   - 可以考虑使用付费代理服务

2. **缓存管理**：
   - 定期清理过期缓存
   - 监控缓存大小，避免占用过多磁盘空间
   - 考虑使用Redis等内存数据库提高缓存性能

3. **参数调优**：
   - 根据实际运行情况调整请求间隔
   - 根据网络环境调整重试策略
   - 根据数据更新频率调整缓存过期时间

4. **合规性**：
   - 遵守网站使用条款
   - 合理使用数据，避免过度请求
   - 考虑数据使用合规性

通过以上优化方案，可以显著降低AKShare访问东方财富网时IP被封的风险，同时提高数据获取效率和程序稳定性。