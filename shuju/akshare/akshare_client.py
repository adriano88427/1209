import random
import threading
import time
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import akshare as ak
import pandas as pd
import requests

from akshare_config import DEFAULT_CLIENT_CONFIG, AkShareClientConfig


class RateLimiter:
    def __init__(
        self,
        min_interval: float = 0.4,
        max_calls_per_minute: int = 60,
        slow_multiplier: float = 2.5,
    ) -> None:
        self.min_interval = min_interval
        self.max_calls = max_calls_per_minute
        self.slow_multiplier = slow_multiplier
        self.mode = "normal"
        self.lock = threading.Lock()
        self.call_times: deque[float] = deque()

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.mode = mode

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()

            if len(self.call_times) >= self.max_calls:
                sleep_time = 60 - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            effective_interval = (
                self.min_interval * (self.slow_multiplier if self.mode == "slow" else 1.0)
            )
            if self.call_times:
                elapsed = now - self.call_times[-1]
                if elapsed < effective_interval:
                    time.sleep(effective_interval - elapsed)

            self.call_times.append(time.time())


class CacheLayer:
    def __init__(self, cache_root: str = "ak_cache") -> None:
        self.memory_cache: Dict[str, tuple[float, Any]] = {}
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _file_path(self, cache_key: str, cache_type: str) -> Path:
        safe_key = cache_key.replace("/", "_")
        return self.cache_root / f"{cache_type}_{safe_key}.pkl"

    def get(self, cache_key: str, cache_type: str, ttl: Optional[int]) -> Optional[Any]:
        now = time.time()
        mem_key = f"{cache_type}:{cache_key}"
        if mem_key in self.memory_cache:
            timestamp, data = self.memory_cache[mem_key]
            if ttl is None or now - timestamp < ttl:
                return data
            self.memory_cache.pop(mem_key, None)

        file_path = self._file_path(cache_key, cache_type)
        if file_path.exists():
            try:
                timestamp, data = pickle.loads(file_path.read_bytes())
                if ttl is None or now - timestamp < ttl:
                    self.memory_cache[mem_key] = (timestamp, data)
                    return data
            except Exception:
                pass

        return None

    def set(self, cache_key: str, cache_type: str, data: Any) -> None:
        timestamp = time.time()
        mem_key = f"{cache_type}:{cache_key}"
        self.memory_cache[mem_key] = (timestamp, data)
        file_path = self._file_path(cache_key, cache_type)
        try:
            file_path.write_bytes(pickle.dumps((timestamp, data)))
        except Exception:
            pass


class AntiCrawlManager:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/124.0",
    ]

    def __init__(
        self,
        use_proxy: bool = False,
        proxy_pool: Optional[list[str]] = None,
        adaptive_delay_base: float = 0.8,
        adaptive_delay_threshold: int = 45,
    ) -> None:
        self.use_proxy = use_proxy
        self.proxy_pool = proxy_pool or []
        self.adaptive_delay_base = adaptive_delay_base
        self.adaptive_delay_threshold = adaptive_delay_threshold
        self.request_times: deque[float] = deque()
        self.block_events = 0
        self._patch_requests()

    def _patch_requests(self) -> None:
        if getattr(requests.sessions.Session, "_akshare_patched", False):
            return

        original_request = requests.sessions.Session.request
        manager = self

        def patched_request(session_self, method, url, **kwargs):
            headers = kwargs.get("headers") or {}
            headers.setdefault("User-Agent", manager.random_user_agent())
            headers.setdefault("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
            headers.setdefault("Referer", "https://www.eastmoney.com/")
            kwargs["headers"] = headers

            if manager.use_proxy and manager.proxy_pool:
                proxy = manager.get_proxy()
                if proxy:
                    kwargs.setdefault("proxies", proxy)

            manager.adaptive_delay()
            response = original_request(session_self, method, url, **kwargs)
            manager.inspect_response(response)
            return response

        requests.sessions.Session.request = patched_request
        requests.sessions.Session._akshare_patched = True

    def random_user_agent(self) -> str:
        return random.choice(self.USER_AGENTS)

    def get_proxy(self) -> Optional[Dict[str, str]]:
        if not self.proxy_pool:
            return None
        proxy_entry = random.choice(self.proxy_pool)
        return {"http": proxy_entry, "https": proxy_entry}

    def adaptive_delay(self) -> None:
        now = time.time()
        while self.request_times and now - self.request_times[0] > 300:
            self.request_times.popleft()

        delay = self.adaptive_delay_base
        if len(self.request_times) > self.adaptive_delay_threshold:
            delay *= 1.5
        delay += random.uniform(0, 0.5)
        time.sleep(delay)
        self.request_times.append(time.time())

    def inspect_response(self, response: requests.Response) -> None:
        if response.status_code in (403, 429):
            self.block_events += 1
            raise RuntimeError(f"BLOCKED:{response.status_code}")


class ErrorRecoveryManager:
    def __init__(self, rate_limiter: RateLimiter) -> None:
        self.rate_limiter = rate_limiter
        self.block_counter = 0

    def handle(self, exc: Exception) -> None:
        message = str(exc).lower()
        if any(code in message for code in ("blocked", "403", "429")):
            self.block_counter += 1
            self.rate_limiter.set_mode("slow")
            wait_time = min(30, 5 * self.block_counter)
            print(f"⚠️ 检测到可能被限流，暂停 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        else:
            self.rate_limiter.set_mode("normal")
            time.sleep(2)


class PerformanceMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total_requests = 0
        self.cache_hits = 0
        self.failures = 0
        self.blocked = 0
        self.total_time = 0.0

    def record(self, success: bool, cache_hit: bool, elapsed: float, blocked: bool = False) -> None:
        with self.lock:
            self.total_requests += 1
            if cache_hit:
                self.cache_hits += 1
            if not success:
                self.failures += 1
            if blocked:
                self.blocked += 1
            self.total_time += elapsed

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            avg_time = self.total_time / self.total_requests if self.total_requests else 0.0
            hit_rate = (
                self.cache_hits / self.total_requests * 100 if self.total_requests else 0.0
            )
            fail_rate = (
                self.failures / self.total_requests * 100 if self.total_requests else 0.0
            )
            block_rate = (
                self.blocked / self.total_requests * 100 if self.total_requests else 0.0
            )
            return {
                "requests": self.total_requests,
                "cache_hit_rate": f"{hit_rate:.1f}%",
                "failure_rate": f"{fail_rate:.1f}%",
                "block_rate": f"{block_rate:.1f}%",
                "avg_latency": f"{avg_time:.2f}s",
            }


class AkShareClient:
    TTL_MAPPING = {
        "yzxdr": 12 * 3600,
        "fund_hold": 6 * 3600,
        "stock_info": 24 * 3600,
        "management": 6 * 3600,
        "top10_free": 24 * 3600,
        "top10_history": 24 * 3600,
        "hist": 3600,
        "share_change": 24 * 3600,
    }

    def __init__(self, config: AkShareClientConfig = DEFAULT_CLIENT_CONFIG) -> None:
        self.config = config
        self.cache = CacheLayer(config.cache_dir)
        self.rate_limiter = RateLimiter(
            min_interval=config.min_interval,
            max_calls_per_minute=config.max_calls_per_minute,
            slow_multiplier=config.slow_mode_multiplier,
        )
        self.anti_crawl = AntiCrawlManager(
            use_proxy=config.use_proxy,
            proxy_pool=config.proxy_pool,
            adaptive_delay_base=config.adaptive_delay_base,
            adaptive_delay_threshold=config.adaptive_delay_threshold,
        )
        self.error_manager = ErrorRecoveryManager(self.rate_limiter)
        self.monitor = PerformanceMonitor()
        self.stock_hist_disabled = False

    def _call(
        self,
        cache_type: str,
        cache_key: str,
        loader: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        ttl = self.TTL_MAPPING.get(cache_type)
        cached = self.cache.get(cache_key, cache_type, ttl)
        if cached is not None:
            self.monitor.record(True, True, 0.0, blocked=False)
            return cached

        last_exception: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait()
                start = time.time()
                result = loader()
                elapsed = time.time() - start
                self.cache.set(cache_key, cache_type, result)
                self.monitor.record(True, False, elapsed, blocked=False)
                self.rate_limiter.set_mode("normal")
                return result
            except Exception as exc:
                last_exception = exc
                blocked = any(code in str(exc).lower() for code in ("blocked", "403", "429"))
                self.monitor.record(False, False, 0.0, blocked=blocked)
                self.error_manager.handle(exc)
                if blocked and cache_type == "hist":
                    self.stock_hist_disabled = True

        if last_exception:
            raise last_exception
        raise RuntimeError("AKShare request failed without exception detail")

    def stock_yzxdr(self, date: str) -> pd.DataFrame:
        return self._call("yzxdr", date, lambda: ak.stock_yzxdr_em(date=date))

    def stock_report_fund_hold(self, hold_type: str, date: str) -> pd.DataFrame:
        cache_key = f"{hold_type}_{date}"
        return self._call("fund_hold", cache_key, lambda: ak.stock_report_fund_hold(symbol=hold_type, date=date))

    def stock_individual_info(self, symbol: str) -> pd.DataFrame:
        return self._call("stock_info", symbol, lambda: ak.stock_individual_info_em(symbol=symbol))

    def stock_hist(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.stock_hist_disabled:
            raise RuntimeError("stock_hist接口已被禁用")

        cache_key = f"{symbol}_{start_date}_{end_date}"
        return self._call(
            "hist",
            cache_key,
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        )

    def stock_management_change(self, symbol: str) -> pd.DataFrame:
        return self._call("management", symbol, lambda: ak.stock_management_change_ths(symbol=symbol))

    def stock_top10_free(self, symbol: str, date: str) -> pd.DataFrame:
        cache_key = f"{symbol}_{date}"
        return self._call("top10_free", cache_key, lambda: ak.stock_gdfx_free_top_10_em(symbol=symbol, date=date))

    def stock_top10_history(self, symbol: str, date: str) -> pd.DataFrame:
        cache_key = f"{symbol}_{date}"
        return self._call("top10_history", cache_key, lambda: ak.stock_gdfx_top_10_em(symbol=symbol, date=date))

    def stock_share_change(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = f"{symbol}_{start_date}_{end_date}"
        return self._call(
            "share_change",
            cache_key,
            lambda: ak.stock_share_change_cninfo(symbol=symbol, start_date=start_date, end_date=end_date),
        )

    def metrics(self) -> Dict[str, Any]:
        summary = self.monitor.summary()
        summary["rate_mode"] = self.rate_limiter.mode
        summary["blocked_events"] = self.anti_crawl.block_events
        return summary
