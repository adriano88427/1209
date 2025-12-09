from dataclasses import dataclass, field
from typing import List
import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class AkShareClientConfig:
    cache_dir: str = os.getenv("AKSHARE_CACHE_DIR", "ak_cache")
    min_interval: float = _env_float("AKSHARE_MIN_INTERVAL", 0.4)
    max_calls_per_minute: int = _env_int("AKSHARE_MAX_CALLS_PER_MINUTE", 60)
    max_retries: int = _env_int("AKSHARE_MAX_RETRIES", 3)
    adaptive_delay_base: float = _env_float("AKSHARE_ADAPTIVE_DELAY_BASE", 0.8)
    adaptive_delay_threshold: int = _env_int("AKSHARE_ADAPTIVE_DELAY_THRESHOLD", 45)
    slow_mode_multiplier: float = _env_float("AKSHARE_SLOW_MODE_MULTIPLIER", 2.5)
    use_proxy: bool = _env_bool("AKSHARE_USE_PROXY", False)
    proxy_pool: List[str] = field(default_factory=list)
    enable_async_prefetch: bool = _env_bool("AKSHARE_ENABLE_ASYNC_PREFETCH", False)


DEFAULT_CLIENT_CONFIG = AkShareClientConfig()
