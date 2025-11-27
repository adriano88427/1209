# -*- coding: utf-8 -*-

"""配置与常量模块，集中管理路径/文件相关设置。"""

import os

# 默认数据文件路径设置（方便用户修改）
DEFAULT_DATA_FILE = "创业板单日下跌14%详细交易日数据（清理后）1114.xlsx"

# 统一的报表输出目录（项目根目录下的 baogao）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_OUTPUT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "baogao"))
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


def build_report_path(filename: str) -> str:
    """将文件名映射到报表输出目录."""
    if not filename:
        return REPORT_OUTPUT_DIR
    if os.path.isabs(filename):
        return filename
    return os.path.join(REPORT_OUTPUT_DIR, filename)


# 板块 & 样本相关常量（内置映射，避免额外配置文件）
SEGMENT_MIN_SAMPLES = 30
SEGMENT_MIN_DAILY = 2
MARKET_SEGMENT_RULES = (
    ("创业板", ("SZ300", "SZ301"), ("300", "301")),
    ("科创板", ("SH688",), ("688",)),
    ("北交所", ("BJ",), tuple(str(code) for code in range(830, 840))),
    ("深圳主板", ("SZ000", "SZ001", "SZ002", "SZ003"), ("000", "001", "002", "003")),
    ("沪主板", ("SH600", "SH601", "SH603", "SH605"), ("600", "601", "603", "605")),
)


# 动态可靠性加权配置
RELIABILITY_CONFIG = {
    'base_weights': {
        'base': 0.35,
        'overall': 0.25,
        'rolling': 0.15,
        'temporal': 0.125,
        'sample': 0.125,
    },
    'scale_bounds': (0.4, 1.6),
    'normalized_bounds': (0.05, 0.55),
    'drop_threshold': 0.35,
}
