# -*- coding: utf-8 -*-

"""配置与常量模块，集中管理路径/文件相关设置。"""

import copy
import os
from typing import Dict, Any, List, Tuple

# 年化收益率统一口径：全系统使用“日均收益 × 252”线性年化（不做复利），无配置开关，保持与各报告/榜单一致。

# ============================================================
# 分析开关：用于控制哪些分析模块会被执行。
# - single_nonparam / single_param 默认为 True，保持原有行为。
# - dual_nonparam / dual_param 默认由环境变量控制，便于按需启用双因子分析。
# ============================================================
ANALYSIS_SWITCHES: Dict[str, bool] = {
    "single_nonparam": True,  # 单因子-非参数分析
    "single_param": True,     # 单因子-带参数分析
    "single_param_flex": True,  # 单因子-自由区间挖掘（新增）
    "dual_nonparam": True,    # 双因子-非参数分析（在此处切换，无需环境变量）
    "dual_param": True,       # 双因子-带参数分析（在此处切换，无需环境变量）
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录 / 报表目录（相对于项目根目录）
DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "shuju"))
DATA_TABLE_DIR = os.path.join(DATA_DIR, "biaoge")
REPORT_OUTPUT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "baogao"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
# 报告分类目录（HTML/表格/检验）
REPORT_HTML_DIR = os.path.join(REPORT_OUTPUT_DIR, "baogao")
REPORT_TABLE_DIR = os.path.join(REPORT_OUTPUT_DIR, "biaoge")
REPORT_AUDIT_DIR = os.path.join(REPORT_OUTPUT_DIR, "jianyan")
for _dir in (REPORT_HTML_DIR, REPORT_TABLE_DIR, REPORT_AUDIT_DIR):
    os.makedirs(_dir, exist_ok=True)

# =============================
# 用户可配置项（统一集中管理）
# =============================

# 默认数据文件路径设置（方便用户修改）
# 如果有多个年度/批次的数据文件，可在列表中继续追加路径。
DEFAULT_DATA_FILES = [
   
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2025.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2024.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2023.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2022.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2021.xlsx"),
]
# 为保持兼容性，保留单文件常量（默认取列表中的第一个）
DEFAULT_DATA_FILE = DEFAULT_DATA_FILES[0] if DEFAULT_DATA_FILES else ""

# Excel 解析配置（可根据需要扩展）
DATA_PARSE_CONFIG: Dict[str, Any] = {
    "preferred_engines": ["pyarrow", "openpyxl", "pandas"],
    "sheet_policy": "all",
    "na_values": ["", " ", "NA", "N/A", "-", "—", "--"],
    "column_aliases": {
        "股票代码": ["证券代码", "code", "stock_code"],
        "股票名称": ["证券名称", "name", "stock_name"],
        "信号日期": ["signal_date", "日期", "交易日期"],
        "次日开盘买入持股两日收益率": ["收益率", "return", "return_rate"],
        "当日回调": ["当日回撤", "daily_drawdown"],
        "机构持股比例(%)": ["机构持股比例", "inst_holding"],
        "流通市值(元)": ["流通市值", "market_cap"],
        "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": [
            "财务投资机构合计"
        ],
        "朋友合计（企业大股东（大非）+社保+保险）": ["朋友合计"],
        "十大流通个人持股合计": [],
        "高管/大股东持股比例大非": [],
        "普通散户持股比例": [],
        "企业大股东大非（包含国资）": [],
        "企业大股东（包含国资）（小非）": [],
        "前10大流通股东持股比例合计": [],
        "十大流通股东小非合计": [],
        "十大流通股东大非合计": [],
        "十大流通机构大非": [],
        "十大流通机构小非": [],
        "基金持仓占比": [],
        "QFII持仓占比": [],
        "持有基金家数": [],
        "当日最高涨幅": ["当日最大涨幅"],
         "当日收盘涨跌幅": ["收盘涨跌幅","信号当日收盘涨跌幅"],
         "次日开盘涨跌幅": []
    },
    # column_types 取值说明：
    # - "percent": 原始值以百分比表示，解析时除以100并存为小数
    # - "amount": 金额/数量，保持原值
    # - "auto": 无法预判类型，按默认逻辑解析并在日志中提示确认
    "column_types": {
        "当日回调": "percent",
        "机构持股比例(%)": "percent",
        "流通市值(元)": "amount",
        "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": "percent",
        "朋友合计（企业大股东（大非）+社保+保险）": "percent",
        "十大流通个人持股合计": "percent",
        "高管/大股东持股比例大非": "percent",
        "普通散户持股比例": "percent",
        "企业大股东大非（包含国资）": "percent",
        "企业大股东（包含国资）（小非）": "percent",
        "前10大流通股东持股比例合计": "percent",
        "十大流通股东小非合计": "percent",
        "十大流通股东大非合计": "percent",
        "十大流通机构大非": "percent",
        "十大流通机构小非": "percent",
        "基金持仓占比": "percent",
        "QFII持仓占比": "percent",
        "持有基金家数": "auto",
         "当日最高涨幅": "percent",
        "次日开盘涨跌幅": "percent",
        "当日最高涨幅": "percent",
    },
}

# 收益率列名称（所有IC/收益分析均依赖此列）
RETURN_COLUMN = '次日开盘买入持股两日收益率'


# IC ??????
# - IC_MIN_SAMPLES_DAY: ?/?/????????????????????????
# - IC_MIN_UNIQUE_DAY: ?/?/????????????????????2?
# - IC_MIN_UNIQUE_TOTAL: ??IC??????????
# - IC_USE_RELAXED_UNIQUE: ????????????????????5/3/2
IC_MIN_SAMPLES_DAY: List[int] = [5, 3, 2]
IC_MIN_UNIQUE_DAY: List[int] = [2, 2, 2]
IC_MIN_UNIQUE_TOTAL: int = 2
IC_USE_RELAXED_UNIQUE: bool = True

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized == "":
        return default
    return normalized in {"1", "true", "yes", "on", "enable", "enabled"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variables with graceful fallback."""
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from environment variables with graceful fallback."""
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_csv_numbers(name: str, cast_func=float) -> List:
    """
    Parse numeric CSV-style environment variables, ignoring invalid entries.
    """
    value = os.getenv(name)
    if value is None:
        return []
    entries = str(value).replace(";", ",").split(",")
    parsed: List = []
    for entry in entries:
        cleaned = entry.strip()
        if not cleaned:
            continue
        try:
            parsed.append(cast_func(cleaned))
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_factor_rule_overrides(raw: str) -> Dict[str, str]:
    """
    Parse overrides of the form '因子:mode,因子2:mode' from environment variables.
    """
    if not raw:
        return {}
    overrides: Dict[str, str] = {}
    tokens = str(raw).replace(";", ",").split(",")
    for token in tokens:
        chunk = token.strip()
        if not chunk or ":" not in chunk:
            continue
        factor, mode = chunk.split(":", 1)
        factor_name = factor.strip()
        normalized_mode = mode.strip().lower()
        if not factor_name or normalized_mode not in {"market_cap", "industry", "both", "none"}:
            continue
        overrides[factor_name] = normalized_mode
    return overrides


def _derive_factor_columns() -> List[str]:
    """根据 column_types 定义自动生成因子列列表，避免重复维护。"""
    column_types = DATA_PARSE_CONFIG.get("column_types", {}) or {}
    ordered_columns: List[str] = []
    seen = set()
    for column in column_types.keys():
        normalized = str(column).strip()
        if not normalized or normalized == RETURN_COLUMN:
            continue
        if normalized not in seen:
            ordered_columns.append(normalized)
            seen.add(normalized)
    return ordered_columns

# 因子语义 & 处理策略元数据
FACTOR_META: Dict[str, Dict[str, Any]] = {
    "当日回调": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
        "aliases": ["当日回撤", "daily_drawdown"],
    },
    "次日开盘买入持股两日收益率": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
        "aliases": ["收益率", "return", "return_rate"],
    },
    "机构持股比例(%)": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "普通散户持股比例": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "基金持仓占比": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "QFII持仓占比": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "流通市值(元)": {
        "semantic": "amount",
        "display": "amount",
        "scale_hint": 1.0,
        "unit": "CNY",
        "aliases": ["流通市值", "market_cap"],
    },
    "持有基金家数": {
        "semantic": "numeric",
        "display": "numeric",
    },
}

# 因子分组规则（针对未知列名自动匹配语义）
FACTOR_GROUP_RULES = [
    {
        "pattern": r"(回调|回撤|收益率)",
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    {
        "pattern": r"(占比|比例|持股)",
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    {
        "pattern": r"(市值|金额|资产|流通)",
        "semantic": "amount",
        "display": "amount",
        "scale_hint": 1.0,
    },
]

PERCENT_STYLE_COLUMNS: List[str] = []

COLUMN_ALIGNMENT_RULES: Dict[str, Dict[str, Any]] = {}

# 需要分析的因子列（非参数 & 带参数分析都会引用）
# 程序会自动对这些列（及收益列）尝试进行字符串/百分比到数值的转换，无需额外配置。
FACTOR_COLUMNS = _derive_factor_columns()

# ============================================================
# 因子中性化配置：按照项目方案将适合的因子映射到市值/行业/双重中性化
# ============================================================
FACTOR_NEUTRALIZATION_RULES: Dict[str, str] = {
    "当日回调": "both",
    "当日最高涨幅": "both",
    "次日开盘涨跌幅": "both",
    "基金持仓占比": "both",
    "QFII持仓占比": "both",
    "机构持股比例(%)": "industry",
    "普通散户持股比例": "industry",
    "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": "industry",
    "朋友合计（企业大股东（大非）+社保+保险）": "industry",
    "十大流通个人持股合计": "industry",
    "高管/大股东持股比例大非": "industry",
    "企业大股东大非（包含国资）": "industry",
    "企业大股东（包含国资）（小非）": "industry",
    "前10大流通股东持股比例合计": "industry",
    "十大流通股东小非合计": "industry",
    "十大流通股东大非合计": "industry",
    "十大流通机构大非": "industry",
    "十大流通机构小非": "industry",
    "持有基金家数": "market_cap",
    "流通市值(元)": "none",
}

_FACTOR_RULE_OVERRIDES = _parse_factor_rule_overrides(os.getenv("FA_NEUTRAL_RULES"))
if _FACTOR_RULE_OVERRIDES:
    _RESOLVED_NEUTRAL_RULES = copy.deepcopy(FACTOR_NEUTRALIZATION_RULES)
    _RESOLVED_NEUTRAL_RULES.update(_FACTOR_RULE_OVERRIDES)
else:
    _RESOLVED_NEUTRAL_RULES = copy.deepcopy(FACTOR_NEUTRALIZATION_RULES)

NEUTRALIZATION_CONFIG: Dict[str, Any] = {
    # 默认关闭中性化以加快运行，如需开启请将环境变量 FA_NEUTRALIZATION_ENABLED 设为 1
    "enabled": _env_flag("FA_NEUTRALIZATION_ENABLED", False),
    "signal_date_column": os.getenv("FA_NEUTRAL_SIGNAL_COLUMN", "信号日期") or "信号日期",
    "market_cap_column": os.getenv("FA_NEUTRAL_MCAP_COLUMN", "流通市值(元)") or "流通市值(元)",
    "industry_column": os.getenv("FA_NEUTRAL_INDUSTRY_COLUMN", "所属同花顺行业") or "所属同花顺行业",
    # 默认保留前两级行业（示例：医药生物-化学制药），满足“剔除细分第三级”的要求
    "industry_level": max(1, _env_int("FA_NEUTRAL_INDUSTRY_LEVEL", 2)),
    "industry_separator": os.getenv("FA_NEUTRAL_INDUSTRY_SEP", "-") or "-",
    "min_cross_section": max(2, _env_int("FA_NEUTRAL_MIN_CROSS_SECTION", 8)),
    "min_industry_group": max(2, _env_int("FA_NEUTRAL_MIN_INDUSTRY", 4)),
    "store_raw_suffix": os.getenv("FA_NEUTRAL_RAW_SUFFIX", "__raw") or "__raw",
    "default_method": (os.getenv("FA_NEUTRAL_DEFAULT_METHOD") or "none").strip().lower() or "none",
    "factor_rules": _RESOLVED_NEUTRAL_RULES,
    "qa_threshold": max(0.0, min(1.0, _env_float("FA_NEUTRAL_QA_THRESHOLD", 0.6))),
    "qa_output_path": os.getenv(
        "FA_NEUTRAL_QA_PATH",
        os.path.join(REPORT_OUTPUT_DIR, "QA_neutralization.csv"),
    ),
}

# =============================
# 单因子自由区间挖掘配置（新增）
# =============================
SINGLE_PARAM_FLEX_SETTINGS: Dict[str, Any] = {
    # 样本下限模式：auto 按总样本/20，自适应；fixed 使用 min_samples_fixed
    "min_samples_mode": os.getenv("FA_FLEX_MIN_SAMPLES_MODE", "auto"),
    "min_samples_fixed": max(1, _env_int("FA_FLEX_MIN_SAMPLES_FIXED", 500)),
    # 分档设置：多粒度分位/等距 + 滑窗
    "quantile_bins": _env_csv_numbers("FA_FLEX_QUANTILE_BINS", int) or [8, 12, 16, 20],
    "equal_bins": _env_csv_numbers("FA_FLEX_EQUAL_BINS", int) or [8, 12],
    "sliding_windows": _env_csv_numbers("FA_FLEX_SLIDING_WINDOWS", float) or [0.05, 0.10, 0.20],  # 宽度比例，步长=宽度/2
    # 每个因子最多保留的区间数（仅非用户区间）
    "max_ranges_per_factor": max(1, _env_int("FA_FLEX_MAX_RANGES", 50)),
    # 是否允许用户自定义区间
    "enable_user_ranges": _env_flag("FA_FLEX_ENABLE_USER_RANGES", True),
    # 用户自定义区间，格式 {因子名: [(low, high), ...]}
    # 示例：机构持股比例(%) 在 0%~13.31%
    "user_ranges": {
        "机构持股比例(%)": [(0.0, 0.1331)],
    },
    # 榜单默认展示的 top N
    "report_top_n": max(1, _env_int("FA_FLEX_REPORT_TOP_N", 20)),
    # 是否导出全量候选（过滤样本下限但不截断）
    "export_all": _env_flag("FA_FLEX_EXPORT_ALL", True),
}


def validate_single_param_flex_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """校验/填充自由区间挖掘配置。"""
    base = dict(SINGLE_PARAM_FLEX_SETTINGS)
    cfg = cfg or {}
    base.update({k: v for k, v in cfg.items() if v is not None})
    # 纠正基本类型
    base["min_samples_mode"] = str(base.get("min_samples_mode", "auto")).strip().lower() or "auto"
    base["min_samples_fixed"] = max(1, int(base.get("min_samples_fixed", 500)))
    # 分档/滑窗参数
    base["quantile_bins"] = [int(x) for x in base.get("quantile_bins", [8, 12, 16, 20]) if x]
    base["equal_bins"] = [int(x) for x in base.get("equal_bins", [8, 12]) if x]
    base["sliding_windows"] = [float(x) for x in base.get("sliding_windows", [0.05, 0.10, 0.20]) if x]
    base["max_ranges_per_factor"] = max(1, int(base.get("max_ranges_per_factor", 50)))
    base["enable_user_ranges"] = bool(base.get("enable_user_ranges", True))
    base["report_top_n"] = max(1, int(base.get("report_top_n", 20)))
    base["export_all"] = bool(base.get("export_all", False))
    # user_ranges 保持原样（应为 dict）
    if not isinstance(base.get("user_ranges"), dict):
        base["user_ranges"] = {}
    return base

# =============================
# 双因子分析配置
# =============================

DUAL_FACTOR_SETTINGS: Dict[str, Any] = {
    "nonparam_bins": int(os.getenv("FA_DUAL_NONPARAM_BINS", "5")),
    "nonparam_top_n": int(os.getenv("FA_DUAL_NONPARAM_TOP_N", "6")),
    "max_factor_pairs": int(os.getenv("FA_DUAL_MAX_PAIRS", "30")),
    "min_samples": int(os.getenv("FA_DUAL_MIN_SAMPLES", "800")),
    "nonparam_factor_pairs": [],
    "param_factor_pairs": [],
    "param_ranges": {},
    "param_min_samples": int(os.getenv("FA_DUAL_PARAM_MIN_SAMPLES", "300")),
    "param_default_bins": int(os.getenv("FA_DUAL_PARAM_BINS", "3")),
    "enable_prescreen": os.getenv("FA_DUAL_PRESCREEN", "true").strip().lower() != "false",
}

DUAL_REPORT_OPTIONS: Dict[str, Any] = {
    "nonparam_prefix": "双因子分析",
    "param_prefix": "带参数的双因子自由区间挖掘",
    "output_dir": os.getenv("FA_DUAL_OUTPUT_DIR", REPORT_OUTPUT_DIR),
    "heatmap_enabled": True,
    "max_rank_display": 10,
}

_DUAL_CONFIG_LIMITS = {
    "nonparam_bins": (3, 10),
    "nonparam_top_n": (2, 20),
    "max_factor_pairs": (1, 100),
    "min_samples": (200, 5000),
    "param_min_samples": (50, 2000),
    "param_default_bins": (2, 10),
}
os.makedirs(DUAL_REPORT_OPTIONS["output_dir"], exist_ok=True)


def validate_dual_config(settings: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    校验双因子配置，确保核心参数在合理范围。
    返回安全的配置副本，不会修改原始常量。
    """
    cfg = copy.deepcopy(DUAL_FACTOR_SETTINGS)
    if settings:
        for key, value in settings.items():
            if value is not None:
                cfg[key] = value

    for key, (low, high) in _DUAL_CONFIG_LIMITS.items():
        if key not in cfg:
            continue
        try:
            numeric = int(cfg[key])
        except (TypeError, ValueError):
            numeric = low
        if numeric < low:
            print(f"[WARN] 双因子配置 {key}={numeric} 低于推荐下限，已调整为 {low}")
            numeric = low
        if numeric > high:
            print(f"[WARN] 双因子配置 {key}={numeric} 高于推荐上限，已调整为 {high}")
            numeric = high
        cfg[key] = numeric
    return cfg

# =============================
# 系统内置常量（无需用户调整）
# =============================

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

# =============================
# 辅助稳健性分析配置
# =============================
_AUX_WINDOW_SIZES: Tuple[int, ...] = tuple(
    int(value) for value in (_env_csv_numbers("FA_AUX_WINDOW_SIZES", int) or (30, 60))
)
_AUX_SAMPLE_SIZES: Tuple[float, ...] = tuple(
    float(value) for value in (_env_csv_numbers("FA_AUX_SAMPLE_SIZES", float) or (0.8, 0.9, 1.0))
)
_AUX_FAST_WINDOWS: Tuple[int, ...] = tuple(
    int(value) for value in (_env_csv_numbers("FA_AUX_FAST_WINDOWS", int) or (20, 40))
)
_AUX_FAST_SAMPLES: Tuple[float, ...] = tuple(
    float(value) for value in (_env_csv_numbers("FA_AUX_FAST_SAMPLES", float) or (0.9,))
)

AUX_ANALYSIS_OPTIONS: Dict[str, Any] = {
    # 默认关闭辅助分析以降低耗时，如需开启请将环境变量 FA_AUX_ENABLED 设为 1
    "enabled": _env_flag("FA_AUX_ENABLED", False),
    "mode": os.getenv("FA_AUX_MODE", "full") or "full",
    "window_sizes": _AUX_WINDOW_SIZES,
    "sample_sizes": _AUX_SAMPLE_SIZES,
    "n_iterations": max(1, _env_int("FA_AUX_ITERATIONS", 100)),
    "debug_enabled": _env_flag("FA_AUX_DEBUG_ENABLED", False),
    "log_details": _env_flag("FA_AUX_LOG_DETAILS", False),
    "cache_enabled": _env_flag("FA_AUX_CACHE_ENABLED", False),
    "cache_dir": os.getenv(
        "FA_AUX_CACHE_DIR",
        os.path.join(REPORT_OUTPUT_DIR, "jianyan", "cache", "auxiliary"),
    ),
    "fast_mode_overrides": {
        "window_sizes": _AUX_FAST_WINDOWS,
        "sample_sizes": _AUX_FAST_SAMPLES,
        "n_iterations": max(1, _env_int("FA_AUX_FAST_ITERATIONS", 40)),
    },
}
