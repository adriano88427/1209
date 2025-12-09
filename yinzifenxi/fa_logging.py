# -*- coding: utf-8 -*-
import os
import sys
import re
from datetime import datetime

from .fa_config import build_report_path

try:  # 可选依赖，仅用于中文转拼音，缺失时回退为 unicode 转义
    from pypinyin import lazy_pinyin  # type: ignore
except ImportError:  # pragma: no cover
    lazy_pinyin = None

_PHRASE_REPLACEMENTS = [
    ("因子分析日志", "Factor analysis log"),
    ("日志记录结束", "Log recording finished"),
    ("数据加载成功", "Data load succeeded"),
    ("数据预处理完成", "Data preprocessing complete"),
    ("数据验证通过", "Data validation passed"),
    ("数据完整性验证", "Data integrity validation"),
    ("期望文件数", "Expected file count"),
    ("成功读取", "Successfully read"),
    ("覆盖率", "coverage"),
    ("覆盖年份", "Covered years"),
    ("覆盖交易日", "Covered trading days"),
    ("期望文件", "Expected files"),
    ("逐文件读入情况", "Per-file import summary"),
    ("未配置列对齐规则", "No column alignment rules are configured"),
    ("列对齐检查通过", "Column alignment check passed"),
    ("启动多表格对比验证", "Starting multi-sheet validation"),
    ("多表格对比验证完成", "Multi-sheet validation completed"),
    ("年份覆盖检查", "Year coverage check"),
    ("聚合完成", "Aggregation finished"),
    ("可用样本", "usable samples"),
    ("样本", "samples"),
    ("记录因子", "factors"),
    ("工作表", "worksheet"),
    ("列数", "columns"),
    (" 行", " rows"),
    (" 个工作表", " worksheets"),
    (" 解析诊断", " parsed diagnostics"),
    ("解析", "parsed"),
    ("占比", "ratio"),
    ("个3倍标准差异常值", " three-sigma outliers"),
    ("检测到", "detected"),
    ("分析", "analysis"),
    ("因子", "factor"),
    ("警告", "[WARN]"),
    ("错误", "[ERROR]"),
    ("完成", "completed"),
    ("生成", "generated"),
    ("进度", "progress"),
    ("报告", "report"),
]

_PHRASE_REPLACEMENTS = sorted(_PHRASE_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True)

_PUNCT_TRANSLATIONS = str.maketrans({
    "，": ", ",
    "。": ". ",
    "：": ": ",
    "；": "; ",
    "（": "(",
    "）": ")",
    "、": ", ",
    "％": "%",
    "－": "-",
    "—": "-",
    "·": "-",
})

_CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]+')

_ERROR_KEYWORDS = (
    "错误",
    "出错",
    "失败",
    "[ERROR]",
)

_WARN_KEYWORDS = (
    "警告",
    "[WARN]",
)

_INFO_KEYWORDS = (
    "[INFO]",
    "[STEP]",
    "[OK]",
    "[DEBUG]",
)

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _is_truthy(value: str) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _is_truthy(value)


def detect_debug_enabled(explicit: bool = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return env_flag("FA_DEBUG", False)


def _transliterate_chunk(chunk: str) -> str:
    if lazy_pinyin:
        try:
            return "-".join(lazy_pinyin(chunk, strict=False))
        except Exception:  # pragma: no cover
            pass
    return chunk.encode('unicode_escape').decode('ascii')


def _normalize_chunk(chunk: str) -> str:
    output = chunk
    for cn, en in _PHRASE_REPLACEMENTS:
        if cn in output:
            output = output.replace(cn, en)
    output = output.translate(_PUNCT_TRANSLATIONS)

    def _replace(match):
        return _transliterate_chunk(match.group(0))

    output = _CHINESE_PATTERN.sub(_replace, output)
    # 避免残留的其他非 ASCII 字符（例如 emoji 或希腊字母）
    cleaned = []
    for ch in output:
        if ord(ch) < 128:
            cleaned.append(ch)
        else:
            cleaned.append(ch.encode('unicode_escape').decode('ascii'))
    normalized = "".join(cleaned)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_message(text: str) -> str:
    if not text or not any(ord(ch) > 127 for ch in text):
        return text
    segments = re.split(r'(\r?\n)', text)
    normalized_segments = []
    for segment in segments:
        if segment in ("\r", "\n", "\r\n"):
            normalized_segments.append(segment)
        else:
            normalized_segments.append(_normalize_chunk(segment))
    return "".join(normalized_segments)


class Logger:
    """简单日志记录器：默认仅将错误写入日志，可通过环境变量启用警告。"""

    def __init__(
        self,
        log_file=None,
        log_all=None,
        include_warnings=None,
        debug_enabled=None,
        debug_dump_path=None,
    ):
        """初始化日志记录器

        Args:
            log_file: 日志文件路径，如果为 None 则自动按时间戳生成
            log_all: 是否记录全部输出，默认读取环境变量 FA_LOG_VERBOSE
            include_warnings: 是否记录警告日志，默认读取环境变量 FA_LOG_WARNINGS
            debug_enabled: 是否启用调试模式（开启后默认记录全部输出）
            debug_dump_path: 额外的调试日志镜像文件，未经过滤
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'factor_analysis_log_{timestamp}.txt'

        log_file = build_report_path(log_file)

        self.log_file = log_file
        self.terminal = sys.stdout  # 保存原始终端输出
        env_verbose = os.environ.get("FA_LOG_VERBOSE")
        env_warning = os.environ.get("FA_LOG_WARNINGS")
        self.debug_enabled = detect_debug_enabled(debug_enabled)
        default_log_all = env_verbose == "1"
        if self.debug_enabled:
            default_log_all = True
        self.log_all = log_all if log_all is not None else default_log_all
        if self.debug_enabled:
            default_include_warn = True
        else:
            default_include_warn = env_warning == "1"
        self.include_warnings = (
            include_warnings
            if include_warnings is not None
            else default_include_warn
        )
        self._buffer = ""
        self._debug_stream = None
        self.debug_dump_path = None

        header = normalize_message(f"因子分析日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            f.write("=" * 60 + "\n\n")

        if debug_dump_path:
            self.enable_debug_dump(debug_dump_path)

    def enable_debug_dump(self, dump_path: str):
        """启用未过滤的调试日志镜像输出。"""
        normalized = build_report_path(dump_path)
        os.makedirs(os.path.dirname(normalized), exist_ok=True)
        self.debug_dump_path = normalized
        self._debug_stream = open(normalized, 'w', encoding='utf-8')

    def _line_contains_keyword(self, line: str) -> bool:
        """检查单行是否需要写入到日志。"""
        if self.log_all:
            return True

        content = line.strip()
        if not content:
            return False

        if any(keyword in content for keyword in _ERROR_KEYWORDS):
            return True

        # “异常”一词既可能出现在警告中，也可能表示真正的错误。
        # 仅当不包含“警告”前缀时，才将其视作错误输出。
        if "异常" in content and "警告" not in content:
            return True

        if any(keyword in content for keyword in _INFO_KEYWORDS):
            return True

        if self.include_warnings and any(keyword in content for keyword in _WARN_KEYWORDS):
            return True

        return False

    def _drain_buffer(self, force: bool = False):
        """将缓冲区中带换行的内容输出到日志文件。"""
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._log_line(line.rstrip('\r'))

        if force and self._buffer:
            self._log_line(self._buffer.rstrip('\r'))
            self._buffer = ""

    def _log_line(self, line: str):
        if not self._line_contains_keyword(line):
            return

        normalized = normalize_message(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{normalized}\n")

    def write(self, message):
        """同时输出到终端，并在需要时写入日志文件"""
        normalized = normalize_message(message)
        self.terminal.write(normalized)
        if self._debug_stream:
            self._debug_stream.write(normalized)
        self._buffer += normalized
        self._drain_buffer()

    def flush(self):
        """刷新输出"""
        self.terminal.flush()
        self._drain_buffer()

    def close(self):
        """关闭日志记录器并恢复标准输出"""
        self._drain_buffer(force=True)
        footer = normalize_message(f"日志记录结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{footer}\n")
            f.write("=" * 60 + "\n")

        if self._debug_stream:
            self._debug_stream.flush()
            self._debug_stream.close()
            self._debug_stream = None

        sys.stdout = self.terminal
