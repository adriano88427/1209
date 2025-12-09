#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全A报告期数据补充脚本

目标：
1. 遍历 biaoge 目录下所有“回测详细数据结合股本分析：创业板xxxx.xlsx”文件
2. 根据年份自动匹配 quanAshuju/全A{yyyy-1}年年报关键数据.xlsx
3. 按股票代码补充配置指定列，未匹配的单元格填入“无数据”
4. 写回前创建备份，写回后执行原列完整性检查与新增列抽样校验
5. 全程记录日志，便于调试
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import pandas as pd


# ---------------------------------------------
# 路径 & 全局常量
# ---------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../shuju/biaoge
TARGET_DIR = BASE_DIR
SOURCE_DIR = BASE_DIR / "quanAshuju"
LOG_DIR = BASE_DIR / "logs"
BACKUP_DIR = BASE_DIR / "backup"
LOG_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILL_VALUE = "无数据"
SOURCE_YEAR_OFFSET = 1  # 2025 -> 找 2024 年报
LOG_FILE = LOG_DIR / f"full_a_merge_{datetime.now():%Y%m%d_%H%M%S}.log"
DEFAULT_SOURCE_SUFFIX = "\n{year}.12.31"
DEFAULT_TARGET_SUFFIX = "_{year}.12.31"

# 待补充列配置（可按需增删）
# - 若 source/target 直接给出，则按原样使用
# - 若使用 source_base/target_base，则根据年份拼接 suffix（默认 \n{year}.12.31 / _{year}.12.31）
COLUMN_MAPPING: List[Dict[str, object]] = [
    {"source": "所属同花顺行业", "target": "所属同花顺行业"},
    {
        "source_base": "机构持股市值(元)",
        "target_base": "机构持股市值(元)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "前十大流通股东持股市值(报告期)(元)",
        "target_base": "前十大流通股东持股市值(报告期)(元)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "前十大流通股东持股数量合计(报告期)(股)",
        "target_base": "前十大流通股东持股数量合计(报告期)(股)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "总股本(股)",
        "target_base": "总股本(股)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "总市值(元)",
        "target_base": "总市值(元)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "机构持股数量(股)",
        "target_base": "机构持股数量(股)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "number",
    },
    {
        "source_base": "户均持股比例季度增长率(%)",
        "target_base": "户均持股比例季度增长率(%)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "percent",
    },
    {
        "source_base": "户均持股数季度增长率(%)",
        "target_base": "户均持股数季度增长率(%)",
        "target_suffix": "",
        "coerce_numeric": True,
        "format": "percent",
    },
]


def resolve_runtime_mapping(column_mapping: Sequence[Dict[str, object]], year: int) -> List[Dict[str, object]]:
    runtime: List[Dict[str, object]] = []
    for cfg in column_mapping:
        source = cfg.get("source")
        target = cfg.get("target")
        source_base = cfg.get("source_base")
        target_base = cfg.get("target_base")
        source_suffix = cfg.get("source_suffix", DEFAULT_SOURCE_SUFFIX if source_base else "")
        target_suffix = cfg.get(
            "target_suffix",
            DEFAULT_TARGET_SUFFIX if (target_base or source_base) else "",
        )

        if not source and source_base:
            suffix = source_suffix.format(year=year) if source_suffix else ""
            source = f"{source_base}{suffix}"
        if not target:
            base = target_base or source_base or source
            if not base:
                raise ValueError(f"映射配置缺少 target 定义: {cfg}")
            suffix = target_suffix.format(year=year) if target_suffix else ""
            target = f"{base}{suffix}"

        runtime.append(
            {
                "source": source,
                "target": target,
                "coerce_numeric": cfg.get("coerce_numeric", False),
                "insert_after": cfg.get("insert_after"),
                "format": cfg.get("format", "number"),
                "skip_if_target_missing": cfg.get("skip_if_target_missing", True),
            }
        )
    return runtime


# ---------------------------------------------
# 基础工具
# ---------------------------------------------
def setup_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("full_a_merge")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def extract_year(name: str) -> Optional[int]:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    return None


def normalize_code(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = text.replace(".", "").replace("-", "")
    if text.endswith("SZ") or text.endswith("SH"):
        text = text[:-2]
    if text.startswith("SZ") or text.startswith("SH"):
        text = text[2:]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits.zfill(6)


def load_excel(path: Path, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(path)
    sheet = sheet_name or xls.sheet_names[0]
    df = xls.parse(sheet)
    return df, sheet


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "股票代码" not in df.columns:
        raise KeyError(f"缺少 '股票代码' 列: {df.columns}")
    df["__code_key"] = df["股票代码"].apply(normalize_code)
    return df


def find_source_file(year: int) -> Optional[Path]:
    pattern = f"全A{year}年年报关键数据"
    for file in SOURCE_DIR.glob("*.xlsx"):
        if pattern in file.name:
            return file
    return None


def backup_target(path: Path, logger: logging.Logger) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year = extract_year(path.name) or "unknown"
    dst_dir = BACKUP_DIR / str(year)
    dst_dir.mkdir(parents=True, exist_ok=True)
    backup_path = dst_dir / f"{path.stem}_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    logger.info("Backup saved: %s", backup_path)
    return backup_path


# ---------------------------------------------
# 核心逻辑
# ---------------------------------------------
def fill_columns(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    mapping: Sequence[Dict[str, object]],
    fill_value: str,
    logger: logging.Logger,
    original_columns: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    stats: Dict[str, Dict[str, int]] = {}
    source_indexed = source_df.set_index("__code_key")
    result = target_df.copy()
    original_columns = original_columns or set(target_df.columns)
    for cfg in mapping:
        src_col = str(cfg.get("source"))
        tgt_col = str(cfg.get("target", src_col))
        insert_after = cfg.get("insert_after")
        coerce_numeric = bool(cfg.get("coerce_numeric"))
        data_format = str(cfg.get("format", "number")).lower()
        skip_if_missing_target = bool(cfg.get("skip_if_target_missing", True))

        column_existed = tgt_col in original_columns
        if tgt_col not in result.columns:
            if insert_after and insert_after in result.columns:
                pos = result.columns.get_loc(insert_after) + 1
                result.insert(pos, tgt_col, fill_value)
            else:
                result[tgt_col] = fill_value

        if (not column_existed) and skip_if_missing_target:
            result[tgt_col] = fill_value
            stats[tgt_col] = {"filled": 0, "missing": len(result)}
            logger.info("[SKIP] %s 缺少原始列，已全部填充为默认值", tgt_col)
            continue

        filled = 0
        missing = 0
        for idx, key in result["__code_key"].items():
            if key is None or key not in source_indexed.index:
                result.at[idx, tgt_col] = fill_value
                missing += 1
                continue
            if src_col not in source_indexed.columns:
                result.at[idx, tgt_col] = fill_value
                missing += 1
                continue
            value = source_indexed.at[key, src_col]
            if pd.isna(value):
                result.at[idx, tgt_col] = fill_value
                missing += 1
                continue
            if coerce_numeric:
                try:
                    value = pd.to_numeric(value)
                except Exception:
                    logger.debug("Column %s value %s cannot convert to numeric", tgt_col, value)
            formatted_value = apply_format(value, data_format)
            result.at[idx, tgt_col] = formatted_value
            filled += 1
        stats[tgt_col] = {"filled": filled, "missing": missing}
        logger.info("[FILL] %s filled=%s missing=%s", tgt_col, filled, missing)
    return result, stats


def apply_format(value: object, fmt: str) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return DEFAULT_FILL_VALUE
    if fmt == "percent":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return DEFAULT_FILL_VALUE
        return numeric / 100.0
    if fmt == "number":
        return value
    return value


def _to_float(value: object) -> Optional[float]:
    if value is None or value == DEFAULT_FILL_VALUE:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def verify_original_columns(
    original: pd.DataFrame,
    written_path: Path,
    columns: Sequence[str],
    logger: logging.Logger,
) -> None:
    new_df, _ = load_excel(written_path)
    new_df = prepare_df(new_df)
    sample_size = min(5, len(original))
    original_sample = original[columns].head(sample_size)
    new_sample = new_df[columns].head(sample_size)
    diff = original_sample.compare(new_sample)
    if diff.empty:
        logger.info("[VERIFY] 原有列完整性检查通过（采样 %s 行）", sample_size)
    else:
        logger.error("[VERIFY] 原有列存在差异：\n%s", diff)


def verify_new_columns(
    enriched: pd.DataFrame,
    source_df: pd.DataFrame,
    mapping: Sequence[Dict[str, object]],
    logger: logging.Logger,
    samples: int = 5,
    original_columns: Optional[Set[str]] = None,
) -> None:
    indexed_source = source_df.set_index("__code_key")
    sample_rows = enriched.dropna(subset=["__code_key"]).head(samples)
    original_columns = original_columns or set()
    for _, row in sample_rows.iterrows():
        key = row["__code_key"]
        if key not in indexed_source.index:
            logger.warning("[VERIFY] 样本 %s 无源数据", key)
            continue
        for cfg in mapping:
            src_col = str(cfg.get("source"))
            tgt_col = str(cfg.get("target", src_col))
            data_format = str(cfg.get("format", "number")).lower()
            skip_if_missing_target = bool(cfg.get("skip_if_target_missing", True))
            column_existed = bool(cfg.get("_target_present", tgt_col in original_columns))
            if skip_if_missing_target and not column_existed:
                continue
            src_value = indexed_source.at[key, src_col] if src_col in indexed_source.columns else None
            tgt_value = row.get(tgt_col)
            if pd.isna(src_value):
                src_value = None
            if pd.isna(tgt_value):
                tgt_value = None
            expected = apply_format(src_value, data_format)
            if tgt_value is None:
                tgt_value = DEFAULT_FILL_VALUE
            if data_format == "number":
                expected_num = _to_float(expected)
                target_num = _to_float(tgt_value)
                if expected_num is not None and target_num is not None:
                    if math.isclose(expected_num, target_num, rel_tol=1e-9, abs_tol=1e-6):
                        logger.debug("[VERIFY] 样本 %s 列 %s 校验一致", key, tgt_col)
                        continue
            if expected == DEFAULT_FILL_VALUE and tgt_value == DEFAULT_FILL_VALUE:
                continue
            if str(expected) != str(tgt_value):
                logger.warning("[VERIFY] 样本 %s 列 %s 源=%s 目标=%s", key, tgt_col, expected, tgt_value)
            else:
                logger.debug("[VERIFY] 样本 %s 列 %s 校验一致", key, tgt_col)


def write_enriched_excel(path: Path, df: pd.DataFrame, sheet_name: str, logger: logging.Logger) -> None:
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(prefix=f"{path.stem}_tmp_", suffix=path.suffix, dir=str(path.parent))
        os.close(temp_fd)
        with pd.ExcelWriter(temp_path, engine="openpyxl", mode="w") as writer:
            df.drop(columns=["__code_key"]).to_excel(writer, sheet_name=sheet_name, index=False)
        for attempt in range(3):
            try:
                os.replace(temp_path, path)
                logger.info("File written: %s", path)
                break
            except PermissionError as exc:
                if attempt == 2:
                    logger.error("写入 %s 时权限受限，请关闭该文件后重试: %s", path, exc)
                    raise
                logger.warning("目标文件被占用，%s 秒后重试... (%s/3)", 2, attempt + 1)
                time.sleep(2)
        else:
            raise PermissionError(f"Unable to replace {path}")  # pragma: no cover
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def process_target(path: Path, logger: logging.Logger) -> None:
    logger.info("Processing target: %s", path)
    target_year = extract_year(path.name)
    if not target_year:
        logger.error("无法从文件名提取年份：%s", path.name)
        return
    source_year = target_year - SOURCE_YEAR_OFFSET
    source_path = find_source_file(source_year)
    if not source_path:
        logger.error("未找到对应全A数据文件：%s", source_year)
        return

    target_df, sheet_name = load_excel(path)
    source_df, _ = load_excel(source_path)
    target_df = prepare_df(target_df)
    source_df = prepare_df(source_df)

    runtime_mapping = resolve_runtime_mapping(COLUMN_MAPPING, source_year)
    original_columns = [c for c in target_df.columns if c != "__code_key"]
    original_snapshot = target_df[original_columns].copy()

    original_column_list = list(original_columns)
    original_column_set = set(original_columns)
    enriched_df, stats = fill_columns(
        target_df,
        source_df,
        runtime_mapping,
        DEFAULT_FILL_VALUE,
        logger,
        original_columns=original_column_set,
    )
    backup_path = backup_target(path, logger)

    write_enriched_excel(path, enriched_df, sheet_name, logger)

    verify_original_columns(original_snapshot, path, original_column_list, logger)
    verify_new_columns(enriched_df, source_df, runtime_mapping, logger, original_columns=original_column_set)

    for col, info in stats.items():
        logger.info("[STAT] %s filled=%s missing=%s", col, info["filled"], info["missing"])
    logger.info("Backup file: %s", backup_path)


# ---------------------------------------------
# CLI
# ---------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="全A报告期数据补充工具")
    parser.add_argument(
        "--target-pattern",
        default="回测详细数据结合股本分析：创业板*.xlsx",
        help="目标文件 glob 模式（默认：回测详细数据结合股本分析：创业板*.xlsx）",
    )
    parser.add_argument("--debug", action="store_true", help="输出调试日志")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件")
    args = parser.parse_args(argv)

    logger = setup_logger(args.debug)
    logger.info("Target dir: %s", TARGET_DIR)
    logger.info("Source dir: %s", SOURCE_DIR)

    files = sorted(TARGET_DIR.glob(args.target_pattern))
    if not files:
        logger.warning("No target files matched pattern: %s", args.target_pattern)
        return

    processed = 0
    for path in files:
        try:
            process_target(path, logger)
            processed += 1
            if args.limit and processed >= args.limit:
                break
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Processing failed for %s: %s", path, exc)
    logger.info("All done. processed=%s", processed)


if __name__ == "__main__":
    main()
