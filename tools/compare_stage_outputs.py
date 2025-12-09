# -*- coding: utf-8 -*-
"""
对比阶段输出与 baseline 的哈希，用于快速验证拆分后的脚本行为。

默认比较所有 CSV 文件，以及名称以“带参数”“辅助分析报告”“因子分析详情_精简版”
开头的 TXT 报告（其余 TXT 包含大量时间戳信息，不做二进制对比）。
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

TXT_TIMESTAMP_PREFIXES = (
    "带参数",
    "辅助分析报告",
    "因子分析详情_精简版",
)
TXT_IGNORED_PREFIXES = ("生成时间:",)
_VARIANT_CACHE: Dict[Path, Optional[str]] = {}


def _strip_timestamp(name: str) -> Tuple[str, str]:
    """
    将文件名拆分为“前缀 + 扩展名”，并移除末尾的 _YYYYMMDD_HHMMSS。
    若不匹配，则直接返回原始 stem。
    """
    stem, dot, ext = name.rpartition(".")
    if dot:
        suffix = f".{ext}"
        base = stem
    else:
        base = name
        suffix = ""

    if len(base) > 16 and base[-15:-7].isdigit() and base[-6:].isdigit():
        # 匹配 *_YYYYMMDD_HHMMSS
        return base[:-16], suffix
    return base, suffix


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix.lower() == ".txt" and path.name.startswith(TXT_TIMESTAMP_PREFIXES):
        text = path.read_text(encoding="utf-8")
        filtered = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(TXT_IGNORED_PREFIXES):
                continue
            filtered.append(line)
        data = "\n".join(filtered).encode("utf-8")
        digest.update(data)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _should_compare(file: Path) -> bool:
    if file.suffix.lower() == ".csv":
        return True
    if file.suffix.lower() == ".txt" and file.name.startswith(TXT_TIMESTAMP_PREFIXES):
        return True
    return False


def _detect_variant(path: Path) -> Optional[str]:
    """
    对需要区分模式的文件返回 variant 标签，目前用于区分
    因子分析详情_精简版 的“完整版/摘要版”。
    """
    if path in _VARIANT_CACHE:
        return _VARIANT_CACHE[path]

    variant: Optional[str] = None
    if path.suffix.lower() == ".txt" and path.name.startswith("因子分析详情_精简版"):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(2048)
        except Exception:
            snippet = ""
        if "摘要模式" in snippet:
            variant = "summary"
        else:
            variant = "full"

    _VARIANT_CACHE[path] = variant
    return variant


def _find_baseline(stage_file: Path, baseline_dir: Path, stage_variant: Optional[str] = None) -> Optional[Path]:
    exact_match = baseline_dir / stage_file.name
    if exact_match.exists():
        return exact_match

    prefix, suffix = _strip_timestamp(stage_file.name)
    candidates = sorted(baseline_dir.glob(f"{prefix}_*{suffix}"))
    if stage_variant and candidates:
        matched = [c for c in candidates if _detect_variant(c) == stage_variant]
        if matched:
            return matched[-1]
    return candidates[-1] if candidates else None


def compare(stage_dir: Path, baseline_dir: Path) -> int:
    stage_files = sorted(p for p in stage_dir.iterdir() if p.is_file() and _should_compare(p))
    if not stage_files:
        print("未找到需要比对的文件")
        return 1

    exit_code = 0
    for file in stage_files:
        stage_variant = _detect_variant(file)
        baseline_file = _find_baseline(file, baseline_dir, stage_variant=stage_variant)
        if not baseline_file:
            print(f"[MISS] {file.name} -> baseline 缺失")
            exit_code = 1
            continue

        stage_hash = _hash_file(file)
        baseline_hash = _hash_file(baseline_file)
        status = "Match" if stage_hash == baseline_hash else "Mismatch"
        print(f"{status:<8} {file.name} -> {baseline_file.name}")
        if status != "Match":
            exit_code = 1
    return exit_code


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="比较阶段输出与 baseline 的 SHA256 哈希。")
    parser.add_argument("stage_dir", type=Path, help="需要比对的阶段目录，例如 stage_param_analysis_wiring")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("baseline_outputs"),
        help="baseline 目录（默认为项目根目录下的 baseline_outputs）",
    )
    args = parser.parse_args(argv)

    if not args.stage_dir.is_dir():
        print(f"阶段目录不存在: {args.stage_dir}")
        return 1
    if not args.baseline.is_dir():
        print(f"baseline 目录不存在: {args.baseline}")
        return 1

    return compare(args.stage_dir, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
