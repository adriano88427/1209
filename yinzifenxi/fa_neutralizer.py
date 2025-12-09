# -*- coding: utf-8 -*-
"""Factor neutralization utilities (market-cap & industry)."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .fa_config import NEUTRALIZATION_CONFIG, REPORT_OUTPUT_DIR


@dataclass
class NeutralizationStats:
    factor: str
    mode: str
    total_rows: int
    applied_rows: int
    market_rows: int = 0
    industry_rows: int = 0
    notes: Optional[str] = None
    total_seconds: float = 0.0
    market_seconds: float = 0.0
    industry_seconds: float = 0.0

    @property
    def coverage(self) -> float:
        if self.total_rows <= 0:
            return 0.0
        return float(self.applied_rows) / float(self.total_rows)


class FactorNeutralizer:
    """Encapsulate market-cap & industry neutralization logic."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        self.config = dict(config or NEUTRALIZATION_CONFIG or {})
        self.logger = logger
        self.enabled = bool(self.config.get("enabled", False))
        self.signal_date_column = self.config.get("signal_date_column", "信号日期")
        self.market_cap_column = self.config.get("market_cap_column", "流通市值(元)")
        self.industry_column = self.config.get("industry_column", "所属同花顺行业")
        self.industry_level = max(1, int(self.config.get("industry_level", 2) or 2))
        self.industry_separator = self.config.get("industry_separator", "-")
        self.min_cross_section = int(self.config.get("min_cross_section", 8) or 8)
        self.min_industry_group = int(self.config.get("min_industry_group", 4) or 4)
        self.raw_suffix = self.config.get("store_raw_suffix", "__raw") or "__raw"
        self.default_mode = str(self.config.get("default_method", "none") or "none").lower()
        self.factor_rules = dict(self.config.get("factor_rules") or {})
        self.qa_threshold = float(self.config.get("qa_threshold", 0.6) or 0.6)
        self.qa_output_path = self.config.get("qa_output_path") or os.path.join(
            REPORT_OUTPUT_DIR, "QA_neutralization.csv"
        )

        self._group_indices: Dict[Any, pd.Index] = {}
        self._log_cap_series: Optional[pd.Series] = None
        self._industry_series: Optional[pd.Series] = None
        self.summary: List[NeutralizationStats] = []

    def _log(self, level: str, message: str) -> None:
        prefix = f"[NEUTRAL][{level}] "
        if self.logger is not None and hasattr(self.logger, "write"):
            self.logger.write(prefix + message + "\n")
            if hasattr(self.logger, "flush"):
                self.logger.flush()
        else:
            print(prefix + message)

    def apply(self, df: pd.DataFrame, factors: Sequence[str]) -> List[NeutralizationStats]:
        if not self.enabled:
            return []
        if df is None or df.empty or not factors:
            self._log("WARN", "neutralization skipped: empty data or factor list")
            return []
        if not self._prepare_cache(df):
            return []

        start_ts = time.perf_counter()
        self.summary = []
        for factor in factors:
            if factor not in df.columns:
                continue
            mode = self._resolve_mode(factor)
            if mode == "none":
                continue
            try:
                stats = self._apply_to_factor(df, factor, mode)
                self.summary.append(stats)
                timing_notes = f"total {stats.total_seconds:.2f}s"
                if stats.market_seconds:
                    timing_notes += f", market {stats.market_seconds:.2f}s"
                if stats.industry_seconds:
                    timing_notes += f", industry {stats.industry_seconds:.2f}s"
                self._log(
                    "INFO",
                    (
                        f"{factor} - mode {mode}, coverage {stats.coverage:.1%} "
                        f"(market rows: {stats.market_rows}, industry rows: {stats.industry_rows}) "
                        f"[{timing_notes}]"
                    ),
                )
                if stats.coverage < self.qa_threshold:
                    self._log("WARN", f"factor {factor} coverage {stats.coverage:.1%} below QA threshold")
                self._append_qa_record(stats)
            except Exception as exc:  # pragma: no cover
                self._log("ERROR", f"factor {factor} neutralization failed: {exc}")
        elapsed = time.perf_counter() - start_ts
        self._log(
            "INFO",
            f"neutralized {len(self.summary)} factors in {elapsed:.2f}s "
            f"(signal groups: {len(self._group_indices)})",
        )
        return self.summary

    def _resolve_mode(self, factor: str) -> str:
        mode = self.factor_rules.get(factor, self.default_mode)
        normalized = str(mode or "none").lower()
        if normalized not in {"market_cap", "industry", "both", "none"}:
            self._log("WARN", f"unknown neutralization mode {mode} for {factor}, fallback to none")
            return "none"
        return normalized

    def _prepare_cache(self, df: pd.DataFrame) -> bool:
        if self.signal_date_column not in df.columns:
            self._log("ERROR", f"missing signal date column '{self.signal_date_column}'")
            return False
        grouped = df.groupby(self.signal_date_column)
        self._group_indices = grouped.groups
        if not self._group_indices:
            self._log("WARN", "unable to build signal-date groups; skipped neutralization")
            return False

        market_col = self.market_cap_column
        raw_market_col = f"{market_col}{self.raw_suffix}"
        source_market_col = raw_market_col if raw_market_col in df.columns else market_col
        if source_market_col in df.columns:
            cap_series = pd.to_numeric(df[source_market_col], errors="coerce")
            cap_series = cap_series.where(cap_series > 0)
            self._log_cap_series = np.log(cap_series.clip(lower=1.0))
        else:
            self._log_cap_series = None
            self._log("WARN", f"missing market-cap column '{source_market_col}', market neutral skipped")

        if self.industry_column in df.columns:
            self._industry_series = self._normalize_industry_series(df[self.industry_column])
        else:
            self._industry_series = None
            self._log("WARN", f"missing industry column '{self.industry_column}', industry neutral skipped")
        return True

    def _normalize_industry_series(self, series: pd.Series) -> pd.Series:
        sep = self.industry_separator or "-"
        level = self.industry_level

        def _split(value: Any) -> str:
            if value is None:
                return "Unspecified"
            text = str(value).strip()
            if not text or text.lower() == "nan":
                return "Unspecified"
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            if not parts:
                return "Unspecified"
            level_idx = min(level, len(parts))
            return sep.join(parts[:level_idx])

        normalized = series.astype(str).map(_split)
        return normalized.fillna("Unspecified")

    def _apply_to_factor(self, df: pd.DataFrame, factor: str, mode: str) -> NeutralizationStats:
        total_rows = len(df)
        stats = NeutralizationStats(factor=factor, mode=mode, total_rows=total_rows, applied_rows=0)
        factor_start = time.perf_counter()
        market_time = 0.0
        industry_time = 0.0
        raw_col = f"{factor}{self.raw_suffix}"
        if raw_col not in df.columns:
            df[raw_col] = df[factor]

        working = pd.to_numeric(df[factor], errors="coerce")
        market_rows = 0
        industry_rows = 0

        if mode in ("market_cap", "both"):
            if self._log_cap_series is None:
                self._log("WARN", f"{factor} requires market neutralization but market-cap data is missing")
            else:
                start = time.perf_counter()
                working, market_rows = self._apply_market_cap(working)
                market_time = time.perf_counter() - start

        if mode in ("industry", "both"):
            if self._industry_series is None:
                self._log("WARN", f"{factor} requires industry neutralization but industry data is missing")
            else:
                start = time.perf_counter()
                working, industry_rows = self._apply_industry(working)
                industry_time = time.perf_counter() - start

        df[factor] = working
        stats.applied_rows = int(working.notna().sum())
        stats.market_rows = int(market_rows)
        stats.industry_rows = int(industry_rows)
        stats.market_seconds = market_time
        stats.industry_seconds = industry_time
        stats.total_seconds = time.perf_counter() - factor_start
        return stats

    def _apply_market_cap(self, series: pd.Series) -> (pd.Series, int):
        result = pd.Series(np.nan, index=series.index, dtype=float)
        applied = 0
        for _, idx in self._group_indices.items():
            values = series.loc[idx]
            caps = self._log_cap_series.loc[idx]
            mask = values.notna() & caps.notna()
            if mask.sum() == 0:
                continue
            subset_idx = mask[mask].index
            if len(subset_idx) < 2:
                continue
            values_subset = values.loc[subset_idx].astype(float)
            caps_subset = caps.loc[subset_idx].astype(float)
            if len(subset_idx) < self.min_cross_section or np.isclose(caps_subset.std(ddof=0), 0.0):
                resid = values_subset - values_subset.mean()
            else:
                design = np.column_stack([np.ones(len(caps_subset)), caps_subset])
                try:
                    coef, _, _, _ = np.linalg.lstsq(design, values_subset, rcond=None)
                    fitted = design @ coef
                    resid = values_subset - fitted
                except np.linalg.LinAlgError:
                    resid = values_subset - values_subset.mean()
            result.loc[subset_idx] = resid
            applied += len(subset_idx)
        return result, applied

    def _apply_industry(self, series: pd.Series) -> (pd.Series, int):
        result = pd.Series(np.nan, index=series.index, dtype=float)
        applied = 0
        industries = self._industry_series
        for _, idx in self._group_indices.items():
            values = series.loc[idx]
            industry_slice = industries.loc[idx]
            valid = values.notna()
            if not valid.any():
                continue
            fallback_mean = values[valid].mean()
            grouped = industry_slice.groupby(industry_slice)
            for _, industry_idx in grouped.groups.items():
                current_idx = values.loc[industry_idx].dropna().index
                if not len(current_idx):
                    continue
                subset = values.loc[current_idx].astype(float)
                if len(subset) < self.min_industry_group:
                    centered = subset - fallback_mean
                else:
                    centered = subset - subset.mean()
                result.loc[current_idx] = centered
                applied += len(current_idx)
        return result, applied

    def _append_qa_record(self, stats: NeutralizationStats) -> None:
        if not self.qa_output_path:
            return
        os.makedirs(os.path.dirname(self.qa_output_path), exist_ok=True)
        file_exists = os.path.exists(self.qa_output_path)
        with open(self.qa_output_path, "a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow([
                    "factor",
                    "mode",
                    "total_rows",
                    "applied_rows",
                    "coverage",
                    "market_rows",
                    "industry_rows",
                    "notes",
                ])
            writer.writerow([
                stats.factor,
                stats.mode,
                stats.total_rows,
                stats.applied_rows,
                f"{stats.coverage:.6f}",
                stats.market_rows,
                stats.industry_rows,
                stats.notes or "",
            ])
