"""Multi-period monitoring run-log + trend / AvE / migration analytics (refactor §5.7).

Monitoring must be a *time series*, not a single snapshot (diagnostic §3 DEMOLISH rule).
This module provides:

* **Run-log store** — SQLite (default) or JSONL append-only log; one row per metric per
  period: ``{run_id, run_date, period_id, metric_name, value, threshold_warn,
  threshold_alert, status}``.
* **PSI/CSI trend** — after ``psi_history_min_periods`` periods, fit the slope of a metric
  over time and flag a *rising* trend even when no single period breaches the alert.
* **AvE backtest** — per period per grade, compare expected (forecast PD) vs actual default
  rate with the per-grade Jeffreys traffic light (reusing :mod:`evaluation.calibration`).
* **Score migration matrix** — rating-grade transitions between consecutive periods and the
  off-diagonal (migration) mass.

CLI: ``scorecard monitor --new-data … --period-id …`` appends; ``scorecard monitor-report``
reads the log and writes the trend report.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

_COLUMNS = [
    "run_id",
    "run_date",
    "period_id",
    "metric_name",
    "value",
    "threshold_warn",
    "threshold_alert",
    "status",
]


class RunLog:
    """Append-only monitoring store (SQLite or JSONL)."""

    def __init__(self, config: Config) -> None:
        self.backend = config.monitoring_extended.runlog_backend
        self.path = config._resolve(config.monitoring_extended.runlog_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS runlog (
                    run_id TEXT, run_date TEXT, period_id TEXT, metric_name TEXT,
                    value REAL, threshold_warn REAL, threshold_alert REAL, status TEXT
                )"""
            )

    def append(self, period_id: str, rows: list[dict]) -> str:
        """Append one run's metric rows; returns the generated ``run_id``."""
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_date = datetime.now(UTC).isoformat()
        records = [
            {
                "run_id": run_id,
                "run_date": run_date,
                "period_id": period_id,
                "metric_name": r["metric_name"],
                "value": float(r["value"]),
                "threshold_warn": float(r.get("threshold_warn", float("nan"))),
                "threshold_alert": float(r.get("threshold_alert", float("nan"))),
                "status": r.get("status", ""),
            }
            for r in rows
        ]
        if self.backend == "sqlite":
            cols = ", ".join(_COLUMNS)
            placeholders = ", ".join(["?"] * len(_COLUMNS))
            with sqlite3.connect(self.path) as con:
                con.executemany(
                    f"INSERT INTO runlog ({cols}) VALUES ({placeholders})",
                    [tuple(rec[c] for c in _COLUMNS) for rec in records],
                )
        else:  # jsonl
            with self.path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
        logger.info(
            "Run-log appended %d rows for period %s (run %s).", len(records), period_id, run_id
        )
        return run_id

    def fetch(self) -> pd.DataFrame:
        """Return the full run-log as a DataFrame (empty frame if nothing logged)."""
        if self.backend == "sqlite":
            if not self.path.exists():
                return pd.DataFrame(columns=_COLUMNS)
            with sqlite3.connect(self.path) as con:
                return pd.read_sql_query("SELECT * FROM runlog", con)
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        rows = [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line
        ]
        return pd.DataFrame(rows, columns=_COLUMNS)


# --------------------------------------------------------------------------- #
# Trend detection
# --------------------------------------------------------------------------- #
@dataclass
class TrendResult:
    metric_name: str
    n_periods: int
    slope: float
    latest_value: float
    rising: bool
    breached_alert: bool

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def metric_trend(log: pd.DataFrame, metric_name: str, min_periods: int) -> TrendResult | None:
    """Fit the slope of ``metric_name`` over ordered periods; flag a rising trend.

    ``rising`` is True when there are ≥ ``min_periods`` periods and the OLS slope over the
    period sequence is positive — this catches gradual drift *before* any single period
    breaches the alert threshold.
    """
    sub = log[log["metric_name"] == metric_name]
    if sub.empty:
        return None
    series = sub.sort_values("period_id").groupby("period_id", sort=True)["value"].mean()
    n = len(series)
    values = series.to_numpy(dtype=float)
    if n < min_periods:
        rising = False
        slope = 0.0
    else:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])
        rising = slope > 0
    alert = sub["threshold_alert"].dropna()
    alert_thr = float(alert.iloc[0]) if not alert.empty else float("inf")
    breached = bool(values[-1] > alert_thr)
    return TrendResult(
        metric_name=metric_name,
        n_periods=n,
        slope=slope,
        latest_value=float(values[-1]),
        rising=rising and n >= min_periods,
        breached_alert=breached,
    )


# --------------------------------------------------------------------------- #
# AvE backtest + score migration
# --------------------------------------------------------------------------- #
def ave_backtest(
    grades: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    pd_forecast: np.ndarray | pd.Series,
    config: Config,
) -> list[dict]:
    """Actual-vs-Expected per grade for one period, with per-grade Jeffreys traffic light."""
    from creditscorecard.evaluation.calibration import build_grade_aggregates, per_grade_backtest

    aggregates = build_grade_aggregates(grades, y_true, pd_forecast)
    return [b.to_dict() for b in per_grade_backtest(aggregates, config)]


def score_migration_matrix(
    prev_grades: pd.Series | np.ndarray, curr_grades: pd.Series | np.ndarray
) -> tuple[pd.DataFrame, float]:
    """Per-account grade transition matrix (row-normalised) + off-diagonal mass.

    Requires the two periods to be aligned by account (same order/index).
    """
    prev = pd.Series(np.asarray(prev_grades), name="prev").astype(str)
    curr = pd.Series(np.asarray(curr_grades), name="curr").astype(str)
    grades = sorted(set(prev) | set(curr))
    mat = pd.crosstab(prev, curr).reindex(index=grades, columns=grades, fill_value=0)
    row_sums = mat.sum(axis=1).replace(0, 1)
    normed = mat.div(row_sums, axis=0)
    off_diagonal = float(1.0 - np.trace(normed.to_numpy()) / len(grades))
    return normed, off_diagonal
