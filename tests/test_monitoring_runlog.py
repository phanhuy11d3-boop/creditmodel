"""Multi-period monitoring (§5.7): run-log store, trend detection, AvE, migration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.monitoring.runlog import (
    RunLog,
    ave_backtest,
    metric_trend,
    score_migration_matrix,
)


@pytest.fixture
def cfg(tmp_path):
    c = load_config("configs/german_credit.yaml")
    c.monitoring_extended.runlog_path = str(tmp_path / "runlog.db")
    c.monitoring_extended.psi_history_min_periods = 3
    return c


def _append_psi_series(cfg, values, warn=0.10, alert=0.25):
    log = RunLog(cfg)
    for i, v in enumerate(values, start=1):
        log.append(
            f"2025-P{i}",
            [
                {
                    "metric_name": "score_psi",
                    "value": v,
                    "threshold_warn": warn,
                    "threshold_alert": alert,
                    "status": "OK",
                }
            ],
        )
    return log


def test_runlog_append_and_fetch_sqlite(cfg):
    log = _append_psi_series(cfg, [0.02, 0.03])
    df = log.fetch()
    assert len(df) == 2
    assert set(df["period_id"]) == {"2025-P1", "2025-P2"}


def test_runlog_jsonl_backend(tmp_path):
    c = load_config("configs/german_credit.yaml")
    c.monitoring_extended.runlog_backend = "jsonl"
    c.monitoring_extended.runlog_path = str(tmp_path / "runlog.jsonl")
    log = RunLog(c)
    log.append("2025-P1", [{"metric_name": "score_psi", "value": 0.05}])
    assert len(log.fetch()) == 1


def test_rising_trend_flagged_even_when_each_period_under_alert(cfg):
    # Three periods of drift, every value below the 0.25 alert, but monotonically rising.
    _append_psi_series(cfg, [0.05, 0.12, 0.20])
    tr = metric_trend(RunLog(cfg).fetch(), "score_psi", min_periods=3)
    assert tr.slope > 0
    assert tr.rising is True  # gradual drift caught before any single breach
    assert tr.breached_alert is False  # 0.20 < 0.25


def test_trend_not_flagged_below_min_periods(cfg):
    _append_psi_series(cfg, [0.05, 0.20])  # only 2 periods, min is 3
    tr = metric_trend(RunLog(cfg).fetch(), "score_psi", min_periods=3)
    assert tr.rising is False


def test_ave_backtest_flags_miscalibrated_grade(cfg):
    # Grade C observed default rate is far above its forecast PD → RED.
    n = 1000
    grades = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    pd_forecast = np.array([0.02] * n + [0.10] * n + [0.10] * n)
    y = np.concatenate(
        [
            (np.arange(n) < 0.02 * n).astype(int),  # A calibrated
            (np.arange(n) < 0.10 * n).astype(int),  # B calibrated
            (np.arange(n) < 0.35 * n).astype(int),  # C: 35% actual vs 10% forecast → breach
        ]
    )
    rows = ave_backtest(grades, y, pd_forecast, cfg)
    lights = {r["grade"]: r["traffic_light"] for r in rows}
    assert lights["C"] == "red"
    assert lights["A"] == "green"


def test_monitor_writes_runlog_and_trend_report(config, pipeline_payload, dataset, tmp_path):
    """End-to-end: three monitoring runs populate the run-log; monitor-report reads it."""
    from creditscorecard.monitoring.monitor import run_monitoring, run_monitoring_report

    config.monitoring_extended.runlog_path = str(tmp_path / "rl.db")
    for pid in ["2025-Q1", "2025-Q2", "2025-Q3"]:
        run_monitoring(config, dataset.copy(), period_id=pid)
    report = run_monitoring_report(config)
    assert report["n_periods"] == 3
    assert "score_psi" in report["trends"]
    assert (config.reports_path() / "monitoring_trend_report.json").exists()


def test_score_migration_matrix_off_diagonal():
    prev = pd.Series(["A", "A", "B", "B"])
    curr = pd.Series(["A", "B", "B", "C"])
    mat, off_diag = score_migration_matrix(prev, curr)
    assert mat.shape == (3, 3)  # grades A, B, C
    assert 0.0 <= off_diag <= 1.0
    # Two of four accounts migrated → non-zero off-diagonal mass.
    assert off_diag > 0
