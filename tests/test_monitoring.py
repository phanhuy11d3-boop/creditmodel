"""Monitoring runner: PSI/CSI vs frozen reference with warn/alert escalation."""

from __future__ import annotations

import numpy as np

from creditscorecard.monitoring.monitor import _status, run_monitoring


def test_status_thresholds():
    assert _status(0.05, 0.10, 0.25) == "OK"
    assert _status(0.15, 0.10, 0.25) == "WARN"
    assert _status(0.40, 0.10, 0.25) == "ALERT"


def test_monitor_stable_on_same_distribution(config, pipeline_payload, dataset):
    report = run_monitoring(config, dataset.copy())
    assert report.psi_status == "OK"
    assert report.escalate is False
    assert (config.reports_path() / "monitoring_report.json").exists()


def test_monitor_escalates_on_shift(config, pipeline_payload, dataset):
    shifted = dataset.copy()
    # Push everyone toward high risk to move the score distribution hard.
    shifted["credit_amount"] = shifted["credit_amount"] * 3 + 5000
    shifted["duration_months"] = np.clip(shifted["duration_months"] * 2, 4, 120)
    shifted["checking_status"] = "A11"
    report = run_monitoring(config, shifted)
    assert report.psi > config.monitoring.psi_warn
    assert report.psi_status in {"WARN", "ALERT"}
