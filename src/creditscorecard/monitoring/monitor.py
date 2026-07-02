"""Post-deployment monitoring: PSI/CSI vs the frozen reference with escalation.

Reuses the development-frozen reference bins (constraint 11); new data is scored
with frozen edges and never re-binned.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from creditscorecard.config import Config
from creditscorecard.evaluation.stability import (
    StabilityReference,
    characteristic_stability_index,
    herfindahl_hirschman_index,
    population_stability_index,
)
from creditscorecard.logging import get_logger
from creditscorecard.scoring import ScoringModel

logger = get_logger(__name__)


@dataclass
class MonitoringReport:
    version: str
    n_new: int
    psi: float
    psi_status: str
    csi: dict[str, float] = field(default_factory=dict)
    csi_status: dict[str, str] = field(default_factory=dict)
    hhi: float = 0.0
    hhi_status: str = "OK"
    escalate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _status(value: float, warn: float, alert: float) -> str:
    if value > alert:
        return "ALERT"
    if value > warn:
        return "WARN"
    return "OK"


def run_monitoring(
    config: Config,
    new_data: str | Path | pd.DataFrame,
    period_id: str | None = None,
) -> MonitoringReport:
    model = ScoringModel.from_config(config)
    reference = StabilityReference.from_dict(model.payload["stability_reference"])

    df = new_data if isinstance(new_data, pd.DataFrame) else pd.read_csv(new_data)
    logger.info("Monitoring on %d new records (model version=%s).", len(df), model.version)

    scored = model.score_frame(df)
    codes = model.codes_frame(df)

    warn = config.monitoring.psi_warn
    alert = config.monitoring.psi_alert

    psi = population_stability_index(reference, scored["total_score"].to_numpy())
    psi_status = _status(psi, warn, alert)

    csi = characteristic_stability_index(reference, codes)
    csi_status = {f: _status(v, warn, alert) for f, v in csi.items()}

    hhi_max = config.validation.hhi_max
    hhi = herfindahl_hirschman_index(scored["rating_grade"])
    # HHI is a concentration ceiling, not a two-tier warn/alert band; ALERT above the ceiling.
    hhi_status = "ALERT" if hhi > hhi_max else "OK"

    escalate = (
        psi_status == "ALERT"
        or any(s == "ALERT" for s in csi_status.values())
        or hhi_status == "ALERT"
    )
    report = MonitoringReport(
        version=model.version,
        n_new=len(df),
        psi=psi,
        psi_status=psi_status,
        csi=csi,
        csi_status=csi_status,
        hhi=hhi,
        hhi_status=hhi_status,
        escalate=escalate,
    )

    log = logger.error if escalate else (logger.warning if psi_status != "OK" else logger.info)
    log("Score PSI=%.4f [%s]; escalate=%s", psi, psi_status, escalate)
    _save_report(report, config)
    if period_id is not None:
        _append_runlog(config, period_id, report, warn, alert, hhi_max)
    return report


def _append_runlog(
    config: Config,
    period_id: str,
    report: MonitoringReport,
    warn: float,
    alert: float,
    hhi_max: float,
) -> None:
    """Append this monitoring run's metrics to the multi-period run-log (§5.7)."""
    from creditscorecard.monitoring.runlog import RunLog

    rows = [
        {
            "metric_name": "score_psi",
            "value": report.psi,
            "threshold_warn": warn,
            "threshold_alert": alert,
            "status": report.psi_status,
        },
        {
            "metric_name": "grade_hhi",
            "value": report.hhi,
            "threshold_alert": hhi_max,
            "status": report.hhi_status,
        },
    ]
    rows += [
        {
            "metric_name": f"csi_{feat}",
            "value": val,
            "threshold_warn": warn,
            "threshold_alert": alert,
            "status": report.csi_status[feat],
        }
        for feat, val in report.csi.items()
    ]
    RunLog(config).append(period_id, rows)


def run_monitoring_report(config: Config) -> dict:
    """Read the run-log and produce a trend report over all logged periods (§5.7).

    Flags a *rising* PSI/CSI trend once ``psi_history_min_periods`` periods exist, even if no
    single period has breached its alert threshold.
    """
    from creditscorecard.monitoring.runlog import RunLog, metric_trend

    log = RunLog(config).fetch()
    min_periods = config.monitoring_extended.psi_history_min_periods
    trends: dict[str, dict] = {}
    for metric in sorted(log["metric_name"].unique()) if not log.empty else []:
        tr = metric_trend(log, metric, min_periods)
        if tr is not None:
            trends[metric] = tr.to_dict()
    n_periods = int(log["period_id"].nunique()) if not log.empty else 0
    rising = [m for m, t in trends.items() if t["rising"]]
    report = {
        "n_periods": n_periods,
        "min_periods_for_trend": min_periods,
        "trends": trends,
        "rising_trends": rising,
        "escalate": bool(rising or any(t["breached_alert"] for t in trends.values())),
    }
    out_dir = config.reports_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monitoring_trend_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    logger.info(
        "Monitoring trend report: %d periods, rising trends=%s (saved %s).",
        n_periods,
        rising,
        path.name,
    )
    return report


def _save_report(report: MonitoringReport, config: Config) -> Path:
    out_dir = config.reports_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monitoring_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    logger.info("Saved monitoring report to %s", path)
    return path
