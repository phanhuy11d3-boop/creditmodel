"""Calibration & accuracy checks: MAPE by grade, anchor/CT gap, curve-shape band.

These extend the discrimination (AUC/Gini/KS) and stability (PSI/CSI/HHI)
metrics with checks specific to whether *predicted PD levels* are trustworthy,
not just whether the model ranks applicants correctly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def mape_by_grade(table: list[dict]) -> float:
    """Mean Absolute Percentage Error between avg predicted PD and observed bad
    rate across Master Scale grades (rows with ``avg_pd`` and
    ``observed_bad_rate``, weighted equally per grade, empty grades skipped).
    """
    errors = []
    for row in table:
        obs = row.get("observed_bad_rate")
        pred = row.get("avg_pd")
        if row.get("count", 0) <= 0 or obs is None or pred is None or obs == 0:
            continue
        errors.append(abs(obs - pred) / obs)
    return float(np.mean(errors)) if errors else 0.0


def anchor_gap(mean_pd: float, anchor_rate: float) -> float:
    """Relative gap between the calibrated portfolio PD and the anchor/central
    tendency default rate (TTC/PIT anchor). ~0 means calibration is on target.
    """
    if anchor_rate == 0:
        return 0.0
    return float((mean_pd - anchor_rate) / anchor_rate)


@dataclass
class GradeBandCheck:
    grade: str
    count: int
    avg_pd: float
    observed_bad_rate: float
    std_error: float
    within_band: bool


@dataclass
class CurveShapeResult:
    monotonic: bool
    n_se: float
    bands: list[GradeBandCheck]

    @property
    def all_within_band(self) -> bool:
        return all(b.within_band for b in self.bands)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def curve_shape_check(table: list[dict], n_se: float = 2.0) -> CurveShapeResult:
    """Validate the calibration curve: rank-ordering monotonicity of avg PD
    across grades, plus a per-grade check that the observed bad rate falls
    within ``n_se`` binomial standard errors of the average predicted PD.

    ``std_error = sqrt(p*(1-p)/n)`` using the predicted PD as ``p`` (Wald SE).
    """
    bands: list[GradeBandCheck] = []
    pds: list[float] = []
    for row in table:
        n = int(row.get("count", 0))
        if n <= 0:
            continue
        pred = float(row["avg_pd"])
        obs = float(row["observed_bad_rate"])
        se = float(np.sqrt(max(pred * (1 - pred), 0.0) / n))
        within = abs(obs - pred) <= n_se * se if se > 0 else np.isclose(obs, pred)
        bands.append(
            GradeBandCheck(
                grade=str(row["grade"]),
                count=n,
                avg_pd=pred,
                observed_bad_rate=obs,
                std_error=se,
                within_band=bool(within),
            )
        )
        pds.append(pred)

    # Grades are ordered worst -> best in the Master Scale table; PD must be
    # monotonically non-increasing along that order.
    monotonic = all(b <= a + 1e-12 for a, b in zip(pds, pds[1:], strict=False))
    return CurveShapeResult(monotonic=monotonic, n_se=n_se, bands=bands)
