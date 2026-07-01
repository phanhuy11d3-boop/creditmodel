"""Scorecard scaling: exact Factor/Offset/points formulas and score<->PD consistency."""

from __future__ import annotations

import math

import numpy as np

from creditscorecard.features.selection import woe_col
from creditscorecard.model.scorecard import compute_factor_offset, compute_points


def test_factor_offset_exact():
    factor, offset = compute_factor_offset(pdo=20, target_score=600, target_odds=50)
    assert math.isclose(factor, 20.0 / math.log(2.0), rel_tol=1e-12)
    assert math.isclose(offset, 600.0 - factor * math.log(50.0), rel_tol=1e-12)
    # Guard against silent formula drift with a hardcoded numeric value.
    assert math.isclose(factor, 28.85390081777927, abs_tol=1e-9)
    assert math.isclose(offset, 487.1228762045055, abs_tol=1e-6)


def test_points_formula_exact():
    factor, offset = compute_factor_offset(20, 600, 50)
    pts = compute_points(woe=0.5, beta=-0.8, alpha=-1.0, n=4, factor=factor, offset=offset)
    # points = -(WoE*beta + alpha/n)*Factor + Offset/n
    expected = -(0.5 * -0.8 + (-1.0) / 4) * factor + offset / 4
    assert math.isclose(pts, expected, rel_tol=1e-12)
    assert math.isclose(pts, 140.5357545826829, abs_tol=1e-6)


def test_score_pd_consistency(fitted):
    """PD recovered from the total score equals the calibrated model PD."""
    codes = fitted.binning.transform(fitted.split.train[fitted.feat_cols])[fitted.model.features]
    scored = fitted.scorecard.score_codes(codes)

    alpha = fitted.calibration.calibrated_intercept(fitted.model)
    eta = np.full(len(codes), alpha)
    for f in fitted.model.features:
        eta = eta + fitted.model.coefficients[f] * fitted.Xtr_woe[woe_col(f)].to_numpy()
    pd_expected = 1.0 / (1.0 + np.exp(-eta))
    np.testing.assert_allclose(scored["pd"].to_numpy(), pd_expected, rtol=1e-9, atol=1e-9)


def test_master_scale_monotonic_pd(fitted):
    table = fitted.scorecard.master_scale.table
    pds = [row["avg_pd"] for row in table if row["count"] > 0]
    # grades run worst->best (G..A); avg_pd must decrease monotonically.
    assert all(b <= a + 1e-9 for a, b in zip(pds, pds[1:], strict=False))


def test_points_increase_with_woe():
    """Higher WoE (better applicant) yields more points (beta negative)."""
    factor, offset = compute_factor_offset(20, 600, 50)
    low = compute_points(woe=-1.0, beta=-0.7, alpha=-1.0, n=3, factor=factor, offset=offset)
    high = compute_points(woe=1.0, beta=-0.7, alpha=-1.0, n=3, factor=factor, offset=offset)
    assert high > low
