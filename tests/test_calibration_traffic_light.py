"""Calibration backtest (§5.5): traffic light green on calibrated, red on miscalibrated."""

from __future__ import annotations

import numpy as np
import pytest

from creditscorecard.config import load_config
from creditscorecard.evaluation.calibration import (
    binomial_upper_pvalue,
    brier_score,
    build_grade_aggregates,
    compute_calibration_backtest,
    expected_calibration_error,
    jeffreys_interval,
    per_grade_backtest,
)


@pytest.fixture
def cfg():
    return load_config("configs/german_credit.yaml")


def _grades_pd(n_per_grade: int, grade_pds: dict[str, float], factor: float, seed: int = 0):
    """Build (grades, y, pd_hat) with the observed default count set *exactly* to
    ``forecast_pd × factor × n`` per grade.

    Deterministic construction (no sampling noise) so a calibrated sample (factor=1)
    always reads green and a doubled-rate sample (factor=2) always reads red — the
    stochastic version has a ~5% per-grade false-yellow rate that would make the test
    flaky.
    """
    grades, ys, pds = [], [], []
    for g, pd_g in grade_pds.items():
        true_rate = min(pd_g * factor, 0.999)
        n_def = int(round(true_rate * n_per_grade))
        y = [1] * n_def + [0] * (n_per_grade - n_def)
        grades += [g] * n_per_grade
        ys += y
        pds += [pd_g] * n_per_grade
    return np.array(grades), np.array(ys), np.array(pds)


def test_calibrated_sample_all_green(cfg):
    grade_pds = {"A": 0.02, "B": 0.05, "C": 0.10, "D": 0.20}
    grades, y, pd_hat = _grades_pd(2000, grade_pds, factor=1.0, seed=0)
    aggs = build_grade_aggregates(grades, y, pd_hat)
    rows = per_grade_backtest(aggs, cfg)
    lights = {r.grade: r.traffic_light for r in rows}
    assert all(v == "green" for v in lights.values()), lights


def test_miscalibrated_sample_turns_red(cfg):
    # Observed default rate is ~2x the forecast PD → severe under-prediction → red.
    grade_pds = {"A": 0.02, "B": 0.05, "C": 0.10, "D": 0.20}
    grades, y, pd_hat = _grades_pd(2000, grade_pds, factor=2.0, seed=1)
    aggs = build_grade_aggregates(grades, y, pd_hat)
    rows = per_grade_backtest(aggs, cfg)
    reds = [r.grade for r in rows if r.traffic_light == "red"]
    assert len(reds) >= 2  # multiple grades flagged


def test_jeffreys_interval_bounds():
    lo, hi = jeffreys_interval(k=5, n=100, alpha=0.05)
    assert 0.0 < lo < 0.05 < hi < 0.2
    # Degenerate ends clamp to [0, 1].
    assert jeffreys_interval(0, 50, 0.05)[0] == 0.0
    assert jeffreys_interval(50, 50, 0.05)[1] == 1.0


def test_binomial_pvalue_small_when_underpredicted():
    # 40 defaults observed where 10 expected (n=100, pd=0.10) → tiny p-value.
    assert binomial_upper_pvalue(40, 100, 0.10) < 1e-6
    # Observed == expected → p-value near 0.5+.
    assert binomial_upper_pvalue(10, 100, 0.10) > 0.3


def test_overall_metrics_and_full_result(cfg):
    grade_pds = {"A": 0.02, "B": 0.10, "C": 0.30}
    grades, y, pd_hat = _grades_pd(1500, grade_pds, factor=1.0, seed=2)
    res = compute_calibration_backtest(y, pd_hat, grades, cfg)
    assert 0.0 <= res.brier <= 0.25
    assert res.ece >= 0.0
    assert 0.0 < res.hhi_grades <= 1.0
    assert len(res.per_grade) == 3
    assert res.brier == pytest.approx(brier_score(y, pd_hat))
    assert res.ece == pytest.approx(
        expected_calibration_error(y, pd_hat, cfg.calibration_extended.reliability_curve_bins)
    )
