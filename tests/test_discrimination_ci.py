"""Discrimination with uncertainty (§5.3): CI coverage on known-AUC synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from creditscorecard.config import load_config
from creditscorecard.evaluation.discrimination import (
    auc,
    bootstrap_ci,
    compute_discrimination,
    gini,
    gini_stability_verdict,
    intervals_overlap,
    ks,
    optimism_632plus_auc,
    partial_auc,
    somers_d,
)


@pytest.fixture
def two_gaussians():
    """Two well-separated Gaussian scores → analytic AUC ≈ 0.94.

    For P(Bad) score ~ N(mu, 1) with class means separated by d, AUC = Phi(d/sqrt(2)).
    d = 2.2 → AUC = Phi(1.556) ≈ 0.940.
    """
    rng = np.random.default_rng(0)
    n = 4000
    good = rng.normal(0.0, 1.0, n)
    bad = rng.normal(2.2, 1.0, n)
    scores = np.concatenate([good, bad])
    # squash to (0,1) so it reads as a probability; monotone → AUC unchanged.
    p = 1 / (1 + np.exp(-scores))
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return y, p


def test_point_auc_matches_analytic(two_gaussians):
    y, p = two_gaussians
    assert auc(y, p) == pytest.approx(0.94, abs=0.02)
    assert gini(y, p) == pytest.approx(2 * 0.94 - 1, abs=0.04)
    assert somers_d(y, p) == pytest.approx(gini(y, p), abs=1e-9)  # identical for binary


def test_bootstrap_ci_covers_point(two_gaussians):
    y, p = two_gaussians
    for method in ("percentile", "basic", "bca"):
        ci = bootstrap_ci(y, p, auc, n_iter=300, level=0.95, method=method, seed=7)
        assert ci.lower <= ci.point <= ci.upper
        assert ci.upper - ci.lower < 0.05  # tight CI on 8k rows
        assert ci.lower <= 0.94 <= ci.upper


def test_bootstrap_is_seeded_deterministic(two_gaussians):
    y, p = two_gaussians
    a = bootstrap_ci(y, p, ks, n_iter=200, level=0.95, method="percentile", seed=3)
    b = bootstrap_ci(y, p, ks, n_iter=200, level=0.95, method="percentile", seed=3)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_partial_auc_in_range(two_gaussians):
    y, p = two_gaussians
    pauc = partial_auc(y, p, (0.0, 0.4))
    assert 0.5 <= pauc <= 1.0


def test_optimism_correction_reduces_auc():
    # An overfit-prone tiny sample: corrected AUC should not exceed apparent.
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({f"woe_{i}": rng.normal(size=n) for i in range(4)})
    y = pd.Series((rng.uniform(size=n) < 0.3).astype(int))
    out = optimism_632plus_auc(X, y, list(X.columns), n_iter=50, seed=2)
    assert out["corrected_auc"] <= out["apparent_auc"] + 1e-9
    assert 0.0 <= out["weight_632plus"] <= 1.0 or out["weight_632plus"] >= 0.632


def test_intervals_overlap():
    assert intervals_overlap((0.4, 0.5), (0.45, 0.6)) is True
    assert intervals_overlap((0.4, 0.5), (0.5, 0.6)) is True  # touching endpoints
    assert intervals_overlap((0.4, 0.5), (0.55, 0.7)) is False


def test_gini_stability_verdict_overlap_and_disjoint():
    # Overlapping CIs → within sampling noise.
    per_split = {
        "train": {"gini": {"point": 0.50, "lower": 0.46, "upper": 0.54}},
        "oot": {"gini": {"point": 0.47, "lower": 0.43, "upper": 0.51}},
    }
    v = gini_stability_verdict(per_split)
    assert v["ci_overlap"] is True
    assert "sampling noise" in v["verdict"]
    assert v["point_drop"] == pytest.approx(0.03)

    # Disjoint CIs → significant degradation.
    per_split["oot"]["gini"] = {"point": 0.30, "lower": 0.25, "upper": 0.35}
    v2 = gini_stability_verdict(per_split)
    assert v2["ci_overlap"] is False
    assert "significant" in v2["verdict"]


def test_compute_discrimination_shapes(two_gaussians):
    y, p = two_gaussians
    cfg = load_config("configs/german_credit.yaml")
    cfg.discrimination.bootstrap_iterations = 100
    res = compute_discrimination({"test": (y, p)}, cfg)
    entry = res.per_split["test"]
    assert set(entry["auc"]) >= {"point", "lower", "upper", "method", "level"}
    assert len(entry["lift_gains"]) == 10
    assert entry["lift_gains"][0]["lift"] >= entry["lift_gains"][-1]["lift"]  # top decile riskier
