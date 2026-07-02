"""Champion vs challenger (§5.4): DeLong test + interpretability parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.evaluation.benchmark import (
    delong_roc_test,
    fit_challenger,
    run_benchmark,
)
from creditscorecard.evaluation.explainability import jaccard, top_k_features


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.discrimination.bootstrap_iterations = 100
    return c


def test_delong_detects_real_difference():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.uniform(size=n) < 0.3).astype(int)
    # Strong score correlates with y; weak score is near-random.
    strong = y * 0.6 + rng.normal(0, 0.3, n)
    weak = y * 0.05 + rng.normal(0, 1.0, n)
    res = delong_roc_test(y, strong, weak)
    assert res["auc_a"] > res["auc_b"]
    assert res["z"] > 0  # model a (strong) favoured
    assert res["pvalue"] < 0.05


def test_delong_no_difference_for_identical_scores():
    rng = np.random.default_rng(1)
    n = 2000
    y = (rng.uniform(size=n) < 0.4).astype(int)
    p = y * 0.5 + rng.normal(0, 0.5, n)
    res = delong_roc_test(y, p, p)
    assert abs(res["z"]) < 1e-6
    assert res["pvalue"] == pytest.approx(1.0, abs=1e-6)


def test_jaccard_parity():
    assert jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard([], []) == 0.0
    assert top_k_features({"a": 3.0, "b": 1.0, "c": 2.0}, 2) == ["a", "c"]


def test_challenger_beats_linear_on_interaction(cfg):
    """XOR-like interaction: a linear model cannot capture it, GBM can → under-specified."""
    rng = np.random.default_rng(2)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 3.0 * np.sign(x1) * np.sign(x2)  # pure interaction, no main effect
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"woe_x1": x1, "woe_x2": x2})

    split = n // 2
    Xtr, ytr = X.iloc[:split], y[:split]
    Xte, yte = X.iloc[split:], y[split:]

    from sklearn.linear_model import LogisticRegression

    lin = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr, ytr)
    rep_p = lin.predict_proba(Xte)[:, 1]

    challenger = fit_challenger(Xtr, pd.Series(ytr), cfg)
    chal_p = challenger.predict_proba(Xte)[:, 1]

    res = run_benchmark(yte, rep_p, chal_p, cfg)
    assert res.challenger_gini_oot["point"] > res.reportable_gini_oot["point"]
    assert res.delong["pvalue"] < 0.05
    assert res.under_specified is True
    assert "UNDER-SPECIFIED" in res.verdict


def test_delong_degenerate_single_class():
    y = np.zeros(100, dtype=int)  # no positives
    res = delong_roc_test(y, np.random.rand(100), np.random.rand(100))
    assert np.isnan(res["z"]) and np.isnan(res["pvalue"])


def test_fit_challenger_random_forest(cfg):
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"woe_a": rng.normal(size=300), "woe_b": rng.normal(size=300)})
    y = pd.Series((rng.uniform(size=300) < 0.4).astype(int))
    cfg.benchmark.challenger = "random_forest"
    cfg.benchmark.challenger_params = {"n_estimators": 30, "max_depth": 3}
    model = fit_challenger(X, y, cfg)
    assert hasattr(model, "predict_proba")


def test_fit_challenger_xgboost_falls_back_if_absent(cfg):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"woe_a": rng.normal(size=300), "woe_b": rng.normal(size=300)})
    y = pd.Series((rng.uniform(size=300) < 0.4).astype(int))
    cfg.benchmark.challenger = "xgboost"
    cfg.benchmark.challenger_params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}
    model = fit_challenger(X, y, cfg)  # xgboost may be absent → gradient_boosting fallback
    assert hasattr(model, "predict_proba")


def test_save_benchmark(cfg, tmp_path):
    from creditscorecard.evaluation.benchmark import save_benchmark

    rng = np.random.default_rng(5)
    y = (rng.uniform(size=500) < 0.3).astype(int)
    res = run_benchmark(y, rng.uniform(size=500), rng.uniform(size=500), cfg)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    path = save_benchmark(res, cfg)
    assert path.exists()
