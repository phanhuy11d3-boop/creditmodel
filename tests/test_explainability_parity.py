"""Explainability (§5.8): exact linear SHAP, reason parity, global importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditscorecard.evaluation.explainability import (
    explain_applicant,
    linear_shap_row,
    points_vs_shap_agreement,
    reportable_global_importance,
    woe_means,
)


def test_linear_shap_is_exact():
    coef = {"a": -0.5, "b": 0.3}
    woe = {"a": 1.0, "b": -2.0}
    means = {"a": 0.2, "b": 0.1}
    shap = linear_shap_row(coef, woe, means)
    assert shap["a"] == pytest.approx(-0.5 * (1.0 - 0.2))
    assert shap["b"] == pytest.approx(0.3 * (-2.0 - 0.1))


def test_local_shap_additivity_matches_logodds():
    # Sum of SHAP contributions + base = model log-odds deviation from base.
    coef = {"a": -0.5, "b": 0.3, "c": 0.9}
    means = {"a": 0.0, "b": 0.0, "c": 0.0}
    woe = {"a": 1.0, "b": 2.0, "c": -1.0}
    shap = linear_shap_row(coef, woe, means)
    contribution = sum(shap.values())
    expected = sum(coef[f] * woe[f] for f in coef)  # means are 0 → base offset 0
    assert contribution == pytest.approx(expected)


def test_explain_applicant_orders_by_magnitude_and_direction():
    coef = {"a": -0.5, "b": 0.3, "c": 0.9}
    means = {"a": 0.0, "b": 0.0, "c": 0.0}
    woe = {"a": 1.0, "b": 2.0, "c": -1.0}  # |shap|: a .5, b .6, c .9 → order c,b,a
    reasons = explain_applicant(coef, woe, means, top_n=2)
    assert [r["feature"] for r in reasons] == ["c", "b"]
    assert reasons[0]["direction"] == "decreases_risk"  # c: 0.9 * -1 = -0.9 (<0)
    assert reasons[1]["direction"] == "increases_risk"  # b: 0.3 * 2 = 0.6 (>0)


def test_points_vs_shap_agreement():
    assert points_vs_shap_agreement(["a", "b"], ["b", "c"]) == pytest.approx(0.5)
    assert points_vs_shap_agreement(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert np.isnan(points_vs_shap_agreement([], ["a"]))


def test_reportable_global_importance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"woe_a": rng.normal(size=200), "woe_b": rng.normal(size=200)})
    coef = {"a": -1.0, "b": 0.1}
    means = woe_means(X, ["a", "b"])
    imp = reportable_global_importance(coef, X, ["a", "b"])
    # Feature a has a 10x larger coefficient → larger mean|SHAP|.
    assert imp["a"] > imp["b"]
    assert imp["a"] == pytest.approx(
        float(np.mean(np.abs(-1.0 * (X["woe_a"] - means["a"])))), rel=1e-9
    )


def test_challenger_importance_shap_and_fallback():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from creditscorecard.evaluation.explainability import challenger_global_importance

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"woe_a": rng.normal(size=300), "woe_b": rng.normal(size=300)})
    y = (rng.uniform(size=300) < 0.4).astype(int)

    gbm = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
    imp, matrix = challenger_global_importance(gbm, X, sample_size=100, seed=0)
    assert set(imp) == {"woe_a", "woe_b"} and matrix is not None

    # A linear model is not tree-based → TreeExplainer fails → impurity fallback.
    lin = LogisticRegression(max_iter=500).fit(X, y)
    imp2, matrix2 = challenger_global_importance(lin, X, sample_size=100, seed=0)
    assert matrix2 is None  # no SHAP matrix on the fallback path


def test_plot_and_save_global_importance(tmp_path):
    from creditscorecard.config import load_config
    from creditscorecard.evaluation.explainability import (
        ExplainabilityResult,
        plot_shap_summary,
        save_global_importance,
    )

    cfg = load_config("configs/german_credit.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    res = ExplainabilityResult(
        reportable_importance={"a": 0.3, "b": 0.1},
        challenger_importance={"a": 0.2, "b": 0.4},
        interpretability_parity={"jaccard": 0.5},
    )
    fig = plot_shap_summary(res, tmp_path / "figs")  # bar-chart path (no SHAP matrix)
    assert fig.exists()
    art = save_global_importance(res, cfg)
    assert art.exists()
