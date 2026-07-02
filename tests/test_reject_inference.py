"""Reject inference (§5.2): selection-bias recovery + method sensitivity.

Under outcome-correlated approval (a bureau signal seen by the credit officer but not in
the model), the KGB accepts sample has a far lower bad rate than the through-the-door
population, so the KGB **intercept / base rate** is badly biased while the slopes stay
roughly consistent (selection on y shifts the intercept only). Parceling — which injects
inferred bads into the declined region — recovers the population intercept markedly closer
than KGB. This matches the reject-inference literature the module cites (RI mainly corrects
the base rate; discrimination gains are usually modest).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.data.reject_inference import _fit_logit, run_reject_inference


@pytest.fixture
def biased_sample():
    rng = np.random.default_rng(11)
    n = 10000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.2 + 1.8 * x1 + 1.2 * x2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"woe_x1": x1, "woe_x2": x2})
    oracle_int, oracle_coef = _fit_logit(X, y)  # full-population truth
    # Approve based on an outcome-correlated signal not in the model → KGB base-rate bias.
    w = 3.0 * y + rng.normal(0, 1, n)
    accept = w < np.quantile(w, 0.55)
    return {
        "X_accept": X[accept].reset_index(drop=True),
        "y_accept": y[accept],
        "X_reject": X[~accept].reset_index(drop=True),
        "oracle_intercept": oracle_int,
        "oracle_coef": oracle_coef,
        "pop_bad_rate": float(y.mean()),
    }


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.reject_inference.enabled = True
    c.reject_inference.reject_data_path = "dummy.csv"  # satisfies validator; not read here
    c.reject_inference.bad_rate_multiplier = 3.0
    return c


def test_kgb_underestimates_base_rate(biased_sample):
    ya = biased_sample["y_accept"]
    assert ya.mean() < biased_sample["pop_bad_rate"] - 0.2  # accepts far cleaner than population


def test_parceling_recovers_intercept_closer_than_kgb(biased_sample, cfg):
    s = biased_sample
    kgb_int, _ = _fit_logit(s["X_accept"], s["y_accept"])
    res = run_reject_inference(s["X_accept"], s["y_accept"], s["X_reject"], cfg)

    oracle = s["oracle_intercept"]
    kgb_gap = abs(kgb_int - oracle)
    parceling_gap = abs(res.methods["parceling"]["intercept"] - oracle)
    assert parceling_gap < kgb_gap  # reject inference corrects the base-rate bias


def test_all_methods_report_sensitivity(biased_sample, cfg):
    s = biased_sample
    res = run_reject_inference(s["X_accept"], s["y_accept"], s["X_reject"], cfg)
    assert set(res.methods) == {"parceling", "reweighting", "fuzzy_augmentation"}
    for m in res.methods.values():
        assert m["coef_shift_l2"] >= 0.0
        assert "gini_kgb" in m and "gini_method" in m and "gini_shift" in m
        assert set(m["coef"]) == {"woe_x1", "woe_x2"}


def test_single_method_selection(biased_sample, cfg):
    s = biased_sample
    res = run_reject_inference(
        s["X_accept"], s["y_accept"], s["X_reject"], cfg, methods=["reweighting"]
    )
    assert list(res.methods) == ["reweighting"]


def test_save_writes_json(biased_sample, cfg, tmp_path):
    from creditscorecard.data.reject_inference import save_reject_inference_sensitivity

    s = biased_sample
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    res = run_reject_inference(
        s["X_accept"], s["y_accept"], s["X_reject"], cfg, methods=["parceling"]
    )
    path = save_reject_inference_sensitivity(res, cfg)
    assert path.exists()
    d = json.loads(path.read_text(encoding="utf-8"))
    assert "baseline_coef" in d and "methods" in d


def test_gini_is_nan_when_eval_single_class(biased_sample, cfg):
    # Evaluation labels all one class → Gini undefined → reported as NaN, not a crash.
    s = biased_sample
    eval_X = s["X_accept"].iloc[:50]
    eval_y = np.zeros(50, dtype=int)
    res = run_reject_inference(
        s["X_accept"],
        s["y_accept"],
        s["X_reject"],
        cfg,
        eval_X=eval_X,
        eval_y=eval_y,
        methods=["fuzzy_augmentation"],
    )
    assert np.isnan(res.methods["fuzzy_augmentation"]["gini_method"])
