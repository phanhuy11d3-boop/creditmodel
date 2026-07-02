"""Fairness (§5.6): AIR flags disparate impact; proxy scan identifies a planted proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.evaluation.fairness import (
    FairnessBuildError,
    protected_group_mask,
    run_fairness,
)


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.fairness.protected_attributes = ["grp"]
    c.fairness.acknowledge_failure = True
    return c


@pytest.fixture
def disparate_data():
    """Protected group with a much lower favourable rate + a planted proxy feature."""
    rng = np.random.default_rng(0)
    n = 2000
    grp_protected = rng.uniform(size=n) < 0.4  # minority-ish protected group
    grp = np.where(grp_protected, "B", "A")  # 'B' is the minority → protected
    # Favourable (approved) far less often for the protected group → AIR well below 0.8.
    fav_prob = np.where(grp_protected, 0.30, 0.80)
    favourable = rng.uniform(size=n) < fav_prob
    scores = np.where(grp_protected, rng.normal(480, 30, n), rng.normal(560, 30, n))
    y = (rng.uniform(size=n) < np.where(grp_protected, 0.4, 0.2)).astype(int)
    features = pd.DataFrame(
        {
            "proxy": grp_protected.astype(float) + rng.normal(0, 0.15, n),  # strong proxy
            "noise": rng.normal(size=n),  # unrelated
        }
    )
    data = pd.DataFrame({"grp": grp})
    return data, favourable, scores, y, features


def test_protected_group_mask_numeric_age():
    s = pd.Series([20, 24, 25, 40, 60])
    mask = protected_group_mask(s, "age_years")
    assert mask.tolist() == [True, True, False, False, False]  # < 25 dichotomy


def test_protected_group_mask_categorical_minority():
    s = pd.Series(["A"] * 90 + ["B"] * 10)
    mask = protected_group_mask(s, "foreign_worker")
    assert mask.sum() == 10  # minority 'B' is the protected group


def test_air_flags_disparate_impact(cfg, disparate_data):
    data, favourable, scores, y, features = disparate_data
    res = run_fairness(data, favourable, scores, cfg, y_true=y, feature_frame=features)
    attr = res.attributes[0]
    assert attr["adverse_impact_ratio"] < 0.80  # breaches the 80% rule
    assert attr["air_status"] == "ALERT"
    assert attr["standardized_mean_difference"] < 0  # protected scores lower


def test_proxy_scan_flags_planted_proxy(cfg, disparate_data):
    data, favourable, scores, y, features = disparate_data
    res = run_fairness(data, favourable, scores, cfg, y_true=y, feature_frame=features)
    scan = res.proxies["grp"]
    flagged = {s["feature"] for s in scan if s["flagged"]}
    assert "proxy" in flagged
    assert "noise" not in flagged


def test_build_fails_when_air_breach_not_acknowledged(cfg, disparate_data):
    data, favourable, scores, y, features = disparate_data
    cfg.fairness.acknowledge_failure = False
    with pytest.raises(FairnessBuildError):
        run_fairness(data, favourable, scores, cfg, y_true=y, feature_frame=features)


def test_disabled_or_missing_attrs_is_noop(cfg, disparate_data):
    data, favourable, scores, y, features = disparate_data
    cfg.fairness.protected_attributes = ["not_present"]
    res = run_fairness(data, favourable, scores, cfg, y_true=y, feature_frame=features)
    assert res.attributes == []
    assert "N/A" in res.note or "not" in res.note.lower()
