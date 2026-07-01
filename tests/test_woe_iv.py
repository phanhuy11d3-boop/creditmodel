"""WoE orientation (ln %Good/%Bad) and exact IV numerics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from creditscorecard.features.woe import compute_woe_iv


def _controlled_sample():
    # bin 0: 80 Good / 20 Bad  ; bin 1: 20 Good / 80 Bad  (target 1 == Bad)
    codes = pd.Series([0] * 100 + [1] * 100)
    y = pd.Series([1] * 20 + [0] * 80 + [1] * 80 + [0] * 20)
    return codes, y


def test_woe_orientation_positive_for_good_heavy_bin():
    codes, y = _controlled_sample()
    woe_map, _, _ = compute_woe_iv(codes, y)
    assert woe_map[0] > 0  # Good-heavy bin -> positive WoE
    assert woe_map[1] < 0  # Bad-heavy bin -> negative WoE


def test_woe_value_exact():
    codes, y = _controlled_sample()
    woe_map, _, _ = compute_woe_iv(codes, y)
    # WoE_0 = ln((80/100)/(20/100)) = ln 4
    assert math.isclose(woe_map[0], math.log(4.0), rel_tol=1e-9)
    assert math.isclose(woe_map[1], math.log(0.25), rel_tol=1e-9)


def test_iv_value_exact():
    codes, y = _controlled_sample()
    _, iv, _ = compute_woe_iv(codes, y)
    expected = (0.8 - 0.2) * math.log(4.0) + (0.2 - 0.8) * math.log(0.25)
    assert math.isclose(iv, expected, rel_tol=1e-9)


def test_woe_matches_manual_formula_on_real_feature(fitted):
    """Our WoE equals ln(%Good/%Bad) computed independently."""
    feat = fitted.model.features[0]
    codes = fitted.binning.transform(fitted.split.train[fitted.feat_cols])[feat]
    y = fitted.ytr
    woe_map, _, _ = compute_woe_iv(codes, y)
    df = pd.DataFrame({"c": codes.to_numpy(), "bad": np.asarray(y)})
    df["good"] = 1 - df["bad"]
    g = df.groupby("c").agg(good=("good", "sum"), bad=("bad", "sum"))
    manual = np.log((g["good"] / g["good"].sum()) / (g["bad"] / g["bad"].sum()))
    for code, val in manual.items():
        if g.loc[code, "good"] > 0 and g.loc[code, "bad"] > 0:  # no smoothing on these
            assert math.isclose(woe_map[int(code)], val, rel_tol=1e-9)


def test_iv_frame_sorted_descending(fitted):
    frame = fitted.woe.iv_frame()
    assert list(frame["iv"]) == sorted(frame["iv"], reverse=True)
