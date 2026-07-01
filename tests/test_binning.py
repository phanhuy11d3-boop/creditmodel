"""Binning: monotonic WoE, explicit missing/special bins, frozen edges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditscorecard.features.binning import (
    MISSING_CODE,
    OTHER_CODE,
    FeatureBinning,
    assign_codes,
)
from creditscorecard.features.woe import compute_woe_iv


def _is_monotonic(values: list[float]) -> bool:
    inc = all(b >= a for a, b in zip(values, values[1:], strict=False))
    dec = all(b <= a for a, b in zip(values, values[1:], strict=False))
    return inc or dec


def test_woe_monotonic_across_normal_bins(fitted):
    """WoE must be monotonic across ordered normal bins (constraint 3)."""
    codes = fitted.binning.transform(fitted.split.train[fitted.feat_cols])
    y = fitted.ytr
    checked = 0
    for feat in fitted.model.features:
        woe_map, _, table = compute_woe_iv(codes[feat], y)
        normal = table[table["code"] >= 0].sort_values("code")
        if len(normal) >= 2:
            assert _is_monotonic(list(normal["woe"])), f"{feat} WoE not monotonic"
            checked += 1
    assert checked > 0


def test_missing_value_gets_explicit_bin():
    spec = FeatureBinning(name="x", dtype="numerical", splits=[10.0, 20.0], n_bins=3)
    s = pd.Series([5.0, 15.0, 25.0, np.nan])
    codes = assign_codes(spec, s)
    assert list(codes) == [0, 1, 2, MISSING_CODE]


def test_unseen_category_maps_to_other():
    spec = FeatureBinning(name="c", dtype="categorical", groups=[["A", "B"], ["C"]], n_bins=2)
    codes = assign_codes(spec, pd.Series(["A", "C", "ZZZ", None]))
    assert list(codes) == [0, 1, OTHER_CODE, MISSING_CODE]


def test_transform_is_frozen_and_deterministic(fitted):
    X = fitted.split.test[fitted.feat_cols]
    first = fitted.binning.transform(X)
    second = fitted.binning.transform(X)
    pd.testing.assert_frame_equal(first, second)


def test_min_bin_population_respected(fitted):
    """Every normal bin should hold at least the configured minimum share."""
    min_pct = fitted.config.binning.min_bin_pct
    codes = fitted.binning.transform(fitted.split.train[fitted.feat_cols])
    n = len(codes)
    for feat in fitted.model.features:
        counts = codes[feat][codes[feat] >= 0].value_counts()
        assert (counts / n >= min_pct - 1e-9).all(), f"{feat} has an under-populated bin"
