"""Feature selection: IV filter (drop/flag), iterative VIF, forward selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditscorecard.features.selection import (
    forward_select,
    iterative_vif,
    iv_filter,
    woe_col,
)


def test_iv_filter_drops_low_and_flags_suspicious(config):
    iv = {"good": 0.10, "weak": 0.01, "leaky": 0.90}
    kept, dropped, suspicious, table = iv_filter(iv, config)
    assert "weak" in dropped  # IV < iv_min
    assert "good" in kept and "leaky" in kept  # suspicious is FLAGGED, not dropped
    assert suspicious == ["leaky"]
    assert set(table["feature"]) == {"good", "weak", "leaky"}


def test_iterative_vif_drops_collinear_feature():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    df = pd.DataFrame(
        {
            "a": base + rng.normal(scale=0.01, size=500),
            "b": base + rng.normal(scale=0.01, size=500),  # near-duplicate of a
            "c": rng.normal(size=500),
        }
    )
    cols, dropped, final = iterative_vif(df, threshold=5.0)
    assert len(dropped) >= 1
    assert all(v <= 5.0 for v in final.values())
    assert "c" in cols


def test_forward_select_picks_predictive_feature(config):
    rng = np.random.default_rng(1)
    n = 800
    signal = rng.normal(size=n)
    y = pd.Series((signal + rng.normal(scale=0.5, size=n) > 0).astype(int))
    X = pd.DataFrame(
        {
            woe_col("signal"): signal,
            woe_col("noise"): rng.normal(size=n),
        }
    )
    selected, trail = forward_select(X, y, [woe_col("signal"), woe_col("noise")], config)
    assert woe_col("signal") in selected
    assert trail[0][0] == woe_col("signal")  # strongest added first


def test_run_selection_returns_nonempty(fitted):
    assert fitted.selection.selected_features
    assert all(isinstance(f, str) for f in fitted.selection.selected_features)
