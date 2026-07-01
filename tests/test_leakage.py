"""Leakage guards: no fit/fit_transform ever touches test or OOT (constraint 1)."""

from __future__ import annotations

import copy

import pytest

from creditscorecard.features import binning as binning_mod
from creditscorecard.features.binning import BinningModel
from creditscorecard.features.woe import WoETransformer


def test_binning_never_refits_on_test_or_oot(config, split, monkeypatch):
    counter = {"fits": 0}
    original_fit = binning_mod.OptimalBinning.fit

    def counting_fit(self, x, y=None, **kwargs):
        counter["fits"] += 1
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(binning_mod.OptimalBinning, "fit", counting_fit)

    feat_cols = [
        c for c in split.train.columns if c not in (config.data.target, config.data.date_column)
    ]
    target = config.data.target
    bm = BinningModel(config).fit(split.train[feat_cols], split.train[target])
    fits_after_train = counter["fits"]
    assert fits_after_train == len(feat_cols)  # exactly one fit per characteristic

    # Transforms on test/OOT must NOT trigger any additional fit.
    bm.transform(split.test[feat_cols])
    bm.transform(split.oot[feat_cols])
    assert counter["fits"] == fits_after_train


def test_woe_maps_frozen_after_transform(config, split):
    feat_cols = [
        c for c in split.train.columns if c not in (config.data.target, config.data.date_column)
    ]
    target = config.data.target
    bm = BinningModel(config).fit(split.train[feat_cols], split.train[target])
    woe = WoETransformer(bm).fit(split.train[feat_cols], split.train[target])

    maps_before = copy.deepcopy(woe.woe_maps)
    iv_before = dict(woe.iv)
    woe.transform(split.test[feat_cols])
    woe.transform(split.oot[feat_cols])
    assert woe.woe_maps == maps_before  # transform did not recompute WoE
    assert woe.iv == iv_before


def test_oot_uses_train_woe_values(config, split):
    feat_cols = [
        c for c in split.train.columns if c not in (config.data.target, config.data.date_column)
    ]
    target = config.data.target
    bm = BinningModel(config).fit(split.train[feat_cols], split.train[target])
    woe = WoETransformer(bm).fit(split.train[feat_cols], split.train[target])

    feat = feat_cols[0]
    codes_oot = bm.transform(split.oot[feat_cols])[feat]
    woe_oot = woe.transform(split.oot[feat_cols])[f"woe_{feat}"]
    # Each OOT row's WoE must equal the frozen train map for its code.
    for code, val in zip(codes_oot, woe_oot, strict=True):
        if int(code) in woe.woe_maps[feat]:
            assert val == woe.woe_maps[feat][int(code)]


def test_transform_before_fit_raises(config, split):
    feat_cols = [
        c for c in split.train.columns if c not in (config.data.target, config.data.date_column)
    ]
    bm = BinningModel(config).fit(split.train[feat_cols], split.train[config.data.target])
    woe = WoETransformer(bm)
    with pytest.raises(RuntimeError):
        woe.transform(split.test[feat_cols])
