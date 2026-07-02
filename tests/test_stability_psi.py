"""PSI/CSI must reuse FROZEN reference bins and never re-bin new data (constraint 11)."""

from __future__ import annotations

import pytest

from creditscorecard.evaluation import stability
from creditscorecard.evaluation.stability import (
    StabilityReference,
    assert_is_reference,
    characteristic_stability_index,
    freeze_reference,
    population_stability_index,
)


@pytest.fixture(scope="module")
def reference(fitted):
    codes = fitted.binning.transform(fitted.split.train[fitted.feat_cols])[fitted.model.features]
    scored = fitted.scorecard.score_codes(codes)
    return freeze_reference(scored["total_score"].to_numpy(), codes, fitted.config), scored, codes


def test_identical_distribution_has_zero_psi(reference):
    ref, scored, _ = reference
    psi = population_stability_index(ref, scored["total_score"].to_numpy())
    assert psi < 1e-6


def test_shift_increases_psi(reference):
    ref, scored, _ = reference
    shifted = scored["total_score"].to_numpy() + 80
    psi = population_stability_index(ref, shifted)
    assert psi > 0.25  # a large shift should trip the ALERT band


def test_psi_status_bands():
    from creditscorecard.evaluation.stability import psi_status

    assert psi_status(0.05, 0.10, 0.25) == "OK"
    assert psi_status(0.15, 0.10, 0.25) == "WARN"
    assert psi_status(0.30, 0.10, 0.25) == "ALERT"


def test_split_psi_identical_and_shifted(reference, fitted):
    from creditscorecard.evaluation.stability import split_psi

    ref, scored, _ = reference
    train_scores = scored["total_score"].to_numpy()
    out = split_psi(ref, {"same": train_scores, "shifted": train_scores + 80}, fitted.config)
    assert out["same"]["psi"] < 1e-6 and out["same"]["status"] == "OK"
    assert out["shifted"]["status"] == "ALERT"


def test_frozen_edges_not_recomputed_on_new_data(reference, monkeypatch):
    """PSI on new data must NOT call quantile/qcut (edges are frozen)."""
    ref, scored, _ = reference
    edges_before = list(ref.score_edges)

    def _boom(*args, **kwargs):  # pragma: no cover - must never be hit
        raise AssertionError("quantile called on new data -> forbidden re-binning")

    monkeypatch.setattr(stability.np, "quantile", _boom)
    # Different range on purpose; edges must still be the frozen ones.
    _ = population_stability_index(ref, scored["total_score"].to_numpy() * 2 + 5)
    assert ref.score_edges == edges_before


def test_csi_detects_characteristic_shift(reference):
    ref, _, codes = reference
    same = characteristic_stability_index(ref, codes)
    assert all(v < 1e-6 for v in same.values())

    feat = next(iter(ref.csi_ref_pct))
    shifted_codes = codes.copy()
    shifted_codes[feat] = 0  # collapse everyone into one bin
    csi = characteristic_stability_index(ref, shifted_codes)
    assert csi[feat] > 0.25


def test_assert_is_reference_guard():
    with pytest.raises(ValueError):
        assert_is_reference(object())  # type: ignore[arg-type]
    bad = StabilityReference(
        score_edges=[1.0], score_ref_pct=[0.5, 0.5], csi_ref_pct={}, developed=False
    )
    with pytest.raises(ValueError):
        assert_is_reference(bad)


def test_reference_roundtrip(reference):
    ref, _, _ = reference
    restored = StabilityReference.from_dict(ref.to_dict())
    assert restored.score_edges == ref.score_edges
    assert restored.csi_ref_pct.keys() == ref.csi_ref_pct.keys()
