"""End-to-end: one command produces every artifact, and runs are reproducible."""

from __future__ import annotations

import json

from creditscorecard.config import Config
from creditscorecard.pipeline import run_pipeline


def test_pipeline_produces_all_artifacts(config: Config, pipeline_payload):
    art = config.artifacts_path()
    assert (art / "model.json").exists()
    for name in ["iv", "metrics", "master_scale"]:
        assert (art / "tables" / f"{name}.csv").exists()

    figs = config.reports_path() / "figures"
    for fig in ["roc_curve.png", "cap_curve.png", "calibration.png", "score_distribution.png"]:
        assert (figs / fig).exists()

    mdd = config.reports_path() / "mdd"
    assert (mdd / "model_development_document.md").exists()
    assert (mdd / "model_development_document.html").exists()


def test_payload_core_contract(pipeline_payload):
    p = pipeline_payload
    for key in [
        "selected_features",
        "binning_specs",
        "woe_maps",
        "points_card",
        "master_scale",
        "stability_reference",
        "scaling",
        "calibration",
        "performance",
        "version",
    ]:
        assert key in p, f"missing artifact key: {key}"
    # Orientation: every coefficient negative; sign check + parity pass.
    assert all(v < 0 for v in p["model"]["coefficients"].values())
    assert p["model"]["sign_ok"] is True
    assert p["model"]["parity_passed"] is True


def test_performance_is_reasonable(pipeline_payload):
    perf = pipeline_payload["performance"]
    assert perf["train"]["gini"] > 0.3
    assert perf["oot"]["gini"] > 0.2  # generalises to out-of-time


def test_run_is_reproducible(config: Config):
    """Same seed/config -> identical model version hash."""
    v1 = run_pipeline(config).payload["version"]
    v2 = run_pipeline(config).payload["version"]
    assert v1 == v2

    stored = json.loads((config.artifacts_path() / "model.json").read_text())
    assert stored["version"] == v2


def test_frozen_reference_persisted(config: Config, pipeline_payload):
    payload = json.loads((config.artifacts_path() / "model.json").read_text())
    ref = payload["stability_reference"]
    assert ref["score_edges"]
    assert len(ref["score_ref_pct"]) == len(ref["score_edges"]) + 1
    assert ref["csi_ref_pct"]
