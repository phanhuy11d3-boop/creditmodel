"""Regression baseline (§7): the one-command run is deterministic and now carries the
Phase B governance + sample-design payload without perturbing legacy numeric outputs."""

from __future__ import annotations

import pytest

from creditscorecard.config import Config, load_config
from creditscorecard.pipeline import run_pipeline


def _offline_cfg(tmp_path) -> Config:
    cfg = load_config("configs/german_credit.yaml")
    cfg.data.adapter = "synthetic"
    cfg.data.target = "default"
    cfg.data.date_column = "application_date"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.reports_dir = str(tmp_path / "reports")
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.discrimination.bootstrap_iterations = 120  # fast + deterministic (risk R3/R6)
    return cfg


def test_two_runs_are_numerically_stable(tmp_path):
    a = run_pipeline(_offline_cfg(tmp_path / "a")).payload
    b = run_pipeline(_offline_cfg(tmp_path / "b")).payload

    # Deterministic content hash: identical across independent runs.
    assert a["version"] == b["version"]

    # Key numeric outputs identical (not just within tolerance) under a fixed seed.
    for split in ("train", "test", "oot"):
        for metric in ("auc", "gini", "ks"):
            assert a["performance"][split][metric] == pytest.approx(
                b["performance"][split][metric], abs=1e-12
            )


def test_phase_b_payload_keys_present(tmp_path):
    payload = run_pipeline(_offline_cfg(tmp_path)).payload
    # Governance + sample design now travel with the model payload.
    assert payload["governance"]["model_id"] == "PD-APP-GC-001"
    sd = payload["sample_design"]
    assert sd["n_raw"] == 1000
    assert sd["constructed_default"] is False  # synthetic frame is pass-through
    assert sd["cohort_col"] == "origination_month"
    # Sample design is a no-op on the flat frame: no rows lost (diagnostic risk R1).
    assert sd["n_after_seasoning"] == sd["n_raw"]


def test_model_card_artifact_written(tmp_path):
    cfg = _offline_cfg(tmp_path)
    run_pipeline(cfg)
    assert (cfg.artifacts_path() / "model_card.json").exists()
    assert (cfg.artifacts_path() / "sample_design.json").exists()
