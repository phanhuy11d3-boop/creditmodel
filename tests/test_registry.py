"""Registry: deterministic versioning, load errors, and MLflow guard."""

from __future__ import annotations

import numpy as np
import pytest

from creditscorecard.config import load_config
from creditscorecard.registry import compute_version, load_payload, save_artifacts


def _payload():
    return {"selected_features": ["a", "b"], "model": {"intercept": -1.0}, "scaling": {"pdo": 20}}


def test_version_is_deterministic_and_ignores_timestamp():
    p1 = {**_payload(), "created_at": "2020-01-01", "version": "x"}
    p2 = {**_payload(), "created_at": "2099-12-31", "version": "y"}
    assert compute_version(p1) == compute_version(p2)


def test_version_changes_with_content():
    a = compute_version(_payload())
    b = compute_version({**_payload(), "selected_features": ["a"]})
    assert a != b


def test_load_missing_artifacts_raises(tmp_path):
    cfg = load_config("configs/home_credit.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        load_payload(cfg)


def test_save_and_load_roundtrip_with_numpy(tmp_path):
    cfg = load_config("configs/home_credit.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "art")
    payload = {**_payload(), "np_value": np.float64(0.5)}
    save_artifacts(payload, cfg)
    loaded = load_payload(cfg)
    assert loaded["np_value"] == 0.5
    assert loaded["version"] == compute_version(payload)


def test_mlflow_guard_when_not_installed(tmp_path):
    cfg = load_config("configs/home_credit.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "art")
    cfg.tracking.mlflow_enabled = True  # mlflow extra not installed -> graceful skip
    save_artifacts(_payload(), cfg)  # must not raise
    assert load_payload(cfg)["selected_features"] == ["a", "b"]
