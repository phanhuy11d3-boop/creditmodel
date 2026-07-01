"""Governance metadata & model card (§5.9)."""

from __future__ import annotations

import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.governance.metadata import (
    KGB_LIMITATION,
    build_model_card,
    collect_assumptions,
    dataset_hash,
    save_model_card,
)


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.data.adapter = "synthetic"
    c.data.target = "default"
    return c


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "default": [0, 1, 0]})


def test_required_fields_present_and_non_null(cfg, frame):
    card = build_model_card(cfg, frame)
    assert card.governance["model_id"]
    assert card.provenance["git_sha"]  # non-empty ('unknown' outside a work tree is acceptable)
    assert card.dataset_hash
    assert card.provenance["run_timestamp"]
    assert card.provenance["seed"] == cfg.seed
    assert set(card.provenance["package_versions"]) >= {"numpy", "pandas", "scikit-learn"}


def test_kgb_limitation_when_reject_inference_disabled(cfg, frame):
    cfg.reject_inference.enabled = False
    _, limitations = collect_assumptions(cfg)
    assert KGB_LIMITATION in limitations


def test_no_kgb_limitation_when_reject_inference_enabled(cfg, frame):
    cfg.reject_inference.enabled = True
    cfg.reject_inference.reject_data_path = "some/path.csv"
    _, limitations = collect_assumptions(cfg)
    assert KGB_LIMITATION not in limitations


def test_dataset_hash_is_deterministic(frame):
    assert dataset_hash(frame) == dataset_hash(frame.copy())


def test_dataset_hash_changes_with_content(frame):
    other = frame.copy()
    other.loc[0, "a"] = 999
    assert dataset_hash(frame) != dataset_hash(other)


def test_artifact_hashes_stable_across_runs(cfg, frame, tmp_path):
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    # Two cards built over the same on-disk artifacts must agree on hashes + dataset hash.
    (tmp_path / "artifacts").mkdir(parents=True)
    (tmp_path / "artifacts" / "model.json").write_text('{"x": 1}', encoding="utf-8")
    c1 = build_model_card(cfg, frame, artifacts_dir=cfg.artifacts_path())
    c2 = build_model_card(cfg, frame, artifacts_dir=cfg.artifacts_path())
    assert c1.artifact_hashes == c2.artifact_hashes
    assert c1.dataset_hash == c2.dataset_hash


def test_save_model_card_excludes_itself_from_hashes(cfg, frame, tmp_path):
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    card = build_model_card(cfg, frame, artifacts_dir=cfg.artifacts_path())
    path = save_model_card(card, cfg)
    assert path.exists()
    # The card's own file is not one of the hashed artifacts.
    assert "model_card.json" not in card.artifact_hashes
