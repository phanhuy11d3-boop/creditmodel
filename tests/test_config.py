"""Config loading, deep-merge, and fail-fast validation."""

from __future__ import annotations

import pytest

from creditscorecard.config import load_config


def test_base_and_named_merge():
    cfg = load_config("configs/home_credit.yaml")
    assert cfg.data.adapter == "csv"
    assert cfg.seed == 42  # inherited from base
    assert cfg.scaling.pdo == 20


def test_invalid_split_sizes_fail_fast(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("split:\n  test_size: 0.7\n  oot_size: 0.5\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(bad)


def test_invalid_iv_order_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("selection:\n  iv_min: 0.6\n  iv_suspicious: 0.5\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(bad)


def test_invalid_psi_order_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("monitoring:\n  psi_warn: 0.30\n  psi_alert: 0.20\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(bad)


def test_csv_adapter_requires_path(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  adapter: csv\n  path: null\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(bad)
