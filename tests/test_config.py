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


# --------------------------------------------------------------------------- #
# Refactor §4 expanded config blocks — defaults present + fail-fast validation
# --------------------------------------------------------------------------- #
def test_expanded_blocks_have_defaults():
    cfg = load_config("configs/home_credit.yaml")
    assert cfg.sample_design.default_definition.dpd_threshold == 90
    assert cfg.reject_inference.enabled is False
    assert cfg.discrimination.bootstrap_method == "bca"
    assert cfg.benchmark.challenger == "gradient_boosting"
    assert cfg.calibration_extended.per_grade_backtest.method == "jeffreys"
    assert cfg.fairness.air_threshold_alert == 0.80
    assert cfg.explainability.reason_codes_top_n == 4
    assert cfg.monitoring_extended.runlog_backend == "sqlite"
    assert cfg.governance.model_tier == 2


def test_reject_inference_enabled_requires_path(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "reject_inference:\n  enabled: true\n  reject_data_path: null\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        load_config(bad)


def test_fairness_threshold_order_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    # alert must be strictly below warn (80% rule below the warn band).
    bad.write_text(
        "fairness:\n  air_threshold_warn: 0.80\n  air_threshold_alert: 0.90\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        load_config(bad)


def test_partial_auc_range_order_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("discrimination:\n  partial_auc_range: [0.5, 0.2]\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(bad)


def test_traffic_light_quantile_order_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "calibration_extended:\n"
        "  per_grade_backtest:\n"
        "    traffic_light:\n"
        "      green_upper_quantile: 0.99\n"
        "      yellow_upper_quantile: 0.95\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_config(bad)


def test_german_credit_config_loads_with_protected_attrs():
    cfg = load_config("configs/german_credit.yaml")
    assert cfg.fairness.enabled is True
    assert "foreign_worker" in cfg.fairness.protected_attributes
    assert cfg.governance.model_id == "PD-APP-GC-001"
