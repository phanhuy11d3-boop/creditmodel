"""HHI, MAPE, anchor gap, and curve-shape (±nSE) validation checks."""

from __future__ import annotations

import math

import pytest

from creditscorecard.evaluation.calibration_checks import (
    anchor_gap,
    curve_shape_check,
    mape_by_grade,
)
from creditscorecard.evaluation.stability import herfindahl_hirschman_index


def test_hhi_uniform_distribution_equals_one_over_k():
    labels = ["A", "B", "C", "D"] * 25  # 4 equally-sized grades
    hhi = herfindahl_hirschman_index(labels)
    assert math.isclose(hhi, 0.25, rel_tol=1e-9)


def test_hhi_fully_concentrated_equals_one():
    labels = ["A"] * 100
    assert math.isclose(herfindahl_hirschman_index(labels), 1.0, rel_tol=1e-9)


def test_hhi_more_concentrated_scores_higher():
    balanced = ["A", "B", "C", "D"] * 25
    skewed = ["A"] * 85 + ["B", "C", "D"] * 5
    assert herfindahl_hirschman_index(skewed) > herfindahl_hirschman_index(balanced)


def test_mape_zero_for_perfect_calibration():
    table = [
        {"grade": "A", "count": 100, "avg_pd": 0.05, "observed_bad_rate": 0.05},
        {"grade": "B", "count": 100, "avg_pd": 0.20, "observed_bad_rate": 0.20},
    ]
    assert mape_by_grade(table) == pytest.approx(0.0)


def test_mape_exact_value():
    table = [
        {"grade": "A", "count": 100, "avg_pd": 0.10, "observed_bad_rate": 0.20},  # 50% error
        {"grade": "B", "count": 100, "avg_pd": 0.30, "observed_bad_rate": 0.30},  # 0% error
    ]
    assert mape_by_grade(table) == pytest.approx(0.25)


def test_mape_skips_empty_grades():
    table = [
        {"grade": "A", "count": 0, "avg_pd": 0.9, "observed_bad_rate": 0.0},
        {"grade": "B", "count": 50, "avg_pd": 0.10, "observed_bad_rate": 0.10},
    ]
    assert mape_by_grade(table) == pytest.approx(0.0)


def test_anchor_gap_zero_when_calibrated():
    assert anchor_gap(mean_pd=0.29, anchor_rate=0.29) == pytest.approx(0.0)


def test_anchor_gap_relative_direction():
    gap = anchor_gap(mean_pd=0.33, anchor_rate=0.30)
    assert gap == pytest.approx(0.10)  # +10% over anchor


def test_curve_shape_monotonic_and_within_band():
    # Large n -> tight SE -> observed must be close to predicted to pass.
    table = [
        {"grade": "G", "count": 500, "avg_pd": 0.70, "observed_bad_rate": 0.705},
        {"grade": "F", "count": 500, "avg_pd": 0.40, "observed_bad_rate": 0.41},
        {"grade": "E", "count": 500, "avg_pd": 0.10, "observed_bad_rate": 0.09},
    ]
    result = curve_shape_check(table, n_se=2.0)
    assert result.monotonic is True
    assert result.all_within_band is True


def test_curve_shape_detects_non_monotonic_pd():
    table = [
        # worst grade with a low PD, better grade with a high PD -> not monotonic
        {"grade": "G", "count": 500, "avg_pd": 0.20, "observed_bad_rate": 0.20},
        {"grade": "F", "count": 500, "avg_pd": 0.70, "observed_bad_rate": 0.70},
    ]
    result = curve_shape_check(table, n_se=2.0)
    assert result.monotonic is False


def test_curve_shape_detects_out_of_band_grade():
    table = [
        {"grade": "A", "count": 1000, "avg_pd": 0.05, "observed_bad_rate": 0.30},  # way off
    ]
    result = curve_shape_check(table, n_se=2.0)
    assert result.all_within_band is False
    assert result.bands[0].within_band is False


def test_pipeline_validation_summary_present(pipeline_payload):
    summary = pipeline_payload["validation_summary"]
    for section in ("discriminatory_power", "stability_concentration", "calibration_accuracy"):
        assert section in summary
    assert summary["stability_concentration"]["hhi"]["metric"] == "hhi_train_grades"
    assert summary["calibration_accuracy"]["curve_shape"]["monotonic"] is True


def test_monitor_report_includes_hhi(config, pipeline_payload, dataset):
    from creditscorecard.monitoring.monitor import run_monitoring

    report = run_monitoring(config, dataset.copy())
    assert 0.0 <= report.hhi <= 1.0
    assert report.hhi_status in {"OK", "ALERT"}
