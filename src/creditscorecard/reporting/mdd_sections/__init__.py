"""Ordered MDD chapters (§6). One module per chapter; each exposes ``render(ctx) -> str``.

The order of ``CHAPTERS`` is the order of the assembled document and the per-chapter files.
"""

from __future__ import annotations

from creditscorecard.reporting.mdd_sections import (
    ch01_executive,
    ch02_sample_design,
    ch03_reject_inference,
    ch04_data_quality,
    ch05_feature_engineering,
    ch06_feature_selection,
    ch07_model,
    ch08_calibration,
    ch09_scaling,
    ch10_discrimination,
    ch11_benchmark,
    ch12_fairness,
    ch13_explainability,
    ch14_monitoring,
    ch15_limitations,
    ch16_references,
)

CHAPTERS = [
    ch01_executive,
    ch02_sample_design,
    ch03_reject_inference,
    ch04_data_quality,
    ch05_feature_engineering,
    ch06_feature_selection,
    ch07_model,
    ch08_calibration,
    ch09_scaling,
    ch10_discrimination,
    ch11_benchmark,
    ch12_fairness,
    ch13_explainability,
    ch14_monitoring,
    ch15_limitations,
    ch16_references,
]
