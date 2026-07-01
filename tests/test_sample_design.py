"""Sample design (§5.1): cohort assignment, exclusions, seasoning, per-cohort base rate."""

from __future__ import annotations

import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.data.definition_of_default import (
    apply_exclusions,
    run_sample_design,
    save_sample_design,
)


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.data.adapter = "synthetic"
    c.data.target = "default"
    c.data.date_column = "orig_date"
    return c


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "orig_date": pd.to_datetime(
                ["2021-01-05", "2021-01-20", "2021-02-10", "2021-02-15", "2021-03-01", "2021-03-09"]
            ),
            "default": [1, 0, 1, 1, 0, 0],
            "staff": [0, 0, 0, 0, 1, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )


def test_cohort_assignment_and_base_rates(cfg):
    res = run_sample_design(_frame(), cfg)
    assert res.cohort_col == "origination_month"
    by_cohort = {c["cohort"]: c["bad_rate"] for c in res.cohort_summary}
    assert by_cohort["2021-01"] == pytest.approx(0.5)
    assert by_cohort["2021-02"] == pytest.approx(1.0)
    assert by_cohort["2021-03"] == pytest.approx(0.0)


def test_exclusions_reduce_size_deterministically(cfg):
    # Without the rule: all 6 rows survive.
    res_none = run_sample_design(_frame(), cfg)
    assert res_none.n_after_exclusions == 6

    # Adding the staff exclusion deterministically drops exactly the 2 staff rows.
    cfg.sample_design.exclusions = [_rule("staff_loans", "staff == 1")]
    res = run_sample_design(_frame(), cfg)
    assert res.exclusion_counts["staff_loans"] == 2
    assert res.n_after_exclusions == 4
    # Determinism: a second run yields the identical count.
    assert run_sample_design(_frame(), cfg).n_after_exclusions == 4


def test_bad_exclusion_rule_is_skipped_not_fatal(cfg):
    out, counts = apply_exclusions(_frame(), _cfg_with_rule(cfg, "bad", "nonexistent_col == 1"))
    assert counts["bad"] == 0
    assert len(out) == 6


def test_seasoning_drops_fresh_cohorts(cfg):
    cfg.sample_design.origination_date_column = "orig_date"
    cfg.sample_design.reference_date = "2021-02-15"
    cfg.sample_design.performance_window_months = 1
    cfg.sample_design.minimum_seasoning_months = 0
    res = run_sample_design(_frame(), cfg)
    # Only the 2021-01 cohort is >= 1 month seasoned as of 2021-02.
    assert res.seasoning_dropped == 4
    assert res.n_after_seasoning == 2


def test_save_sample_design_writes_json(cfg, tmp_path):
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    res = run_sample_design(_frame(), cfg)
    path = save_sample_design(res, cfg)
    assert path.exists()
    import json

    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["n_raw"] == 6
    assert "frame" not in d  # the dataframe itself must not be serialised


# --- helpers ---------------------------------------------------------------- #
def _rule(name: str, expr: str):
    from creditscorecard.config import ExclusionRule

    return ExclusionRule(name=name, rule=expr)


def _cfg_with_rule(cfg, name, expr):
    cfg.sample_design.exclusions = [_rule(name, expr)]
    return cfg
