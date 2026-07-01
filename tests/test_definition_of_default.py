"""Definition of default (§5.1): DPD threshold, cure period, re-default treatment.

Ground-truth panels are hand-constructed so the expected default flag / spell count is
unambiguous, per the §7 test requirement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from creditscorecard.config import load_config
from creditscorecard.data.definition_of_default import (
    construct_default_flag,
    resolve_default_from_panel,
)


@pytest.fixture
def cfg():
    c = load_config("configs/german_credit.yaml")
    c.data.adapter = "synthetic"
    c.data.target = "default"
    return c


def _panel() -> pd.DataFrame:
    """Three accounts over 7 months (dpd), threshold 90, cure 3.

    A1: breaches, cures after 3 clean months, breaches again -> 2 spells (separate).
    A2: never breaches -> 0 spells.
    A3: breaches, only 2 clean months (< cure) -> stays defaulted, 1 spell.
    """
    rows = []
    seqs = {
        "A1": [0, 100, 100, 0, 0, 0, 100],
        "A2": [0, 0, 0, 0, 0, 0, 0],
        "A3": [50, 95, 40, 30, 0, 0, 0],
    }
    # A3 has clean months after breach only up to <3 consecutive before more zeros;
    # months 2,3 are 40,30 (2 clean) then 0,0,0 -> actually cures. Rebuild A3 to stay in default:
    seqs["A3"] = [50, 95, 40, 30, 95, 95, 95]  # breach, 2 clean, breach again before cure
    for acct, seq in seqs.items():
        for m, d in enumerate(seq):
            rows.append({"account_id": acct, "month_index": m, "dpd": d})
    return pd.DataFrame(rows)


def test_panel_default_flag_and_threshold(cfg):
    out = resolve_default_from_panel(_panel(), cfg).set_index("account_id")
    assert out.loc["A1", "default"] == 1
    assert out.loc["A2", "default"] == 0
    assert out.loc["A3", "default"] == 1


def test_cure_and_separate_redefault(cfg):
    cfg.sample_design.default_definition.re_default_treatment = "separate"
    out = resolve_default_from_panel(_panel(), cfg).set_index("account_id")
    # A1 cured (3 clean months) then re-defaulted -> two separate spells.
    assert out.loc["A1", "n_default_spells"] == 2
    # A3 never accrues 3 consecutive clean months -> single continuous spell.
    assert out.loc["A3", "n_default_spells"] == 1


def test_merge_redefault_collapses_spells(cfg):
    cfg.sample_design.default_definition.re_default_treatment = "merge"
    out = resolve_default_from_panel(_panel(), cfg).set_index("account_id")
    assert out.loc["A1", "n_default_spells"] == 1  # merged into one spell
    assert out.loc["A1", "default"] == 1


def test_threshold_is_respected(cfg):
    # Raise the threshold above every observed DPD -> nobody defaults.
    cfg.sample_design.default_definition.dpd_threshold = 200
    out = resolve_default_from_panel(_panel(), cfg)
    assert int(out["default"].sum()) == 0


def test_flat_dpd_construction(cfg):
    cfg.sample_design.dpd_column = "max_dpd"
    df = pd.DataFrame({"max_dpd": [10, 90, 120, 0], "default": [0, 0, 0, 0]})
    flag, constructed, notes = construct_default_flag(df, cfg)
    assert constructed is True
    assert flag.tolist() == [0, 1, 1, 0]
    assert any("Basel" in n for n in notes)


def test_flat_status_override(cfg):
    cfg.sample_design.dpd_column = "max_dpd"
    cfg.sample_design.status_column = "loan_status"
    cfg.sample_design.default_statuses = ["charged_off"]
    df = pd.DataFrame(
        {
            "max_dpd": [10, 10, 120],
            "loan_status": ["current", "charged_off", "current"],
            "default": [0, 0, 0],
        }
    )
    flag, constructed, _ = construct_default_flag(df, cfg)
    assert constructed is True
    assert flag.tolist() == [0, 1, 1]  # row 1 via status, row 2 via DPD


def test_passthrough_when_no_columns(cfg):
    df = pd.DataFrame({"default": [0, 1, 1, 0], "x": [1, 2, 3, 4]})
    flag, constructed, notes = construct_default_flag(df, cfg)
    assert constructed is False
    assert flag.tolist() == [0, 1, 1, 0]
    assert any("Pass-through" in n for n in notes)
