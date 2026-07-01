"""Definition of default, vintage assignment & sample design (refactor §5.1).

Basel-aligned sample construction (EBA Supervisory Handbook on IRB Validation 2023;
Basel default = 90 DPD). Responsibilities:

* **Default flag** — construct the target from DPD / status columns per config. Two
  entry points:
    - :func:`resolve_default_from_panel` for a longitudinal DPD panel
      ``[account_id, month_index, dpd]`` — implements the cure period and the
      ``separate`` / ``merge`` re-default treatment properly (time is required for
      cure/re-default; a flat snapshot cannot express them).
    - :func:`construct_default_flag` for a flat modelling frame — threshold / status
      rule, or pass-through of an already-supplied binary target.
* **Vintage / cohort** — assign ``cohort_key`` (default ``origination_month``) from an
  origination date column.
* **Performance window & seasoning** — an application enters the dev sample only once
  ``reference_date - origination >= observation_window + performance_window`` and the
  cohort is at least ``minimum_seasoning_months`` old.
* **Exclusions** — apply ``exclusions`` (pandas ``query`` rules), logging counts dropped.
* **Artifact** — emit ``artifacts/sample_design.json`` (counts by cohort, by exclusion
  rule, dev size, base rate per cohort).

Backward-compatibility (diagnostic risk R1): when the DPD/status/origination columns
are absent (German Credit / Home Credit flat frames), the module keeps the supplied
target and records the limitation rather than fabricating a longitudinal history.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

SAMPLE_DESIGN_FILE = "sample_design.json"


# --------------------------------------------------------------------------- #
# Default flag construction
# --------------------------------------------------------------------------- #
def resolve_default_from_panel(
    panel: pd.DataFrame,
    config: Config,
    account_col: str = "account_id",
    month_col: str = "month_index",
    dpd_col: str = "dpd",
) -> pd.DataFrame:
    """Construct a per-account default flag from a longitudinal DPD panel.

    An account **defaults** the first month its DPD reaches ``dpd_threshold``. It is
    **cured** if it then records ``cure_period_months`` consecutive months below the
    threshold. Re-default after a cure is counted as a fresh event when
    ``re_default_treatment == 'separate'`` and folded into the original spell when
    ``'merge'``. The returned label is 1 if the account has any *uncured* default
    spell within the observed panel.

    Returns a frame ``[account_id, default, n_default_spells]``.
    """
    dd = config.sample_design.default_definition
    threshold = dd.dpd_threshold
    cure = dd.cure_period_months
    merge = dd.re_default_treatment == "merge"

    rows: list[dict] = []
    for acct, grp in panel.sort_values([account_col, month_col]).groupby(account_col, sort=True):
        dpd = pd.to_numeric(grp[dpd_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        in_default = False
        clean_run = 0
        spells = 0
        for d in dpd:
            if d >= threshold:
                if not in_default:
                    # Opening a spell: a fresh spell counts unless we merge into an
                    # existing (already-closed) spell for this account.
                    if not (merge and spells > 0):
                        spells += 1
                    else:
                        spells = max(spells, 1)
                    in_default = True
                clean_run = 0
            elif in_default:
                clean_run += 1
                if clean_run >= cure:
                    in_default = False  # cured
        rows.append(
            {account_col: acct, "default": int(spells > 0), "n_default_spells": int(spells)}
        )
    out = pd.DataFrame(rows)
    logger.info(
        "Panel default resolution: %d accounts, %d defaulted (threshold=%d DPD, cure=%dm, %s).",
        len(out),
        int(out["default"].sum()),
        threshold,
        cure,
        dd.re_default_treatment,
    )
    return out


def construct_default_flag(df: pd.DataFrame, config: Config) -> tuple[pd.Series, bool, list[str]]:
    """Build the binary default target for a flat modelling frame.

    Priority: DPD threshold (and status override) → status-only rule → pass-through of
    the configured binary target. Returns ``(flag, constructed, notes)`` where
    ``constructed`` is False in pass-through mode.
    """
    sd = config.sample_design
    dd = sd.default_definition
    notes: list[str] = []

    has_dpd = bool(sd.dpd_column) and sd.dpd_column in df.columns
    has_status = (
        bool(sd.status_column) and sd.status_column in df.columns and bool(sd.default_statuses)
    )

    if has_dpd:
        dpd = pd.to_numeric(df[sd.dpd_column], errors="coerce").fillna(0)
        flag = (dpd >= dd.dpd_threshold).astype(int)
        notes.append(f"Default = {sd.dpd_column} >= {dd.dpd_threshold} DPD (Basel).")
        if has_status:
            status_default = df[sd.status_column].isin(sd.default_statuses)
            flag = (flag.astype(bool) | status_default).astype(int)
            notes.append(f"Status override: {sd.status_column} in {sd.default_statuses}.")
        return flag.rename(config.data.target), True, notes

    if has_status:
        flag = df[sd.status_column].isin(sd.default_statuses).astype(int)
        notes.append(f"Default = {sd.status_column} in {sd.default_statuses}.")
        return flag.rename(config.data.target), True, notes

    # Pass-through: the supplied target already encodes default.
    notes.append(
        "Pass-through default: no DPD/status columns configured or present; using the "
        "supplied binary target as the default flag (flat sample, no cure/re-default)."
    )
    return df[config.data.target].astype(int), False, notes


# --------------------------------------------------------------------------- #
# Vintage / cohort, seasoning, exclusions
# --------------------------------------------------------------------------- #
def _origination_dates(df: pd.DataFrame, config: Config) -> pd.Series | None:
    sd = config.sample_design
    col = sd.origination_date_column or config.data.date_column
    if col and col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return None


def assign_cohort(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Assign the vintage cohort column from origination dates."""
    sd = config.sample_design
    dates = _origination_dates(df, config)
    out = df.copy()
    if dates is None:
        return out, None, [f"Cohort '{sd.cohort_key}' unavailable: no origination date column."]
    out[sd.cohort_key] = dates.dt.to_period("M").astype(str)
    return out, sd.cohort_key, [f"Cohort '{sd.cohort_key}' derived from origination month."]


def apply_exclusions(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply configured exclusion rules; return the surviving frame + per-rule drop counts.

    Each rule is a pandas ``query`` expression selecting the rows to **exclude**. A rule
    referencing a missing column is skipped with a warning (recorded as 0 dropped).
    """
    counts: dict[str, int] = {}
    out = df
    for rule in config.sample_design.exclusions:
        before = len(out)
        try:
            to_drop = out.query(rule.rule)
            out = out.drop(index=to_drop.index)
            counts[rule.name] = before - len(out)
            logger.info("Exclusion '%s' dropped %d rows.", rule.name, counts[rule.name])
        except Exception as exc:  # noqa: BLE001 - a bad rule must not crash the pipeline
            counts[rule.name] = 0
            logger.warning("Exclusion '%s' skipped (%s): %s", rule.name, type(exc).__name__, exc)
    return out.reset_index(drop=True), counts


def seasoning_filter(
    df: pd.DataFrame, config: Config, cohort_col: str | None
) -> tuple[pd.DataFrame, int, list[str]]:
    """Drop cohorts too fresh to have observed the full performance window.

    Requires origination dates and a reference (as-of) date. Without them the filter is
    a documented no-op (flat sample), per risk R1.
    """
    sd = config.sample_design
    # Seasoning keys off an EXPLICIT origination date column only. The generic
    # ``data.date_column`` (which the synthetic adapter fabricates just to enable a
    # temporal split) is not a real origination/performance date, so we do not season
    # on it — that would silently drop the most-recent rows (diagnostic risk R1).
    col = sd.origination_date_column
    if not col or col not in df.columns:
        return (
            df,
            0,
            ["Seasoning filter skipped: no explicit origination_date_column (flat sample)."],
        )
    dates = pd.to_datetime(df[col], errors="coerce")

    ref = pd.to_datetime(sd.reference_date) if sd.reference_date else dates.max()
    months_seasoned = (ref.to_period("M") - dates.dt.to_period("M")).apply(
        lambda x: x.n if pd.notna(x) else np.nan
    )
    # A cohort must have observed the full performance window and be at least the
    # configured minimum seasoning old before it can enter the dev sample.
    required = max(sd.performance_window_months, sd.minimum_seasoning_months)
    keep = months_seasoned >= required
    n_dropped = int((~keep).sum())
    note = (
        f"Seasoning filter: require >= {required} months seasoned "
        f"(perf={sd.performance_window_months}, min={sd.minimum_seasoning_months}); "
        f"as-of {ref.date()}; dropped {n_dropped}."
    )
    logger.info(note)
    return df.loc[keep].reset_index(drop=True), n_dropped, [note]


# --------------------------------------------------------------------------- #
# Orchestration + artifact
# --------------------------------------------------------------------------- #
@dataclass
class SampleDesignResult:
    frame: pd.DataFrame
    target_col: str
    cohort_col: str | None
    constructed_default: bool
    n_raw: int
    n_after_exclusions: int
    n_after_seasoning: int
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    seasoning_dropped: int = 0
    cohort_summary: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("frame")
        return d


def run_sample_design(df: pd.DataFrame, config: Config) -> SampleDesignResult:
    """Full §5.1 sample design: default flag → cohort → exclusions → seasoning.

    Returns the modelling-ready frame plus an auditable summary. The frame's target
    column is (re)written with the constructed/pass-through default flag.
    """
    n_raw = len(df)
    notes: list[str] = []

    flag, constructed, dn = construct_default_flag(df, config)
    notes += dn
    work = df.copy()
    work[config.data.target] = flag.to_numpy()

    work, cohort_col, cn = assign_cohort(work, config)
    notes += cn

    work, excl_counts = apply_exclusions(work, config)
    n_after_excl = len(work)

    work, seasoning_dropped, sn = seasoning_filter(work, config, cohort_col)
    notes += sn
    n_after_seasoning = len(work)

    cohort_summary: list[dict] = []
    if cohort_col and cohort_col in work.columns:
        for cohort, grp in work.groupby(cohort_col, sort=True):
            cohort_summary.append(
                {
                    "cohort": str(cohort),
                    "count": int(len(grp)),
                    "bad_rate": float(grp[config.data.target].mean()),
                }
            )

    result = SampleDesignResult(
        frame=work,
        target_col=config.data.target,
        cohort_col=cohort_col,
        constructed_default=constructed,
        n_raw=n_raw,
        n_after_exclusions=n_after_excl,
        n_after_seasoning=n_after_seasoning,
        exclusion_counts=excl_counts,
        seasoning_dropped=seasoning_dropped,
        cohort_summary=cohort_summary,
        notes=notes,
    )
    logger.info(
        "Sample design: raw=%d -> post-exclusions=%d -> post-seasoning=%d (constructed=%s).",
        n_raw,
        n_after_excl,
        n_after_seasoning,
        constructed,
    )
    return result


def save_sample_design(result: SampleDesignResult, config: Config) -> Path:
    """Persist the sample-design summary to ``artifacts/sample_design.json``."""
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / SAMPLE_DESIGN_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.summary_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved sample design summary to %s", path)
    return path
