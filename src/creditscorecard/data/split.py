"""Train / in-time test / out-of-time (OOT) splitting.

Splitting is the second stage of sample construction: the frame first passes through
the §5.1 sample-design module (default flag, cohort, exclusions, seasoning — see
:mod:`creditscorecard.data.definition_of_default`), then this module carves it into
train / in-time test / out-of-time slices.

When a date column exists the OOT slice is the **most recent** fraction, cut *before*
any fitting; the older development portion is split (stratified, random) into train and
in-time test. Without a date column, all three splits are stratified-random and we log
that temporal validation is unavailable.

Backward-compatibility (diagnostic risk R1): with the default configs (no DPD/status/
origination columns, no exclusions) sample design is a no-op on the modelling frame, so
splits are byte-identical to the legacy behaviour. The sample-design *summary* is still
computed and attached to :class:`SplitData` for the artifact/MDD.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from creditscorecard.config import Config
from creditscorecard.data.definition_of_default import SampleDesignResult, run_sample_design
from creditscorecard.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    temporal: bool
    sample_design: SampleDesignResult | None = None

    def describe(self) -> dict[str, int]:
        return {"train": len(self.train), "test": len(self.test), "oot": len(self.oot)}


def split_data(df: pd.DataFrame, config: Config) -> SplitData:
    # Stage 1: sample design (default flag, cohort, exclusions, seasoning).
    design = run_sample_design(df, config)
    modelling = design.frame
    # The cohort column is auditing metadata, not a feature — drop it before splitting
    # so the feature matrix is unchanged versus the legacy pipeline (risk R1).
    if design.cohort_col and design.cohort_col in modelling.columns:
        modelling = modelling.drop(columns=[design.cohort_col])

    split = _carve(modelling, config)
    return SplitData(
        train=split.train,
        test=split.test,
        oot=split.oot,
        temporal=split.temporal,
        sample_design=design,
    )


def _carve(df: pd.DataFrame, config: Config) -> SplitData:
    """Legacy train/test/OOT carving on an already-designed modelling frame."""
    target = config.data.target
    date_col = config.data.date_column
    seed = config.seed
    stratify_col = df[target] if config.split.stratify else None

    if date_col and date_col in df.columns:
        return _temporal_split(df, config)

    if date_col:
        logger.warning(
            "date_column '%s' not found; falling back to stratified random OOT "
            "(temporal validation unavailable).",
            date_col,
        )
    else:
        logger.warning(
            "No date_column configured; using stratified random OOT "
            "(temporal validation unavailable)."
        )

    dev, oot = train_test_split(
        df,
        test_size=config.split.oot_size,
        random_state=seed,
        stratify=stratify_col,
    )
    train, test = _split_dev(dev, config)
    return SplitData(_reset(train), _reset(test), _reset(oot), temporal=False)


def _temporal_split(df: pd.DataFrame, config: Config) -> SplitData:
    date_col = config.data.date_column
    assert date_col is not None
    ordered = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_oot = int(round(n * config.split.oot_size))
    if n_oot <= 0:
        raise ValueError("oot_size too small: produced an empty OOT slice")
    dev = ordered.iloc[: n - n_oot]
    oot = ordered.iloc[n - n_oot :]
    train, test = _split_dev(dev, config)
    logger.info(
        "Temporal split: dev [%s .. %s], OOT [%s .. %s]",
        dev[date_col].min(),
        dev[date_col].max(),
        oot[date_col].min(),
        oot[date_col].max(),
    )
    return SplitData(_reset(train), _reset(test), _reset(oot), temporal=True)


def _split_dev(dev: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the development portion into train and in-time test."""
    target = config.data.target
    # Keep the test share ~test_size of the *original* dataset.
    rel_test = config.split.test_size / (1.0 - config.split.oot_size)
    rel_test = min(max(rel_test, 0.01), 0.9)
    stratify_col = dev[target] if config.split.stratify else None
    train, test = train_test_split(
        dev,
        test_size=rel_test,
        random_state=config.seed,
        stratify=stratify_col,
    )
    return train, test


def _reset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(drop=True)
