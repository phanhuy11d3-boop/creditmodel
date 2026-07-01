"""Pandera data contract.

A light-but-strict contract validated at ingestion: the target must be binary
and non-null, and the frame must be non-empty. Contract violations fail the
pipeline early rather than surfacing as silent modeling bugs later. Add
per-column range checks here if your dataset needs stricter plausibility
bounds (see README "Swapping in a real dataset").
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)


def build_schema(df: pd.DataFrame, config: Config) -> DataFrameSchema:
    """Build a data-contract schema for the loaded frame."""
    target = config.data.target
    columns: dict[str, Column] = {
        target: Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Target: 1 = Bad/default event, 0 = Good.",
        )
    }

    return DataFrameSchema(
        columns=columns,
        strict=False,  # allow extra (categorical / date) columns
        coerce=True,
        checks=Check(lambda d: len(d) > 0, error="dataset is empty"),
    )


def validate_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Validate ``df`` against the contract, raising on violation."""
    schema = build_schema(df, config)
    try:
        validated = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error("Data contract validation failed:\n%s", exc.failure_cases)
        raise
    logger.info("Data contract validated: %d rows, %d columns.", *validated.shape)
    return validated
