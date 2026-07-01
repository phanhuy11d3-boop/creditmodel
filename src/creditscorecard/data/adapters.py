"""Dataset adapters: csv, synthetic.

Every adapter returns a single :class:`pandas.DataFrame` where the configured
target column holds ``1 == Bad/default`` and ``0 == Good``. When a
``date_column`` is configured, a deterministic date is attached so temporal OOT
splitting can be exercised offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)


def load_dataset(config: Config) -> pd.DataFrame:
    """Dispatch to the configured adapter and return a modeling-ready frame."""
    adapter = config.data.adapter
    if adapter == "csv":
        df = _load_csv(config)
    elif adapter == "synthetic":
        df = _load_synthetic(config)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unknown adapter: {adapter}")

    _validate_target(df, config)
    df = _attach_date(df, config)
    logger.info(
        "Loaded dataset via '%s': rows=%d cols=%d bad_rate=%.4f",
        adapter,
        len(df),
        df.shape[1],
        df[config.data.target].mean(),
    )
    return df.reset_index(drop=True)


def _validate_target(df: pd.DataFrame, config: Config) -> None:
    target = config.data.target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from dataset")
    vals = set(pd.unique(df[target].dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Target '{target}' must be binary 0/1, found {sorted(vals)}")


def _attach_date(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    date_col = config.data.date_column
    if not date_col or date_col in df.columns:
        return df
    rng = np.random.default_rng(config.seed)
    # Spread applications across 24 months ending "today-ish" for a temporal split.
    n = len(df)
    day_offsets = rng.integers(0, 730, size=n)
    base = np.datetime64("2021-01-01")
    dates = base + day_offsets.astype("timedelta64[D]")
    df = df.copy()
    df[date_col] = pd.to_datetime(dates)
    return df


def _synthesize_offline_demo(config: Config) -> pd.DataFrame:
    """Deterministic synthetic credit dataset (schema-agnostic offline fallback)."""
    rng = np.random.default_rng(config.seed)
    n = 1000

    checking = rng.choice(["A11", "A12", "A13", "A14"], n, p=[0.27, 0.27, 0.06, 0.40])
    credit_history = rng.choice(
        ["A30", "A31", "A32", "A33", "A34"], n, p=[0.04, 0.05, 0.53, 0.09, 0.29]
    )
    purpose = rng.choice(
        ["A40", "A41", "A42", "A43", "A45", "A46", "A49"],
        n,
        p=[0.23, 0.10, 0.18, 0.28, 0.05, 0.05, 0.11],
    )
    savings = rng.choice(["A61", "A62", "A63", "A64", "A65"], n, p=[0.60, 0.10, 0.06, 0.05, 0.19])
    employment = rng.choice(
        ["A71", "A72", "A73", "A74", "A75"], n, p=[0.06, 0.17, 0.34, 0.17, 0.26]
    )
    personal = rng.choice(["A91", "A92", "A93", "A94"], n, p=[0.05, 0.31, 0.55, 0.09])
    other_debtors = rng.choice(["A101", "A102", "A103"], n, p=[0.91, 0.04, 0.05])
    property_mag = rng.choice(["A121", "A122", "A123", "A124"], n, p=[0.28, 0.23, 0.33, 0.16])
    other_plans = rng.choice(["A141", "A142", "A143"], n, p=[0.14, 0.05, 0.81])
    housing = rng.choice(["A151", "A152", "A153"], n, p=[0.18, 0.71, 0.11])
    job = rng.choice(["A171", "A172", "A173", "A174"], n, p=[0.02, 0.20, 0.63, 0.15])
    telephone = rng.choice(["A191", "A192"], n, p=[0.60, 0.40])
    foreign = rng.choice(["A201", "A202"], n, p=[0.96, 0.04])

    duration = rng.integers(4, 72, n)
    amount = rng.gamma(shape=2.0, scale=1600.0, size=n).clip(250, 20000).round().astype(int)
    installment_rate = rng.integers(1, 5, n)
    residence = rng.integers(1, 5, n)
    age = rng.integers(19, 76, n)
    existing_credits = rng.integers(1, 5, n)
    dependents = rng.integers(1, 3, n)

    # Latent risk: higher => more likely Bad. Signs chosen for realistic WoE monotonicity.
    checking_risk = (
        pd.Series(checking).map({"A11": 0.9, "A12": 0.4, "A13": 0.1, "A14": -0.7}).to_numpy()
    )
    savings_risk = (
        pd.Series(savings)
        .map({"A61": 0.5, "A62": 0.2, "A63": -0.1, "A64": -0.6, "A65": -0.3})
        .to_numpy()
    )
    history_risk = (
        pd.Series(credit_history)
        .map({"A30": 0.8, "A31": 0.6, "A32": 0.0, "A33": -0.3, "A34": -0.6})
        .to_numpy()
    )
    employ_risk = (
        pd.Series(employment)
        .map({"A71": 0.6, "A72": 0.3, "A73": 0.0, "A74": -0.3, "A75": -0.4})
        .to_numpy()
    )

    z = (
        -1.05
        + checking_risk
        + 0.6 * savings_risk
        + 0.7 * history_risk
        + 0.4 * employ_risk
        + 0.020 * (duration - duration.mean())
        + 0.00010 * (amount - amount.mean())
        - 0.025 * (age - age.mean())
        + 0.15 * (installment_rate - installment_rate.mean())
        + rng.normal(0, 0.35, n)
    )
    p_bad = 1.0 / (1.0 + np.exp(-z))
    default = (rng.uniform(size=n) < p_bad).astype(int)

    return pd.DataFrame(
        {
            "checking_status": checking,
            "duration_months": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": amount,
            "savings_status": savings,
            "employment_since": employment,
            "installment_rate": installment_rate,
            "personal_status_sex": personal,
            "other_debtors": other_debtors,
            "residence_since": residence,
            "property_magnitude": property_mag,
            "age_years": age,
            "other_installment_plans": other_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "num_dependents": dependents,
            "telephone": telephone,
            "foreign_worker": foreign,
            "default": default,
        }
    )


# --------------------------------------------------------------------------- #
# Generic synthetic + csv
# --------------------------------------------------------------------------- #
def _load_synthetic(config: Config) -> pd.DataFrame:
    """Small generic synthetic credit dataset (used by tests / offline demo)."""
    return _synthesize_offline_demo(config)


def _load_csv(config: Config) -> pd.DataFrame:
    assert config.data.path is not None  # guaranteed by config validation
    df = pd.read_csv(config.data.path)
    if config.data.target not in df.columns:
        raise ValueError(f"CSV missing target column '{config.data.target}'")
    return df
