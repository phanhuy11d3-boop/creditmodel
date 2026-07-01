"""Monotonic optimal binning (OptBinning) with frozen, serializable bin specs.

Design choice: OptBinning is used **only at development time** to discover
optimal, monotonic bin edges (numeric cut points / categorical groups) plus
explicit Missing and Special bins. We then extract a small serializable spec per
feature and reimplement bin assignment in pure pandas/numpy. Consequences:

* Scoring and monitoring have **no OptBinning dependency** (constraint: no
  training deps at inference).
* Bin edges are **frozen** at development and reused verbatim everywhere
  (constraint 11: no re-binning of new data).

Bin codes: normal bins are ``0..n_bins-1`` (monotonic in event rate). Two
reserved codes carry the explicit special bins:

* :data:`MISSING_CODE` (-1): value is NaN / missing.
* :data:`OTHER_CODE` (-2): categorical level unseen in training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

MISSING_CODE = -1
OTHER_CODE = -2


@dataclass
class FeatureBinning:
    """Frozen, serializable binning spec for a single characteristic."""

    name: str
    dtype: str  # "numerical" | "categorical"
    splits: list[float] = field(default_factory=list)  # numeric cut points
    groups: list[list[str]] = field(default_factory=list)  # categorical bin groups
    n_bins: int = 0
    labels: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureBinning:
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            splits=list(d.get("splits", [])),
            groups=[list(g) for g in d.get("groups", [])],
            n_bins=int(d["n_bins"]),
            labels={int(k): v for k, v in d.get("labels", {}).items()},
        )


def assign_codes(spec: FeatureBinning, series: pd.Series) -> np.ndarray:
    """Assign integer bin codes using only the frozen spec (no OptBinning)."""
    if spec.dtype == "numerical":
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        codes = np.full(values.shape, MISSING_CODE, dtype=int)
        mask = ~np.isnan(values)
        if spec.splits:
            codes[mask] = np.digitize(values[mask], np.asarray(spec.splits), right=False)
        else:
            codes[mask] = 0
        return codes

    # categorical
    lookup: dict[str, int] = {}
    for idx, group in enumerate(spec.groups):
        for cat in group:
            lookup[str(cat)] = idx
    codes = np.empty(len(series), dtype=int)
    for i, raw in enumerate(series.to_numpy()):
        if pd.isna(raw):
            codes[i] = MISSING_CODE
        else:
            codes[i] = lookup.get(str(raw), OTHER_CODE)
    return codes


def _numeric_labels(splits: list[float]) -> dict[int, str]:
    labels: dict[int, str] = {}
    n = len(splits)
    for i in range(n + 1):
        lo = "-inf" if i == 0 else f"{splits[i - 1]:.4g}"
        hi = "inf" if i == n else f"{splits[i]:.4g}"
        labels[i] = f"[{lo}, {hi})"
    labels[MISSING_CODE] = "Missing"
    return labels


def _categorical_labels(groups: list[list[str]]) -> dict[int, str]:
    labels = {i: "[" + ", ".join(map(str, g)) + "]" for i, g in enumerate(groups)}
    labels[MISSING_CODE] = "Missing"
    labels[OTHER_CODE] = "Other/Unseen"
    return labels


class BinningModel:
    """Fits monotonic optimal bins on the **train** sample only."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.specs: dict[str, FeatureBinning] = {}
        self.opt_tables: dict[str, pd.DataFrame] = {}
        self.features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BinningModel:
        b = self.config.binning
        # Constraint 3 requires a *strictly monotonic* WoE trend. OptBinning's
        # plain "auto" permits peak/valley shapes, so map it to "auto_asc_desc"
        # which restricts the solution to ascending or descending only.
        trend = "auto_asc_desc" if b.monotonic_trend == "auto" else b.monotonic_trend
        y_arr = np.asarray(y).astype(int)
        for col in X.columns:
            dtype = "numerical" if pd.api.types.is_numeric_dtype(X[col]) else "categorical"
            x = X[col]
            x_arr = x.to_numpy() if dtype == "categorical" else pd.to_numeric(x).to_numpy()
            optb = OptimalBinning(
                name=col,
                dtype=dtype,
                monotonic_trend=trend,
                min_bin_size=b.min_bin_pct,
                max_n_bins=b.max_n_bins,
            )
            optb.fit(x_arr, y_arr)
            spec = self._extract_spec(col, dtype, optb)
            self.specs[col] = spec
            self.opt_tables[col] = optb.binning_table.build()
        self.features = list(X.columns)
        logger.info("Fitted binning on %d characteristics (train only).", len(self.features))
        return self

    @staticmethod
    def _extract_spec(name: str, dtype: str, optb: OptimalBinning) -> FeatureBinning:
        if dtype == "numerical":
            splits = [float(s) for s in np.asarray(optb.splits).ravel().tolist()]
            return FeatureBinning(
                name=name,
                dtype=dtype,
                splits=splits,
                n_bins=len(splits) + 1,
                labels=_numeric_labels(splits),
            )
        groups = [[str(c) for c in np.asarray(g).ravel().tolist()] for g in optb.splits]
        return FeatureBinning(
            name=name,
            dtype=dtype,
            groups=groups,
            n_bins=len(groups),
            labels=_categorical_labels(groups),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return an integer bin-code frame using frozen specs (transform only)."""
        data = {col: assign_codes(self.specs[col], X[col]) for col in self.features}
        return pd.DataFrame(data, index=X.index)

    def binning_specs_dict(self) -> dict[str, dict]:
        return {name: spec.to_dict() for name, spec in self.specs.items()}
