"""Point discrimination metrics for the pipeline's internal performance summary.

The *reportable* discrimination artifact — with bootstrap confidence intervals, partial
AUC, Somers' D and .632+ optimism correction — lives in
:mod:`creditscorecard.evaluation.discrimination` (refactor §5.3). This module is the thin
point-estimate table used for quick logging / the metrics.csv table; it re-exports the
canonical ``auc``/``gini``/``ks`` from ``discrimination`` so there is a single source of
truth. Never report a bare Gini here without the CI-bearing ``discrimination.json`` beside
it (diagnostic §3 DEMOLISH rule).

All metrics take ``P(Bad)`` as the score so higher == riskier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditscorecard.evaluation.discrimination import auc, gini, ks

__all__ = ["auc", "gini", "ks", "compute_metrics", "metrics_table"]


def compute_metrics(
    y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series
) -> dict[str, float]:
    a = auc(y_true, prob_bad)
    return {"auc": a, "gini": 2.0 * a - 1.0, "ks": ks(y_true, prob_bad)}


def metrics_table(splits: dict[str, tuple]) -> pd.DataFrame:
    """Build a train/test/OOT metrics table.

    ``splits`` maps a split name to ``(y_true, prob_bad)``.
    """
    rows = []
    for name, (y, p) in splits.items():
        m = compute_metrics(y, p)
        rows.append({"split": name, **m, "n": len(y), "bad_rate": float(np.mean(np.asarray(y)))})
    return pd.DataFrame(rows)
