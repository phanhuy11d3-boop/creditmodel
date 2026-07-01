"""Discrimination metrics for an imbalanced PD target: AUC, Gini, KS.

Accuracy is intentionally *not* provided as a headline metric (spec anti-pattern).
All metrics take ``P(Bad)`` as the score so higher == riskier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series) -> float:
    return float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(prob_bad, dtype=float)))


def gini(y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series) -> float:
    return 2.0 * auc(y_true, prob_bad) - 1.0


def ks(y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative Bad and Good distributions."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    order = np.argsort(p)
    y = y[order]
    n_bad = int(y.sum())
    n_good = int((1 - y).sum())
    if n_bad == 0 or n_good == 0:
        return 0.0
    cum_bad = np.cumsum(y) / n_bad
    cum_good = np.cumsum(1 - y) / n_good
    return float(np.max(np.abs(cum_bad - cum_good)))


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
