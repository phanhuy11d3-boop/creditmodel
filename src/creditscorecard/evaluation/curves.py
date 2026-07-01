"""Evaluation figures: ROC, CAP, calibration (Hosmer-Lemeshow), score distribution.

Uses the non-interactive Agg backend so figures render headless (CI/Docker).
Each function saves a PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from creditscorecard.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SplitProbs = dict[str, tuple]  # name -> (y_true, prob_bad)


def hosmer_lemeshow(
    y_true: np.ndarray, prob_bad: np.ndarray, n_groups: int = 10
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit statistic and p-value."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    groups = np.array_split(np.arange(len(p)), n_groups)
    stat = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        obs = y[g].sum()
        exp = p[g].sum()
        n = len(g)
        denom = exp * (1 - exp / n)
        if denom > 0:
            stat += (obs - exp) ** 2 / denom
    dof = max(n_groups - 2, 1)
    pvalue = float(1 - stats.chi2.cdf(stat, dof))
    return float(stat), pvalue


def plot_roc(splits: SplitProbs, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in splits.items():
        fpr, tpr, _ = roc_curve(np.asarray(y).astype(int), np.asarray(p, dtype=float))
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))  # trapezoid
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    return _save(fig, out_dir / "roc_curve.png")


def plot_cap(splits: SplitProbs, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in splits.items():
        yv = np.asarray(y).astype(int)
        pv = np.asarray(p, dtype=float)
        order = np.argsort(-pv)
        cum = np.cumsum(yv[order]) / yv.sum()
        x = np.arange(1, len(yv) + 1) / len(yv)
        ax.plot(x, cum, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set(xlabel="Fraction of population", ylabel="Fraction of Bads captured", title="CAP Curve")
    ax.legend(loc="lower right")
    return _save(fig, out_dir / "cap_curve.png")


def plot_calibration(
    y_true: np.ndarray, prob_bad: np.ndarray, out_dir: Path, n_bins: int = 10
) -> Path:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    groups = np.array_split(np.arange(len(p)), n_bins)
    mean_pred = [p[g].mean() for g in groups if len(g)]
    obs_rate = [y[g].mean() for g in groups if len(g)]
    stat, pval = hosmer_lemeshow(y, p, n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.plot(mean_pred, obs_rate, "o-", label="Model")
    ax.set(
        xlabel="Mean predicted PD",
        ylabel="Observed default rate",
        title=f"Calibration (HL={stat:.2f}, p={pval:.3f})",
    )
    ax.legend(loc="upper left")
    return _save(fig, out_dir / "calibration.png")


def plot_score_distribution(scores: np.ndarray, y_true: np.ndarray, out_dir: Path) -> Path:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(scores.min(), scores.max(), 30)
    ax.hist(scores[y == 0], bins=bins, alpha=0.6, label="Good", density=True)
    ax.hist(scores[y == 1], bins=bins, alpha=0.6, label="Bad", density=True)
    ax.set(xlabel="Score", ylabel="Density", title="Score distribution by class")
    ax.legend()
    return _save(fig, out_dir / "score_distribution.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)
    return path
