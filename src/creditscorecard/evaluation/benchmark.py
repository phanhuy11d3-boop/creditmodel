"""Champion vs challenger benchmark (refactor §5.4).

Fits a non-linear challenger (gradient boosting by default) on the same information as
the reportable WoE-logistic model and asks the validator's first question: *does a more
flexible model do materially better out-of-time?* If it does — OOT Gini gap above
``benchmark.gini_gap_threshold`` **and** DeLong p below ``benchmark.delong_p_threshold`` —
the model may be under-specified, and a prominent warning is raised in the MDD and CLI.

The AUC-difference significance uses **DeLong's test** (DeLong et al. 1988; fast algorithm
of Sun & Xu 2014) for two correlated ROC curves on the same OOT sample. Interpretability
parity (top-K reportable coefficients vs top-K challenger |SHAP|) is delegated to
:mod:`creditscorecard.evaluation.explainability`.

Artifact: ``artifacts/benchmark.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.evaluation.discrimination import bootstrap_ci, gini
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

BENCHMARK_FILE = "benchmark.json"


# --------------------------------------------------------------------------- #
# DeLong test for two correlated AUCs
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_roc_test(
    y_true: np.ndarray | pd.Series,
    prob_a: np.ndarray | pd.Series,
    prob_b: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Two-sided DeLong test that AUC(a) == AUC(b) on the same sample.

    Returns ``{auc_a, auc_b, z, pvalue}``. ``z > 0`` favours model *a*.
    """
    from scipy.stats import norm

    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives (y==1) first
    y = y[order]
    m = int(y.sum())
    n = len(y) - m
    preds = np.vstack(
        [np.asarray(prob_a, dtype=float)[order], np.asarray(prob_b, dtype=float)[order]]
    )
    if m == 0 or n == 0:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "z": float("nan"),
            "pvalue": float("nan"),
        }

    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    cov = np.atleast_2d(cov)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = float((aucs[0] - aucs[1]) / np.sqrt(var)) if var > 0 else 0.0
    pvalue = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": z, "pvalue": pvalue}


# --------------------------------------------------------------------------- #
# Challenger fitting
# --------------------------------------------------------------------------- #
def fit_challenger(X: pd.DataFrame, y: pd.Series, config: Config):
    """Fit the configured non-linear challenger on numeric features (WoE design)."""
    b = config.benchmark
    params = dict(b.challenger_params)
    yv = np.asarray(y).astype(int)
    if b.challenger == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params.pop("learning_rate", None)
        model = RandomForestClassifier(random_state=config.seed, **params)
    elif b.challenger == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(random_state=config.seed, eval_metric="logloss", **params)
        except ImportError:
            logger.warning("xgboost not installed; using gradient_boosting challenger instead.")
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(random_state=config.seed, **params)
    else:  # gradient_boosting (default)
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=config.seed, **params)
    model.fit(X, yv)
    logger.info("Challenger fitted: %s on %d features.", b.challenger, X.shape[1])
    return model


# --------------------------------------------------------------------------- #
# Orchestration + artifact
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    challenger: str
    reportable_gini_oot: dict = field(default_factory=dict)
    challenger_gini_oot: dict = field(default_factory=dict)
    delong: dict = field(default_factory=dict)
    interpretability_parity: dict = field(default_factory=dict)
    under_specified: bool = False
    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    y_oot: np.ndarray | pd.Series,
    reportable_p_oot: np.ndarray | pd.Series,
    challenger_p_oot: np.ndarray | pd.Series,
    config: Config,
    *,
    interpretability_parity: dict | None = None,
) -> BenchmarkResult:
    """Compare reportable vs challenger on OOT: Gini±CI, DeLong, verdict."""
    b = config.benchmark
    d = config.discrimination
    y = np.asarray(y_oot).astype(int)
    rep = np.asarray(reportable_p_oot, dtype=float)
    chal = np.asarray(challenger_p_oot, dtype=float)

    rep_gini = bootstrap_ci(
        y,
        rep,
        gini,
        n_iter=d.bootstrap_iterations,
        level=d.confidence_level,
        method=d.bootstrap_method,
        seed=config.seed,
    )
    chal_gini = bootstrap_ci(
        y,
        chal,
        gini,
        n_iter=d.bootstrap_iterations,
        level=d.confidence_level,
        method=d.bootstrap_method,
        seed=config.seed + 1,
    )
    delong = delong_roc_test(y, chal, rep) if b.delong_test else {}

    gini_gap = chal_gini.point - rep_gini.point
    p = delong.get("pvalue", 1.0)
    under = bool(gini_gap > b.gini_gap_threshold and p < b.delong_p_threshold)
    verdict = (
        f"UNDER-SPECIFIED: challenger Gini exceeds reportable by {gini_gap:.3f} "
        f"(> {b.gini_gap_threshold}) with DeLong p={p:.4f} (< {b.delong_p_threshold}). "
        "Consider added interactions/non-linearity."
        if under
        else (
            f"Reportable model adequate: challenger Gini gap {gini_gap:+.3f}, "
            f"DeLong p={p if isinstance(p, float) else float('nan'):.4f}."
        )
    )
    if under:
        logger.warning(verdict)
    else:
        logger.info(verdict)

    return BenchmarkResult(
        challenger=b.challenger,
        reportable_gini_oot=rep_gini.to_dict(),
        challenger_gini_oot=chal_gini.to_dict(),
        delong=delong,
        interpretability_parity=interpretability_parity or {},
        under_specified=under,
        verdict=verdict,
    )


def save_benchmark(result: BenchmarkResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / BENCHMARK_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved benchmark to %s", path)
    return path
