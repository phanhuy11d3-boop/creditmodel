"""Reject inference (refactor §5.2).

Application scorecards are trained on approved applicants only — the classic
Known-Good-Bad (KGB) sample. Reject inference attempts to correct the through-the-door
population bias using the declined applications (features only, no performance).

Three methods are implemented (all operating on the numeric WoE design):

* **Parceling** — fit the base logit on accepts; score rejects; bin them into ``parcels``
  score bands; within each band assign an inferred bad rate of
  ``bad_rate_multiplier × observed accept bad rate`` (rejects are riskier), label the
  worst-scoring rejects in the band accordingly, append, refit.
* **Reweighting** — fit an acceptance model ``P(accept | X)`` on accepts ∪ rejects; refit
  the scorecard on accepts with inverse-propensity weights, upweighting accepts that
  resemble rejects.
* **Fuzzy augmentation** — score rejects; enter each reject as two weighted rows
  (weight ``p_bad`` as Bad, ``1 − p_bad`` as Good); refit with sample weights.

Guidance (Banasik & Crook 2007; Bücker, van Kampen & Krämer 2013; and the French-bank
study of Ehrhardt et al.): the *simpler* methods (parceling, reweighting) frequently match
or beat elaborate ones (Heckman two-step). No single method is presented as "correct" — the
MDD reports the **sensitivity** across methods, per that literature.

Artifact: ``artifacts/reject_inference_sensitivity.json`` (coefficient shift + Gini shift
per method vs the KGB baseline). Deterministic given ``config.seed``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

REJECT_INFERENCE_FILE = "reject_inference_sensitivity.json"


def _fit_logit(
    X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[float, dict[str, float]]:
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X, y, sample_weight=sample_weight)
    coef = {c: float(v) for c, v in zip(X.columns, model.coef_[0], strict=True)}
    return float(model.intercept_[0]), coef


def _gini_of(intercept: float, coef: dict[str, float], X: pd.DataFrame, y: np.ndarray) -> float:
    eta = intercept + X[list(coef)].to_numpy(dtype=float) @ np.array([coef[c] for c in coef])
    p = 1.0 / (1.0 + np.exp(-eta))
    if y.sum() in (0, len(y)):
        return float("nan")
    return float(2 * roc_auc_score(y, p) - 1)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def parceling(
    accepts_X: pd.DataFrame,
    accepts_y: np.ndarray,
    rejects_X: pd.DataFrame,
    config: Config,
) -> tuple[float, dict[str, float]]:
    """Parcel rejects into score bands and inject inferred bads, then refit."""
    ri = config.reject_inference
    base_int, base_coef = _fit_logit(accepts_X, accepts_y)
    cols = list(base_coef)

    def score(X: pd.DataFrame) -> np.ndarray:
        eta = base_int + X[cols].to_numpy(dtype=float) @ np.array([base_coef[c] for c in cols])
        return 1.0 / (1.0 + np.exp(-eta))

    acc_score = score(accepts_X)
    rej_score = score(rejects_X)
    edges = np.quantile(acc_score, np.linspace(0, 1, ri.parcels + 1)[1:-1])
    acc_band = np.digitize(acc_score, edges)
    rej_band = np.digitize(rej_score, edges)

    inferred = np.zeros(len(rejects_X), dtype=int)
    for band in range(ri.parcels):
        acc_mask = acc_band == band
        rej_mask = rej_band == band
        n_rej = int(rej_mask.sum())
        if n_rej == 0:
            continue
        acc_bad_rate = (
            float(accepts_y[acc_mask].mean()) if acc_mask.any() else float(accepts_y.mean())
        )
        inferred_rate = min(acc_bad_rate * ri.bad_rate_multiplier, 1.0)
        n_bad = int(round(inferred_rate * n_rej))
        if n_bad <= 0:
            continue
        # Label the worst-scoring rejects in the band as Bad (deterministic).
        idx = np.where(rej_mask)[0]
        worst = idx[np.argsort(-rej_score[idx])[:n_bad]]
        inferred[worst] = 1

    X_aug = pd.concat([accepts_X, rejects_X], ignore_index=True)
    y_aug = np.concatenate([accepts_y, inferred])
    return _fit_logit(X_aug, y_aug)


def reweighting(
    accepts_X: pd.DataFrame,
    accepts_y: np.ndarray,
    rejects_X: pd.DataFrame,
    config: Config,
) -> tuple[float, dict[str, float]]:
    """Inverse-propensity reweighting: refit scorecard on accepts with weights 1/P(accept|X)."""
    cols = list(accepts_X.columns)
    X_all = pd.concat([accepts_X, rejects_X], ignore_index=True)
    accepted = np.concatenate([np.ones(len(accepts_X)), np.zeros(len(rejects_X))]).astype(int)
    prop_model = LogisticRegression(penalty=None, max_iter=1000)
    prop_model.fit(X_all[cols], accepted)
    p_accept = prop_model.predict_proba(accepts_X[cols])[:, 1]
    weights = 1.0 / np.clip(p_accept, 1e-3, 1.0)
    return _fit_logit(accepts_X, accepts_y, sample_weight=weights)


def fuzzy_augmentation(
    accepts_X: pd.DataFrame,
    accepts_y: np.ndarray,
    rejects_X: pd.DataFrame,
    config: Config,
) -> tuple[float, dict[str, float]]:
    """Enter each reject twice (Bad weight p_bad, Good weight 1−p_bad); refit weighted."""
    base_int, base_coef = _fit_logit(accepts_X, accepts_y)
    cols = list(base_coef)
    eta = base_int + rejects_X[cols].to_numpy(dtype=float) @ np.array([base_coef[c] for c in cols])
    p_bad = 1.0 / (1.0 + np.exp(-eta))

    X_aug = pd.concat([accepts_X, rejects_X, rejects_X], ignore_index=True)
    y_aug = np.concatenate([accepts_y, np.ones(len(rejects_X), int), np.zeros(len(rejects_X), int)])
    w_aug = np.concatenate([np.ones(len(accepts_X)), p_bad, 1.0 - p_bad])
    return _fit_logit(X_aug, y_aug, sample_weight=w_aug)


_METHODS = {
    "parceling": parceling,
    "reweighting": reweighting,
    "fuzzy_augmentation": fuzzy_augmentation,
}


# --------------------------------------------------------------------------- #
# Orchestration + artifact
# --------------------------------------------------------------------------- #
@dataclass
class RejectInferenceResult:
    baseline_intercept: float
    baseline_coef: dict[str, float]
    methods: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_reject_inference(
    accepts_X: pd.DataFrame,
    accepts_y: np.ndarray | pd.Series,
    rejects_X: pd.DataFrame,
    config: Config,
    *,
    eval_X: pd.DataFrame | None = None,
    eval_y: np.ndarray | pd.Series | None = None,
    methods: list[str] | None = None,
) -> RejectInferenceResult:
    """Run reject-inference methods and report coefficient + Gini shift vs KGB baseline.

    ``eval_X``/``eval_y`` (defaults to the accepts) is where Gini is measured. The point is
    the *sensitivity across methods*, not a single picked winner (see module guidance).
    """
    accepts_y = np.asarray(accepts_y).astype(int)
    eval_X = accepts_X if eval_X is None else eval_X
    eval_y = accepts_y if eval_y is None else np.asarray(eval_y).astype(int)
    method_names = methods or list(_METHODS)

    base_int, base_coef = _fit_logit(accepts_X, accepts_y)
    base_gini = _gini_of(base_int, base_coef, eval_X, eval_y)

    result = RejectInferenceResult(baseline_intercept=base_int, baseline_coef=base_coef)
    for name in method_names:
        intercept, coef = _METHODS[name](accepts_X, accepts_y, rejects_X, config)
        shift_l2 = float(np.sqrt(sum((coef[c] - base_coef[c]) ** 2 for c in base_coef)))
        method_gini = _gini_of(intercept, coef, eval_X, eval_y)
        result.methods[name] = {
            "intercept": intercept,
            "coef": coef,
            "coef_shift_l2": shift_l2,
            "gini_kgb": base_gini,
            "gini_method": method_gini,
            "gini_shift": (method_gini - base_gini) if np.isfinite(method_gini) else float("nan"),
        }
        logger.info(
            "Reject inference [%s]: coef L2 shift=%.4f, Gini %.4f -> %.4f.",
            name,
            shift_l2,
            base_gini,
            method_gini,
        )
    return result


def save_reject_inference_sensitivity(result: RejectInferenceResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / REJECT_INFERENCE_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved reject inference sensitivity to %s", path)
    return path
