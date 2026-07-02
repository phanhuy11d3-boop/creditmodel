"""Explainability: SHAP + points-vs-SHAP parity (refactor §5.8).

Two explanation regimes:

* **Reportable (WoE-logistic) model** — being linear in WoE, its SHAP values on the
  log-odds scale are *exact* and closed-form: ``phi_j(x) = beta_j · (woe_j(x) − E[woe_j])``
  with base value ``intercept + Σ beta_j·E[woe_j]``. This matches ``shap.LinearExplainer``
  exactly and needs no SHAP dependency, so the serving ``/explain`` endpoint stays light.
* **Challenger (tree) model** — uses ``shap.TreeExplainer`` when the ``shap`` package is
  available, else falls back to the model's impurity ``feature_importances_`` (a documented
  degradation, diagnostic risk R4).

Also computes **interpretability parity**: Jaccard overlap between the reportable model's
top-K features (by |coefficient|) and the challenger's top-K (by mean |SHAP|), and the
agreement between points-based adverse-action reasons and SHAP-based reasons for an applicant.

Artifacts: ``artifacts/global_importance.json`` + ``reports/figures/shap_summary.png``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.features.selection import woe_col
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

GLOBAL_IMPORTANCE_FILE = "global_importance.json"


# --------------------------------------------------------------------------- #
# Reportable model: exact linear SHAP
# --------------------------------------------------------------------------- #
def woe_means(X_woe: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Reference (expected) WoE per feature — the SHAP background expectation."""
    return {f: float(X_woe[woe_col(f)].mean()) for f in features}


def linear_shap_row(
    coefficients: dict[str, float],
    woe_values: dict[str, float],
    means: dict[str, float],
) -> dict[str, float]:
    """Exact SHAP contributions (log-odds) for one applicant under the linear model."""
    return {f: coefficients[f] * (woe_values[f] - means[f]) for f in coefficients}


def explain_applicant(
    coefficients: dict[str, float],
    woe_values: dict[str, float],
    means: dict[str, float],
    top_n: int = 4,
) -> list[dict]:
    """Top-N local SHAP reasons as ``[{feature, shap, direction}]``.

    Direction ``increases_risk`` when the contribution raises log-odds of Bad (positive
    SHAP on P(Bad)); ``decreases_risk`` otherwise.
    """
    shap = linear_shap_row(coefficients, woe_values, means)
    ordered = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out = []
    for feat, val in ordered[:top_n]:
        out.append(
            {
                "feature": feat,
                "shap": float(val),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
        )
    return out


def reportable_global_importance(
    coefficients: dict[str, float], X_woe: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Mean |SHAP| per feature for the reportable model (exact linear SHAP)."""
    means = woe_means(X_woe, features)
    out: dict[str, float] = {}
    for f in features:
        contrib = coefficients[f] * (X_woe[woe_col(f)].to_numpy(dtype=float) - means[f])
        out[f] = float(np.mean(np.abs(contrib)))
    return out


# --------------------------------------------------------------------------- #
# Challenger model: TreeExplainer (SHAP) with importance fallback
# --------------------------------------------------------------------------- #
def challenger_global_importance(
    model, X: pd.DataFrame, *, sample_size: int, seed: int
) -> tuple[dict[str, float], np.ndarray | None]:
    """Mean |SHAP| per feature for the challenger. Returns ``(importance, shap_matrix)``.

    Uses ``shap.TreeExplainer`` when available; otherwise falls back to impurity
    importances and returns ``shap_matrix=None`` (no per-row SHAP for the summary plot).
    """
    n = min(sample_size, len(X))
    Xs = X.sample(n=n, random_state=seed) if n < len(X) else X
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs, check_additivity=False)
        arr = np.asarray(sv)
        # Binary classifiers may return (n, features, 2) or a list; take the positive class.
        if arr.ndim == 3:
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr[..., 0]
        elif isinstance(sv, list):
            arr = np.asarray(sv[-1])
        importance = {col: float(np.mean(np.abs(arr[:, i]))) for i, col in enumerate(Xs.columns)}
        return importance, arr
    except Exception as exc:  # noqa: BLE001 - shap missing or incompatible model
        logger.warning(
            "SHAP TreeExplainer unavailable (%s); falling back to feature_importances_.",
            type(exc).__name__,
        )
        fi = getattr(model, "feature_importances_", None)
        if fi is None:
            return {}, None
        return {col: float(v) for col, v in zip(Xs.columns, fi, strict=False)}, None


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #
def top_k_features(importance: dict[str, float], k: int) -> list[str]:
    return [f for f, _ in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 0.0


def points_vs_shap_agreement(points_reasons: list[str], shap_reasons: list[str]) -> float:
    """Fraction of the points-based top reasons that also appear in the SHAP top reasons."""
    if not points_reasons:
        return float("nan")
    hits = sum(1 for f in points_reasons if f in set(shap_reasons))
    return float(hits / len(points_reasons))


# --------------------------------------------------------------------------- #
# Orchestration + artifact + figure
# --------------------------------------------------------------------------- #
@dataclass
class ExplainabilityResult:
    reportable_importance: dict[str, float] = field(default_factory=dict)
    challenger_importance: dict[str, float] = field(default_factory=dict)
    interpretability_parity: dict = field(default_factory=dict)
    method: str = "linear_shap+tree_shap"

    def to_dict(self) -> dict:
        return asdict(self)


def compute_explainability(
    coefficients: dict[str, float],
    X_woe_train: pd.DataFrame,
    features: list[str],
    config: Config,
    challenger=None,
    X_challenger: pd.DataFrame | None = None,
) -> tuple[ExplainabilityResult, np.ndarray | None, pd.DataFrame | None]:
    """Global importances for both models + interpretability parity (Jaccard)."""
    e = config.explainability
    reportable = reportable_global_importance(coefficients, X_woe_train, features)

    challenger_imp: dict[str, float] = {}
    shap_matrix = None
    Xs = None
    if challenger is not None and X_challenger is not None:
        Xs = (
            X_challenger.sample(
                n=min(e.shap_sample_size, len(X_challenger)), random_state=config.seed
            )
            if e.shap_sample_size < len(X_challenger)
            else X_challenger
        )
        challenger_imp, shap_matrix = challenger_global_importance(
            challenger, Xs, sample_size=len(Xs), seed=config.seed
        )

    k = e.interpretability_parity_top_k
    rep_top = top_k_features({f: abs(v) for f, v in coefficients.items()}, k)
    chal_top = top_k_features(challenger_imp, k) if challenger_imp else []
    parity = {
        "top_k": k,
        "reportable_top": rep_top,
        "challenger_top": chal_top,
        "jaccard": jaccard(rep_top, chal_top) if chal_top else float("nan"),
    }

    result = ExplainabilityResult(
        reportable_importance=reportable,
        challenger_importance=challenger_imp,
        interpretability_parity=parity,
        method="linear_shap+tree_shap" if shap_matrix is not None else "linear_shap+importances",
    )
    logger.info(
        "Explainability: reportable=%d feats, challenger=%d feats, parity Jaccard=%.3f.",
        len(reportable),
        len(challenger_imp),
        parity["jaccard"] if isinstance(parity["jaccard"], float) else float("nan"),
    )
    return result, shap_matrix, Xs


def plot_shap_summary(
    result: ExplainabilityResult,
    out_dir: Path,
    shap_matrix: np.ndarray | None = None,
    X_sample: pd.DataFrame | None = None,
) -> Path:
    """Bar chart of global importance (SHAP beeswarm when a SHAP matrix is available)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shap_summary.png"

    if shap_matrix is not None and X_sample is not None:
        try:
            import shap

            plt.figure()
            shap.summary_plot(shap_matrix, X_sample, show=False, plot_size=(7, 5))
            plt.tight_layout()
            plt.savefig(path, dpi=110)
            plt.close()
            logger.info("Saved figure: %s (SHAP beeswarm).", path.name)
            return path
        except Exception as exc:  # noqa: BLE001 - fall back to a bar chart
            logger.warning("SHAP summary plot failed (%s); using bar chart.", type(exc).__name__)

    imp = result.challenger_importance or result.reportable_importance
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f for f, _ in items], [v for _, v in items])
    ax.set(xlabel="Mean |SHAP| / importance", title="Global feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Saved figure: %s (importance bar).", path.name)
    return path


def save_global_importance(result: ExplainabilityResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / GLOBAL_IMPORTANCE_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved global importance to %s", path)
    return path
