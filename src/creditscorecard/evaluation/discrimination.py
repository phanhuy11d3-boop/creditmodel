"""Discrimination with uncertainty (refactor §5.3).

Replaces the point-estimate-only ``metrics.py``: every discrimination metric now
carries a bootstrap confidence interval, and in-sample performance is optimism-corrected.

Metrics (on train / test / OOT):

* AUC, Gini (= 2·AUC − 1), KS.
* Partial AUC over a decision-relevant FPR range (McClish-standardised).
* Somers' D_xy — for a binary outcome this equals Gini; computed from the concordance
  count and reported alongside for validator familiarity.
* Bootstrap CIs (BCa / percentile / basic) with a seeded RNG (reproducibility, §10).
* Lift and cumulative gains at deciles.
* Optimism correction for the *train* sample via the .632+ bootstrap estimator
  (Efron & Gong 1983; Efron & Tibshirani 1997) — refits the reportable logistic form
  on the WoE design so the correction reflects the model actually shipped.

Guidance (EBA Supervisory Handbook on IRB Validation 2023, §5.3): discriminatory power
must be reported with uncertainty, not as a bare number, and in-sample optimism must be
acknowledged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

DISCRIMINATION_FILE = "discrimination.json"


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
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


def partial_auc(
    y_true: np.ndarray | pd.Series,
    prob_bad: np.ndarray | pd.Series,
    fpr_range: tuple[float, float] = (0.0, 0.4),
    standardized: bool = True,
) -> float:
    """Partial AUC over an FPR sub-range (approval-decision-relevant region).

    Returns the McClish-standardised pAUC in ``[0.5, 1]`` when ``standardized`` (0.5 =
    chance over the sub-range), else the raw area normalised by the FPR width.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    lo, hi = fpr_range
    # Interpolate the ROC onto a dense grid within [lo, hi] and integrate.
    grid = np.linspace(lo, hi, 512)
    tpr_i = np.interp(grid, fpr, tpr)
    area = float(np.sum(np.diff(grid) * (tpr_i[:-1] + tpr_i[1:]) / 2.0))  # trapezoid rule
    width = hi - lo
    if width <= 0:
        return float("nan")
    raw = area / width  # average TPR over the sub-range (in [0, 1])
    if not standardized:
        return raw
    # McClish standardisation: min area over [lo,hi] is the chance triangle strip.
    min_area = 0.5 * (hi**2 - lo**2)
    return float(0.5 * (1.0 + (area - min_area) / (width - min_area)))


def somers_d(y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series) -> float:
    """Somers' D_xy = (concordant − discordant) / informative pairs = 2·AUC − 1."""
    return 2.0 * auc(y_true, prob_bad) - 1.0


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the two closed intervals overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


def gini_stability_verdict(per_split: dict) -> dict:
    """Compare train vs OOT Gini bootstrap CIs to judge whether the drop is significant.

    Overlapping CIs ⇒ the train→OOT change is within sampling noise (no evidence of drift);
    disjoint CIs ⇒ a statistically meaningful degradation on the out-of-time sample.
    """
    train = per_split.get("train", {}).get("gini")
    oot = per_split.get("oot", {}).get("gini")
    if not train or not oot:
        return {}
    a = (float(train["lower"]), float(train["upper"]))
    b = (float(oot["lower"]), float(oot["upper"]))
    overlap = intervals_overlap(a, b)
    verdict = (
        "Train and OOT Gini CIs overlap → the train→OOT drop is within sampling noise "
        "(no evidence of drift)."
        if overlap
        else "Train and OOT Gini CIs are disjoint → statistically significant degradation on "
        "OOT (possible drift; investigate / revalidate)."
    )
    return {
        "train_gini_ci": [a[0], a[1]],
        "oot_gini_ci": [b[0], b[1]],
        "point_drop": float(train["point"]) - float(oot["point"]),
        "ci_overlap": overlap,
        "verdict": verdict,
    }


def lift_gains(
    y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series, n_bands: int = 10
) -> list[dict]:
    """Decile lift & cumulative gains, ordered by descending predicted PD."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    order = np.argsort(-p)
    y = y[order]
    total_bad = int(y.sum())
    base_rate = float(y.mean()) if len(y) else float("nan")
    bands = np.array_split(np.arange(len(y)), n_bands)
    rows: list[dict] = []
    cum_bad = 0
    cum_n = 0
    for i, b in enumerate(bands, start=1):
        if len(b) == 0:
            continue
        n_b = len(b)
        bad_b = int(y[b].sum())
        cum_bad += bad_b
        cum_n += n_b
        band_rate = bad_b / n_b
        rows.append(
            {
                "decile": i,
                "count": n_b,
                "bad_rate": band_rate,
                "lift": (band_rate / base_rate) if base_rate else float("nan"),
                "cum_pct_population": cum_n / len(y),
                "cum_pct_bads_captured": (cum_bad / total_bad) if total_bad else float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
@dataclass
class MetricCI:
    point: float
    lower: float
    upper: float
    method: str
    level: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile_ci(boot: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def _basic_ci(boot: np.ndarray, point: float, level: float) -> tuple[float, float]:
    lo, hi = _percentile_ci(boot, level)
    return float(2 * point - hi), float(2 * point - lo)


def _bca_ci(
    boot: np.ndarray, point: float, jackknife: np.ndarray, level: float
) -> tuple[float, float]:
    """Bias-corrected & accelerated interval (Efron 1987)."""
    from scipy.stats import norm

    boot = boot[np.isfinite(boot)]
    if len(boot) < 2:
        return float("nan"), float("nan")
    # Bias-correction z0 from the fraction of bootstrap replicates below the point.
    prop = float(np.mean(boot < point))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop)
    # Acceleration from jackknife skewness.
    jk_mean = jackknife.mean()
    diff = jk_mean - jackknife
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    accel = float(np.sum(diff**3) / denom) if denom != 0 else 0.0
    a = (1 - level) / 2
    zlo, zhi = norm.ppf(a), norm.ppf(1 - a)

    def adj(z: float) -> float:
        val = z0 + (z0 + z) / (1 - accel * (z0 + z))
        return float(norm.cdf(val))

    lo = float(np.quantile(boot, adj(zlo)))
    hi = float(np.quantile(boot, adj(zhi)))
    return lo, hi


def bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    prob_bad: np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iter: int,
    level: float,
    method: str,
    seed: int,
) -> MetricCI:
    """Seeded bootstrap CI for ``metric_fn`` (reproducible per §10)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    n = len(y)
    point = float(metric_fn(y, p))
    rng = np.random.default_rng(seed)

    boot = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        yi, pi = y[idx], p[idx]
        boot[i] = metric_fn(yi, pi) if yi.sum() not in (0, n) else np.nan
    boot = boot[np.isfinite(boot)]
    if len(boot) < 2:
        return MetricCI(point, float("nan"), float("nan"), method, level)

    if method == "percentile":
        lo, hi = _percentile_ci(boot, level)
    elif method == "basic":
        lo, hi = _basic_ci(boot, point, level)
    else:  # bca
        jk = np.array(
            [metric_fn(np.delete(y, k), np.delete(p, k)) for k in range(n)]
            if n <= 400
            else _jackknife_subsample(y, p, metric_fn, rng)
        )
        lo, hi = _bca_ci(boot, point, jk, level)
    return MetricCI(point, lo, hi, method, level)


def _jackknife_subsample(y, p, metric_fn, rng, k_groups: int = 200) -> np.ndarray:
    """Grouped (delete-d) jackknife for large n, to keep BCa acceleration affordable."""
    n = len(y)
    groups = np.array_split(rng.permutation(n), k_groups)
    out = []
    for g in groups:
        mask = np.ones(n, dtype=bool)
        mask[g] = False
        out.append(metric_fn(y[mask], p[mask]))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Optimism correction (.632+ bootstrap)
# --------------------------------------------------------------------------- #
def optimism_632plus_auc(
    X_woe: pd.DataFrame,
    y: pd.Series,
    woe_columns: list[str],
    *,
    n_iter: int,
    seed: int,
) -> dict[str, float]:
    """Optimism-corrected train AUC/Gini via the .632+ estimator (Efron & Gong 1983).

    Refits the reportable logistic form (unregularised logit on the selected WoE columns)
    on each bootstrap resample and evaluates on the out-of-bag rows. The .632+ weight
    adjusts the apparent/OOB blend by the relative overfitting rate versus the
    no-information AUC (0.5).
    """
    Xv = X_woe[woe_columns].to_numpy(dtype=float)
    yv = np.asarray(y).astype(int)
    n = len(yv)
    rng = np.random.default_rng(seed)

    def fit_auc(tr_idx, ev_idx) -> float:
        ytr = yv[tr_idx]
        if ytr.sum() in (0, len(ytr)):
            return float("nan")
        m = LogisticRegression(penalty=None, max_iter=1000)
        m.fit(Xv[tr_idx], ytr)
        yev = yv[ev_idx]
        if yev.sum() in (0, len(yev)):
            return float("nan")
        return float(roc_auc_score(yev, m.predict_proba(Xv[ev_idx])[:, 1]))

    apparent = fit_auc(np.arange(n), np.arange(n))
    oob_scores: list[float] = []
    for _ in range(n_iter):
        boot_idx = rng.integers(0, n, n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        if oob_mask.sum() == 0:
            continue
        val = fit_auc(boot_idx, np.where(oob_mask)[0])
        if np.isfinite(val):
            oob_scores.append(val)
    oob = float(np.mean(oob_scores)) if oob_scores else apparent

    gamma = 0.5  # no-information AUC
    denom = apparent - gamma
    rel_overfit = 0.0 if denom <= 0 else min(max((apparent - oob) / denom, 0.0), 1.0)
    w = 0.632 / (1.0 - 0.368 * rel_overfit)
    corrected = (1 - w) * apparent + w * oob
    return {
        "apparent_auc": apparent,
        "oob_auc": oob,
        "optimism": apparent - corrected,
        "corrected_auc": corrected,
        "corrected_gini": 2 * corrected - 1,
        "weight_632plus": w,
    }


# --------------------------------------------------------------------------- #
# Orchestration + artifact
# --------------------------------------------------------------------------- #
@dataclass
class DiscriminationResult:
    per_split: dict[str, dict] = field(default_factory=dict)
    optimism: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"per_split": self.per_split, "optimism": self.optimism}


def compute_discrimination(
    splits: dict[str, tuple],
    config: Config,
    *,
    optimism_inputs: tuple[pd.DataFrame, pd.Series, list[str]] | None = None,
) -> DiscriminationResult:
    """Point + bootstrap-CI discrimination on every split, plus .632+ train optimism.

    ``splits`` maps split name -> ``(y_true, prob_bad)``. ``optimism_inputs`` is
    ``(X_woe_train, y_train, woe_columns)`` used for the .632+ correction (skipped if None).
    """
    d = config.discrimination
    result = DiscriminationResult()
    for name, (y_raw, p_raw) in splits.items():
        yv = np.asarray(y_raw).astype(int)
        pv = np.asarray(p_raw, dtype=float)
        entry: dict = {
            "n": int(len(yv)),
            "bad_rate": float(yv.mean()) if len(yv) else float("nan"),
            "auc": bootstrap_ci(
                yv,
                pv,
                auc,
                n_iter=d.bootstrap_iterations,
                level=d.confidence_level,
                method=d.bootstrap_method,
                seed=config.seed,
            ).to_dict(),
            "gini": bootstrap_ci(
                yv,
                pv,
                gini,
                n_iter=d.bootstrap_iterations,
                level=d.confidence_level,
                method=d.bootstrap_method,
                seed=config.seed + 1,
            ).to_dict(),
            "ks": bootstrap_ci(
                yv,
                pv,
                ks,
                n_iter=d.bootstrap_iterations,
                level=d.confidence_level,
                method=d.bootstrap_method,
                seed=config.seed + 2,
            ).to_dict(),
            "lift_gains": lift_gains(yv, pv),
        }
        if d.compute_partial_auc:
            entry["partial_auc"] = partial_auc(yv, pv, d.partial_auc_range)
        if d.compute_somers_d:
            entry["somers_d"] = somers_d(yv, pv)
        result.per_split[name] = entry

    if optimism_inputs is not None:
        X_woe, ytr, woe_cols = optimism_inputs
        result.optimism = optimism_632plus_auc(
            X_woe, ytr, woe_cols, n_iter=min(d.bootstrap_iterations, 200), seed=config.seed
        )
    logger.info(
        "Discrimination computed on %d splits (%s CIs, %d iters).",
        len(result.per_split),
        d.bootstrap_method,
        d.bootstrap_iterations,
    )
    return result


def save_discrimination(result: DiscriminationResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / DISCRIMINATION_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved discrimination metrics to %s", path)
    return path
