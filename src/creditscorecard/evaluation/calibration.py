"""Calibration backtesting (refactor §5.5).

Distinct from :mod:`creditscorecard.model.calibrate` (which *sets* the anchor). This
module *tests* whether predicted PD levels are trustworthy:

* Overall: Hosmer-Lemeshow, Brier score, Expected Calibration Error (ECE), reliability
  diagram.
* Per rating-grade backtest using the **Jeffreys interval** (EBA Supervisory Handbook on
  IRB Validation 2023 recommends Jeffreys for uncorrelated defaults): for grade *g*, test
  H₀ that the observed default rate is consistent with the forecast PD_g, one-sided.
* **Traffic-light** classification per grade from the binomial distribution of defaults
  under the forecast PD: green below ``green_upper_quantile``, yellow up to
  ``yellow_upper_quantile``, red above.
* **HHI** of rating-grade concentration (flags a book packed into too few grades).

Artifacts: ``artifacts/calibration_backtest.json`` + ``reports/figures/reliability_curve.png``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.evaluation.stability import herfindahl_hirschman_index
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

CALIBRATION_FILE = "calibration_backtest.json"


# --------------------------------------------------------------------------- #
# Overall calibration metrics
# --------------------------------------------------------------------------- #
def brier_score(y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series) -> float:
    """Mean squared error between predicted PD and the 0/1 outcome (lower = better)."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob_bad, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, prob_bad: np.ndarray | pd.Series, n_bins: int = 10
) -> float:
    """ECE: population-weighted mean gap between confidence and accuracy over PD bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def hosmer_lemeshow(
    y_true: np.ndarray, prob_bad: np.ndarray, n_groups: int = 10
) -> tuple[float, float]:
    """HL goodness-of-fit statistic and p-value (deciles of predicted risk)."""
    from scipy import stats

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


def reliability_curve(
    y_true: np.ndarray, prob_bad: np.ndarray, n_bins: int = 10
) -> tuple[list[float], list[float]]:
    """Return ``(mean_predicted, observed_rate)`` over equal-count PD bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    groups = [g for g in np.array_split(np.arange(len(p)), n_bins) if len(g)]
    return [float(p[g].mean()) for g in groups], [float(y[g].mean()) for g in groups]


# --------------------------------------------------------------------------- #
# Per-grade backtest (Jeffreys + traffic light)
# --------------------------------------------------------------------------- #
def jeffreys_interval(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Jeffreys credible interval for a binomial rate (Beta(k+½, n−k+½))."""
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, a, b))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, a, b))
    return lo, hi


def binomial_upper_pvalue(k: int, n: int, pd_forecast: float) -> float:
    """One-sided p-value P(X ≥ k) under X ~ Binomial(n, PD_forecast).

    Small p ⇒ observed defaults exceed what the forecast PD predicts (under-prediction).
    """
    from scipy.stats import binom

    if n == 0:
        return float("nan")
    return float(binom.sf(k - 1, n, pd_forecast))


@dataclass
class GradeBacktest:
    grade: str
    n: int
    forecast_pd: float
    observed_defaults: int
    observed_rate: float
    jeffreys_lower: float
    jeffreys_upper: float
    binomial_pvalue: float
    traffic_light: str  # green | yellow | red

    def to_dict(self) -> dict:
        return asdict(self)


def _traffic_light(k: int, n: int, pd_forecast: float, green_q: float, yellow_q: float) -> str:
    """Green/yellow/red from the binomial default count under the forecast PD."""
    from scipy.stats import binom

    if n == 0:
        return "green"
    green_upper = binom.ppf(green_q, n, pd_forecast)
    yellow_upper = binom.ppf(yellow_q, n, pd_forecast)
    if k <= green_upper:
        return "green"
    if k <= yellow_upper:
        return "yellow"
    return "red"


def build_grade_aggregates(
    grades: np.ndarray | pd.Series, y_true: np.ndarray | pd.Series, pd_hat: np.ndarray | pd.Series
) -> list[dict]:
    """Aggregate per rating grade: n, observed defaults, observed rate, forecast PD."""
    df = pd.DataFrame(
        {
            "grade": np.asarray(grades),
            "y": np.asarray(y_true).astype(int),
            "pd": np.asarray(pd_hat, dtype=float),
        }
    )
    out: list[dict] = []
    for grade, grp in df.groupby("grade", sort=True):
        n = int(len(grp))
        k = int(grp["y"].sum())
        out.append(
            {
                "grade": str(grade),
                "n": n,
                "observed_defaults": k,
                "observed_rate": k / n if n else float("nan"),
                "forecast_pd": float(grp["pd"].mean()),
            }
        )
    return out


def per_grade_backtest(aggregates: list[dict], config: Config) -> list[GradeBacktest]:
    pgb = config.calibration_extended.per_grade_backtest
    tl = pgb.traffic_light
    rows: list[GradeBacktest] = []
    for agg in aggregates:
        n, k, pdf = agg["n"], agg["observed_defaults"], agg["forecast_pd"]
        lo, hi = jeffreys_interval(k, n, pgb.alpha)
        rows.append(
            GradeBacktest(
                grade=agg["grade"],
                n=n,
                forecast_pd=pdf,
                observed_defaults=k,
                observed_rate=agg["observed_rate"],
                jeffreys_lower=lo,
                jeffreys_upper=hi,
                binomial_pvalue=binomial_upper_pvalue(k, n, pdf),
                traffic_light=_traffic_light(
                    k, n, pdf, tl.green_upper_quantile, tl.yellow_upper_quantile
                ),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Orchestration + artifact + figure
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationBacktestResult:
    brier: float
    ece: float
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    hhi_grades: float
    reliability: dict = field(default_factory=dict)
    per_grade: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_calibration_backtest(
    y_true: np.ndarray | pd.Series,
    prob_bad: np.ndarray | pd.Series,
    grades: np.ndarray | pd.Series,
    config: Config,
) -> CalibrationBacktestResult:
    ce = config.calibration_extended
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_bad, dtype=float)

    hl_stat, hl_p = hosmer_lemeshow(y, p, ce.hosmer_lemeshow_groups)
    mean_pred, obs_rate = reliability_curve(y, p, ce.reliability_curve_bins)
    aggregates = build_grade_aggregates(grades, y, p)
    backtest = per_grade_backtest(aggregates, config)

    ece_val = (
        expected_calibration_error(y, p, ce.reliability_curve_bins)
        if ce.compute_ece
        else float("nan")
    )
    result = CalibrationBacktestResult(
        brier=brier_score(y, p) if ce.compute_brier else float("nan"),
        ece=ece_val,
        hosmer_lemeshow_stat=hl_stat,
        hosmer_lemeshow_pvalue=hl_p,
        hhi_grades=herfindahl_hirschman_index(grades),
        reliability={"mean_predicted": mean_pred, "observed_rate": obs_rate},
        per_grade=[b.to_dict() for b in backtest],
    )
    reds = sum(1 for b in backtest if b.traffic_light == "red")
    logger.info(
        "Calibration backtest: Brier=%.4f ECE=%.4f HL_p=%.3f; %d/%d grades RED.",
        result.brier,
        result.ece,
        hl_p,
        reds,
        len(backtest),
    )
    return result


def plot_reliability_curve(result: CalibrationBacktestResult, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = result.reliability
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.plot(rel["mean_predicted"], rel["observed_rate"], "o-", label="Model")
    ax.set(
        xlabel="Mean predicted PD",
        ylabel="Observed default rate",
        title=f"Reliability (Brier={result.brier:.3f}, ECE={result.ece:.3f})",
    )
    ax.legend(loc="upper left")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reliability_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)
    return path


def save_calibration_backtest(result: CalibrationBacktestResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / CALIBRATION_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved calibration backtest to %s", path)
    return path
