"""Fairness / disparate-impact testing (refactor §5.6).

ECOA / Regulation B disparate-impact doctrine applies to any consumer-lending model, so a
scorecard must be tested for adverse impact across protected classes. Metrics computed per
protected attribute (protected group vs reference group):

* **Adverse Impact Ratio (AIR)** — P(favourable | protected) / P(favourable | reference);
  the *80% rule* flags AIR < ``air_threshold_alert`` (and a softer warn at
  ``air_threshold_warn``).
* **Standardized Mean Difference (SMD)** — score-gap between groups in pooled-SD units.
* **Statistical Parity Difference (SPD)** — favourable-rate difference.
* **Equal Opportunity Difference (EOD)** — gap in the true-positive rate (declined | actually
  Bad) between groups, on the defaulted subset.
* **Proxy scan** — mutual information between each modelling feature and each protected
  attribute; strongly-associated features are flagged as potential proxies.

Protected group derivation: numeric attributes are dichotomised (age-like columns at 25 per
the German Credit convention, otherwise at the median); categorical attributes split the
minority level (protected) from the majority (reference).

Any AIR < ``air_threshold_alert`` raises a **build-failing** error unless
``fairness.acknowledge_failure`` is set — an unfair model must not ship silently.

Artifact: ``artifacts/fairness.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

FAIRNESS_FILE = "fairness.json"


class FairnessBuildError(RuntimeError):
    """Raised when AIR breaches the 80% rule and the failure is not acknowledged."""


def protected_group_mask(series: pd.Series, attribute: str) -> np.ndarray:
    """Boolean mask of the (potentially disadvantaged) *protected* group.

    Numeric → below 25 for age-like columns (German Credit convention), else below the
    median. Categorical → the minority level.
    """
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        threshold = 25.0 if "age" in attribute.lower() else float(values.median())
        return (values < threshold).to_numpy()
    counts = series.value_counts()
    minority = counts.index[-1]
    return (series == minority).to_numpy()


def _rate(favourable: np.ndarray, mask: np.ndarray) -> float:
    return float(favourable[mask].mean()) if mask.any() else float("nan")


@dataclass
class AttributeFairness:
    attribute: str
    protected_n: int
    reference_n: int
    favourable_rate_protected: float
    favourable_rate_reference: float
    adverse_impact_ratio: float
    standardized_mean_difference: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    air_status: str  # OK | WARN | ALERT

    def to_dict(self) -> dict:
        return asdict(self)


def compute_attribute_fairness(
    attribute: str,
    protected: np.ndarray,
    favourable: np.ndarray,
    scores: np.ndarray,
    config: Config,
    y_true: np.ndarray | None = None,
) -> AttributeFairness:
    ref = ~protected
    air_p = _rate(favourable, protected)
    air_r = _rate(favourable, ref)
    air = float(air_p / air_r) if air_r not in (0.0, float("nan")) and air_r > 0 else float("nan")

    # SMD in scores (protected − reference) / pooled SD.
    sp, sr = scores[protected], scores[ref]
    pooled_sd = np.sqrt((np.var(sp) + np.var(sr)) / 2.0) if len(sp) and len(sr) else 0.0
    smd = float((sp.mean() - sr.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
    spd = float(air_p - air_r)

    eod = 0.0
    if y_true is not None:
        bad = y_true == 1
        # TPR = P(declined | Bad) = P(not favourable | Bad), per group on the defaulted subset.
        tpr_p = float((~favourable[protected & bad]).mean()) if (protected & bad).any() else 0.0
        tpr_r = float((~favourable[ref & bad]).mean()) if (ref & bad).any() else 0.0
        eod = tpr_p - tpr_r

    f = config.fairness
    if np.isnan(air):
        status = "OK"
    elif air < f.air_threshold_alert:
        status = "ALERT"
    elif air < f.air_threshold_warn:
        status = "WARN"
    else:
        status = "OK"

    return AttributeFairness(
        attribute=attribute,
        protected_n=int(protected.sum()),
        reference_n=int(ref.sum()),
        favourable_rate_protected=air_p,
        favourable_rate_reference=air_r,
        adverse_impact_ratio=air,
        standardized_mean_difference=smd,
        statistical_parity_difference=spd,
        equal_opportunity_difference=eod,
        air_status=status,
    )


def proxy_scan(
    feature_frame: pd.DataFrame, protected: np.ndarray, *, seed: int, threshold: float = 0.10
) -> list[dict]:
    """Mutual information between each feature and the protected mask; flag strong links."""
    from sklearn.feature_selection import mutual_info_classif

    X = feature_frame.to_numpy(dtype=float)
    mi = mutual_info_classif(X, protected.astype(int), random_state=seed)
    scored = sorted(
        ((str(col), float(val)) for col, val in zip(feature_frame.columns, mi, strict=True)),
        key=lambda t: t[1],
        reverse=True,
    )
    return [
        {"feature": col, "mutual_info": val, "flagged": bool(val > threshold)}
        for col, val in scored
    ]


@dataclass
class FairnessResult:
    enabled: bool
    attributes: list[dict] = field(default_factory=list)
    proxies: dict[str, list[dict]] = field(default_factory=dict)
    acknowledged_failure: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_fairness(
    data: pd.DataFrame,
    favourable: np.ndarray,
    scores: np.ndarray,
    config: Config,
    *,
    y_true: np.ndarray | None = None,
    feature_frame: pd.DataFrame | None = None,
) -> FairnessResult:
    """Compute fairness metrics across configured protected attributes.

    Raises :class:`FairnessBuildError` if any AIR < ``air_threshold_alert`` and
    ``fairness.acknowledge_failure`` is False (an unfair model must not ship silently).
    """
    f = config.fairness
    present = [a for a in f.protected_attributes if a in data.columns]
    if not f.enabled or not present:
        note = (
            "Fairness disabled."
            if not f.enabled
            else "No configured protected attributes present in the data (fairness N/A)."
        )
        logger.info(note)
        return FairnessResult(enabled=f.enabled, note=note)

    attributes: list[dict] = []
    proxies: dict[str, list[dict]] = {}
    alerts: list[str] = []
    for attr in present:
        mask = protected_group_mask(data[attr], attr)
        af = compute_attribute_fairness(attr, mask, favourable, scores, config, y_true)
        attributes.append(af.to_dict())
        if af.air_status == "ALERT":
            alerts.append(f"{attr} (AIR={af.adverse_impact_ratio:.3f})")
        if f.proxy_scan and feature_frame is not None:
            proxies[attr] = proxy_scan(feature_frame, mask, seed=config.seed)

    result = FairnessResult(
        enabled=True,
        attributes=attributes,
        proxies=proxies,
        acknowledged_failure=f.acknowledge_failure,
    )

    if alerts:
        tail = (
            "Acknowledged (fairness.acknowledge_failure=true)."
            if f.acknowledge_failure
            else "BUILD FAILED."
        )
        msg = "Adverse impact below the 80% rule on: " + ", ".join(alerts) + ". " + tail
        if f.acknowledge_failure:
            logger.warning(msg)
            result.note = msg
        else:
            logger.error(msg)
            raise FairnessBuildError(msg)
    else:
        logger.info("Fairness: all protected attributes pass the AIR threshold.")
    return result


def save_fairness(result: FairnessResult, config: Config) -> Path:
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / FAIRNESS_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved fairness report to %s", path)
    return path
