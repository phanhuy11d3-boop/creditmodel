"""Feature selection: IV filter, iterative VIF, and forward selection.

All steps run on the **train** WoE matrix only. The three steps are kept
strictly separate and each is auditable via the returned :class:`SelectionResult`
trail (used verbatim in the MDD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

_FORWARD_EPS = 1e-4


def woe_col(feature: str) -> str:
    return f"woe_{feature}"


def base_feature(woe_column: str) -> str:
    return woe_column[len("woe_") :] if woe_column.startswith("woe_") else woe_column


@dataclass
class SelectionResult:
    iv_table: pd.DataFrame
    dropped_low_iv: list[str]
    suspicious_iv: list[str]
    vif_dropped: list[tuple[str, float]]
    vif_final: dict[str, float]
    forward_trail: list[tuple[str, float]]
    selected_features: list[str] = field(default_factory=list)  # base feature names

    @property
    def selected_woe_columns(self) -> list[str]:
        return [woe_col(f) for f in self.selected_features]


def iv_filter(
    iv: dict[str, float], config: Config
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """Drop IV < iv_min; flag (keep) IV > iv_suspicious for leakage review."""
    s = config.selection
    rows = []
    kept, dropped, suspicious = [], [], []
    for feat, val in sorted(iv.items(), key=lambda kv: kv[1], reverse=True):
        is_kept = val >= s.iv_min
        is_susp = val > s.iv_suspicious
        rows.append({"feature": feat, "iv": val, "kept": is_kept, "suspicious_leakage": is_susp})
        if is_kept:
            kept.append(feat)
            if is_susp:
                suspicious.append(feat)
        else:
            dropped.append(feat)
    if dropped:
        logger.info("IV filter dropped %d non-predictive features: %s", len(dropped), dropped)
    if suspicious:
        logger.warning(
            "IV > %.2f on %s: FLAGGED for leakage review (kept, not dropped).",
            s.iv_suspicious,
            suspicious,
        )
    return kept, dropped, suspicious, pd.DataFrame(rows)


def iterative_vif(
    X: pd.DataFrame, threshold: float
) -> tuple[list[str], list[tuple[str, float]], dict[str, float]]:
    """Iteratively drop the highest-VIF feature above ``threshold``."""
    cols = list(X.columns)
    dropped: list[tuple[str, float]] = []
    while len(cols) > 1:
        vifs = _vifs(X[cols])
        worst = max(vifs, key=vifs.get)  # type: ignore[arg-type]
        if vifs[worst] > threshold:
            dropped.append((worst, vifs[worst]))
            cols.remove(worst)
            logger.info("VIF drop: %s (VIF=%.2f > %.1f)", worst, vifs[worst], threshold)
        else:
            break
    final = _vifs(X[cols]) if len(cols) > 1 else dict.fromkeys(cols, 1.0)
    return cols, dropped, final


def _vifs(X: pd.DataFrame) -> dict[str, float]:
    mat = add_constant(X, has_constant="add").to_numpy(dtype=float)
    names = list(X.columns)
    out: dict[str, float] = {}
    for i, name in enumerate(names, start=1):  # index 0 is the constant
        try:
            out[name] = float(variance_inflation_factor(mat, i))
        except (np.linalg.LinAlgError, ValueError):
            out[name] = float("inf")
    return out


def _cv_gini(X: pd.DataFrame, y: pd.Series, config: Config) -> float:
    skf = StratifiedKFold(
        n_splits=config.selection.cv_folds, shuffle=True, random_state=config.seed
    )
    aucs = []
    yv = np.asarray(y).astype(int)
    for train_idx, val_idx in skf.split(X, yv):
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X.iloc[train_idx], yv[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(yv[val_idx], proba))
    auc = float(np.mean(aucs))
    return 2 * auc - 1  # Gini; monotonic in AUC so ranking is identical for either metric


def forward_select(
    X: pd.DataFrame, y: pd.Series, candidates: list[str], config: Config
) -> tuple[list[str], list[tuple[str, float]]]:
    """Greedy forward selection maximising CV Gini (== AUC ranking)."""
    selected: list[str] = []
    remaining = list(candidates)
    trail: list[tuple[str, float]] = []
    best_overall = -np.inf
    while remaining:
        scored = [(c, _cv_gini(X[selected + [c]], y, config)) for c in remaining]
        cand, score = max(scored, key=lambda t: t[1])
        if score > best_overall + _FORWARD_EPS:
            selected.append(cand)
            remaining.remove(cand)
            trail.append((cand, score))
            best_overall = score
            logger.info("Forward add: %s (CV gini=%.4f)", cand, score)
        else:
            break
    return selected, trail


def run_selection(
    iv: dict[str, float], X_woe: pd.DataFrame, y: pd.Series, config: Config
) -> SelectionResult:
    """Full selection pipeline on the train WoE matrix."""
    kept, dropped, suspicious, iv_table = iv_filter(iv, config)
    kept_cols = [woe_col(f) for f in kept]

    vif_cols, vif_dropped, vif_final = iterative_vif(
        X_woe[kept_cols], config.selection.vif_threshold
    )

    selected_cols, forward_trail = forward_select(X_woe, y, vif_cols, config)
    selected_features = [base_feature(c) for c in selected_cols]
    if not selected_features:
        raise RuntimeError("Forward selection produced no features; check data/config")
    logger.info("Selected %d features: %s", len(selected_features), selected_features)
    return SelectionResult(
        iv_table=iv_table,
        dropped_low_iv=dropped,
        suspicious_iv=suspicious,
        vif_dropped=[(base_feature(f), v) for f, v in vif_dropped],
        vif_final={base_feature(f): v for f, v in vif_final.items()},
        forward_trail=[(base_feature(f), s) for f, s in forward_trail],
        selected_features=selected_features,
    )
