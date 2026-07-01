"""Logistic PD model: statsmodels Logit (reportable) + sklearn parity check.

Orientation chain (see package docstring / MDD):

* Target ``1 == Bad`` -> ``statsmodels.Logit`` estimates ``P(Bad)`` directly.
* Features are ``WoE = ln(%Good/%Bad)`` (positive == better applicant).
* Therefore every WoE coefficient is expected to be **negative** (more Good ->
  lower P(Bad)). Wrong-sign (positive) features are flagged and excluded unless
  explicitly overridden in config.

The selection loop lives in :mod:`features.selection`; here the model is fit
once per candidate feature set. Sign enforcement may drop wrong-sign features
and refit on the reduced set — the *final* reported model is the last fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from creditscorecard.config import Config
from creditscorecard.features.selection import woe_col
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

EXPECTED_SIGN = -1.0  # WoE coefficients should be negative when modeling P(Bad)


@dataclass
class SignCheck:
    feature: str
    coefficient: float
    expected_negative: bool
    ok: bool
    overridden: bool


@dataclass
class ParityResult:
    max_abs_diff: float
    tolerance: float
    passed: bool


@dataclass
class TrainedModel:
    features: list[str]  # base feature names, in model order
    intercept: float
    coefficients: dict[str, float]  # base feature -> beta
    std_errors: dict[str, float]
    p_values: dict[str, float]
    sign_checks: list[SignCheck]
    excluded_wrong_sign: list[str]
    parity: ParityResult
    summary_text: str
    n_obs: int

    @property
    def woe_columns(self) -> list[str]:
        return [woe_col(f) for f in self.features]

    def linear_predictor(self, X_woe: pd.DataFrame) -> np.ndarray:
        eta = np.full(len(X_woe), self.intercept, dtype=float)
        for feat in self.features:
            eta += self.coefficients[feat] * X_woe[woe_col(feat)].to_numpy(dtype=float)
        return eta

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Return P(Bad) for each row."""
        return 1.0 / (1.0 + np.exp(-self.linear_predictor(X_woe)))


def _fit_statsmodels(X_woe: pd.DataFrame, y: pd.Series, cols: list[str]):
    X = sm.add_constant(X_woe[cols], has_constant="add")
    model = sm.Logit(np.asarray(y).astype(int), X)
    return model.fit(disp=0, maxiter=200)


def train_model(
    X_woe: pd.DataFrame, y: pd.Series, features: list[str], config: Config
) -> TrainedModel:
    """Fit the reportable Logit with sign enforcement and an sklearn parity check."""
    overrides = set(config.model.sign_overrides)
    current = list(features)
    excluded: list[str] = []

    while True:
        cols = [woe_col(f) for f in current]
        res = _fit_statsmodels(X_woe, y, cols)
        wrong = [f for f in current if res.params[woe_col(f)] > 0 and f not in overrides]
        if not wrong or not config.model.enforce_sign_check:
            break
        logger.warning("Sign check excluding wrong-sign features: %s", wrong)
        excluded.extend(wrong)
        current = [f for f in current if f not in wrong]
        if not current:
            raise RuntimeError("All features failed the sign check; cannot fit a valid model")

    sign_checks = [
        SignCheck(
            feature=f,
            coefficient=float(res.params[woe_col(f)]),
            expected_negative=True,
            ok=float(res.params[woe_col(f)]) <= 0,
            overridden=f in overrides,
        )
        for f in current
    ]

    parity = _parity_check(X_woe, y, current, res, config)

    coefficients = {f: float(res.params[woe_col(f)]) for f in current}
    std_errors = {f: float(res.bse[woe_col(f)]) for f in current}
    p_values = {f: float(res.pvalues[woe_col(f)]) for f in current}

    logger.info(
        "Final model fit on %d features; intercept=%.4f", len(current), float(res.params["const"])
    )
    return TrainedModel(
        features=current,
        intercept=float(res.params["const"]),
        coefficients=coefficients,
        std_errors=std_errors,
        p_values=p_values,
        sign_checks=sign_checks,
        excluded_wrong_sign=excluded,
        parity=parity,
        summary_text=str(res.summary()),
        n_obs=int(res.nobs),
    )


def _parity_check(
    X_woe: pd.DataFrame, y: pd.Series, features: list[str], sm_res, config: Config
) -> ParityResult:
    """Assert statsmodels and unregularised sklearn coefficients agree."""
    cols = [woe_col(f) for f in features]
    # Tight convergence so the comparison reflects the objective, not the optimiser's
    # early-stopping tolerance (default lbfgs tol=1e-4 leaves a ~1e-3 coefficient gap).
    skl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000, tol=1e-10)
    skl.fit(X_woe[cols], np.asarray(y).astype(int))
    diffs = [abs(sm_res.params["const"] - float(skl.intercept_[0]))]
    for i, f in enumerate(features):
        diffs.append(abs(float(sm_res.params[woe_col(f)]) - float(skl.coef_[0][i])))
    max_diff = float(max(diffs))
    passed = max_diff <= config.model.parity_tol
    log = logger.info if passed else logger.warning
    log(
        "statsmodels/sklearn coefficient parity: max_abs_diff=%.2e (tol=%.1e)",
        max_diff,
        config.model.parity_tol,
    )
    return ParityResult(max_abs_diff=max_diff, tolerance=config.model.parity_tol, passed=passed)
