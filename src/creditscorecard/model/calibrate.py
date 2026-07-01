"""Calibrate model PD to an anchor / central-tendency default rate.

We solve for a single additive shift ``delta`` on the linear predictor so that
the mean predicted ``P(Bad)`` equals the anchor rate (a monotonic 1-D root).
The shift folds cleanly into the scorecard intercept. When
``anchor_default_rate`` is null we anchor to the train base rate (shift ~ 0,
since the MLE already matches the sample mean).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger
from creditscorecard.model.train import TrainedModel

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    anchor_rate: float
    intercept_shift: float
    mean_pd_before: float
    mean_pd_after: float
    method: str

    def calibrated_intercept(self, model: TrainedModel) -> float:
        return model.intercept + self.intercept_shift


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_shift(eta: np.ndarray, target: float, tol: float = 1e-10) -> float:
    lo, hi = -30.0, 30.0

    def mean_pd(delta: float) -> float:
        return float(np.mean(_sigmoid(eta + delta)))

    # Monotonic increasing in delta -> bisection.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = mean_pd(mid) - target
        if abs(val) < tol:
            return mid
        if val < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(
    model: TrainedModel, X_woe: pd.DataFrame, y: pd.Series, config: Config
) -> CalibrationResult:
    train_base = float(np.mean(np.asarray(y).astype(int)))
    anchor = config.calibration.anchor_default_rate
    method = "anchor_default_rate" if anchor is not None else "train_base_rate"
    target = anchor if anchor is not None else train_base

    eta = model.linear_predictor(X_woe)
    mean_before = float(np.mean(_sigmoid(eta)))
    shift = _solve_shift(eta, target)
    mean_after = float(np.mean(_sigmoid(eta + shift)))

    logger.info(
        "Calibration (%s): target=%.4f, mean PD %.4f -> %.4f (shift=%.4f)",
        method,
        target,
        mean_before,
        mean_after,
        shift,
    )
    return CalibrationResult(
        anchor_rate=target,
        intercept_shift=shift,
        mean_pd_before=mean_before,
        mean_pd_after=mean_after,
        method=method,
    )
