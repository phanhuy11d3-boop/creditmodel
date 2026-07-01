"""Scorecard scaling (points) and the Master Scale.

Exact formulas (spec constraint 9), with ``alpha`` = calibrated intercept and
``n`` = number of characteristics::

    Factor    = PDO / ln(2)
    Offset    = TargetScore - Factor * ln(TargetOdds)      # TargetOdds = Good:Bad
    points_i  = -(WoE_i * beta_i + alpha/n) * Factor + Offset/n
    TotalScore = sum_i points_i

Consistency: summing the points gives ``Score = Offset - Factor * eta`` where
``eta = logit(P(Bad))``. Hence ``PD = sigmoid((Offset - Score) / Factor)`` is
recovered exactly from the total score — so score and PD can never disagree, and
serving needs only the frozen points card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger
from creditscorecard.model.calibrate import CalibrationResult
from creditscorecard.model.train import TrainedModel

logger = get_logger(__name__)


def compute_factor_offset(
    pdo: float, target_score: float, target_odds: float
) -> tuple[float, float]:
    factor = pdo / math.log(2.0)
    offset = target_score - factor * math.log(target_odds)
    return factor, offset


def compute_points(
    woe: float, beta: float, alpha: float, n: int, factor: float, offset: float
) -> float:
    """The single-attribute points formula (spec constraint 9)."""
    return -(woe * beta + alpha / n) * factor + offset / n


@dataclass
class MasterScale:
    grades: list[str]
    score_edges: list[float]  # interior cut points, length len(grades) - 1
    table: list[dict] = field(default_factory=list)

    def assign_grade(self, scores: np.ndarray) -> np.ndarray:
        idx = np.digitize(
            np.asarray(scores, dtype=float), np.asarray(self.score_edges), right=False
        )
        return np.array([self.grades[i] for i in idx])


@dataclass
class ScorecardModel:
    factor: float
    offset: float
    pdo: float
    target_score: float
    target_odds: float
    intercept: float  # calibrated alpha
    features: list[str]
    points_card: dict[str, dict[int, float]]  # feature -> {code -> raw points}
    round_points: bool
    master_scale: MasterScale | None = None

    # ---- scoring -------------------------------------------------------- #
    def score_codes(self, codes: pd.DataFrame) -> pd.DataFrame:
        """Score a frame of bin codes. Returns points breakdown + total + pd + grade."""
        breakdown = {}
        for feat in self.features:
            card = self.points_card[feat]
            neutral = _neutral_points(card)
            breakdown[feat] = codes[feat].map(card).fillna(neutral).astype(float).to_numpy()
        bd = pd.DataFrame(breakdown, index=codes.index)
        raw_total = bd.sum(axis=1).to_numpy()
        eta = (self.offset - raw_total) / self.factor
        pd_hat = 1.0 / (1.0 + np.exp(-eta))
        total = np.round(raw_total).astype(int) if self.round_points else raw_total
        result = bd.round().astype(int) if self.round_points else bd
        result = result.copy()
        result["total_score"] = total
        result["pd"] = pd_hat
        if self.master_scale is not None:
            result["rating_grade"] = self.master_scale.assign_grade(raw_total)
        return result

    def points_card_serialisable(self) -> dict[str, dict[str, float]]:
        return {f: {str(c): p for c, p in card.items()} for f, card in self.points_card.items()}


def _neutral_points(card: dict[int, float]) -> float:
    """Points for an unseen code: the population-neutral (mean) contribution."""
    normal = [p for c, p in card.items() if c >= 0]
    return float(np.mean(normal)) if normal else 0.0


def build_scorecard(
    model: TrainedModel,
    calibration: CalibrationResult,
    woe_maps: dict[str, dict[int, float]],
    config: Config,
) -> ScorecardModel:
    s = config.scaling
    factor, offset = compute_factor_offset(s.pdo, s.target_score, s.target_odds)
    alpha = calibration.calibrated_intercept(model)
    n = len(model.features)

    points_card: dict[str, dict[int, float]] = {}
    for feat in model.features:
        beta = model.coefficients[feat]
        card = {
            code: compute_points(woe, beta, alpha, n, factor, offset)
            for code, woe in woe_maps[feat].items()
        }
        points_card[feat] = card

    logger.info(
        "Scorecard scaled: Factor=%.4f Offset=%.4f (PDO=%g, target_score=%g, target_odds=%g).",
        factor,
        offset,
        s.pdo,
        s.target_score,
        s.target_odds,
    )
    return ScorecardModel(
        factor=factor,
        offset=offset,
        pdo=s.pdo,
        target_score=s.target_score,
        target_odds=s.target_odds,
        intercept=alpha,
        features=list(model.features),
        points_card=points_card,
        round_points=s.round_points,
    )


def build_master_scale(
    scorecard: ScorecardModel, codes: pd.DataFrame, y: pd.Series, config: Config
) -> MasterScale:
    """Build monotonic score-band -> rating-grade mapping on the train sample."""
    scored = scorecard.score_codes(codes)
    raw_total = (scored["total_score"]).to_numpy(dtype=float)
    pd_hat = scored["pd"].to_numpy()
    y_arr = np.asarray(y).astype(int)

    n_grades = config.scaling.rating_grades
    quantiles = np.linspace(0, 1, n_grades + 1)[1:-1]
    edges = np.unique(np.quantile(raw_total, quantiles)).tolist()
    grades = _grade_labels(len(edges) + 1)

    ms = MasterScale(grades=grades, score_edges=edges)
    band = np.digitize(raw_total, np.asarray(edges), right=False)
    table = []
    for b in range(len(grades)):
        mask = band == b
        n_b = int(mask.sum())
        table.append(
            {
                "grade": grades[b],
                "score_min": float(raw_total[mask].min()) if n_b else float("nan"),
                "score_max": float(raw_total[mask].max()) if n_b else float("nan"),
                "count": n_b,
                "avg_pd": float(pd_hat[mask].mean()) if n_b else float("nan"),
                "observed_bad_rate": float(y_arr[mask].mean()) if n_b else float("nan"),
            }
        )
    ms.table = table
    scorecard.master_scale = ms
    logger.info("Master Scale built with %d grades: %s", len(grades), grades)
    return ms


def _grade_labels(n: int) -> list[str]:
    """Grade labels ordered worst->best by band index (best = highest score)."""
    if n <= 26:
        # band 0 (lowest score) = worst grade; highest band = 'A'.
        return [chr(ord("A") + (n - 1 - i)) for i in range(n)]
    return [f"R{n - i}" for i in range(n)]
