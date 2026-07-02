"""Score new applicants from frozen artifacts only (no training dependencies).

Reconstructs the binning specs, points card, master scale, and stability
reference from ``model.json`` and reuses :meth:`ScorecardModel.score_codes` so
the served score is identical to the pipeline score (proved by
``test_scoring_parity``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from creditscorecard.config import Config
from creditscorecard.evaluation.explainability import explain_applicant, points_vs_shap_agreement
from creditscorecard.features.binning import FeatureBinning, assign_codes
from creditscorecard.model.scorecard import MasterScale, ScorecardModel
from creditscorecard.reasons import ReasonCode, compute_reasons
from creditscorecard.registry import load_payload


class ScoringModel:
    """Artifact-only scorer used by both the pipeline and the API."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.version: str = payload.get("version", "unknown")
        self.target: str = payload["target"]
        self.date_column: str | None = payload.get("date_column")
        self.selected_features: list[str] = payload["selected_features"]
        self.woe_maps: dict[str, dict[int, float]] = {
            f: {int(k): float(v) for k, v in m.items()} for f, m in payload["woe_maps"].items()
        }
        self.binning_specs: dict[str, FeatureBinning] = {
            f: FeatureBinning.from_dict(d) for f, d in payload["binning_specs"].items()
        }
        self.labels: dict[str, dict[int, str]] = {
            f: spec.labels for f, spec in self.binning_specs.items()
        }
        self.points_card: dict[str, dict[int, float]] = {
            f: {int(k): float(v) for k, v in card.items()}
            for f, card in payload["points_card"].items()
        }
        # Reportable-model coefficients + reference WoE means for exact linear-SHAP
        # explanations (§5.8); default to neutral means for legacy artifacts.
        self.coefficients: dict[str, float] = {
            f: float(v) for f, v in payload.get("model", {}).get("coefficients", {}).items()
        }
        self.woe_means: dict[str, float] = {
            f: float(payload.get("woe_means", {}).get(f, 0.0)) for f in self.selected_features
        }
        sc = payload["scaling"]
        ms = payload["master_scale"]
        master_scale = MasterScale(
            grades=list(ms["grades"]),
            score_edges=[float(e) for e in ms["score_edges"]],
            table=ms.get("table", []),
        )
        self.scorecard = ScorecardModel(
            factor=float(sc["factor"]),
            offset=float(sc["offset"]),
            pdo=float(sc["pdo"]),
            target_score=float(sc["target_score"]),
            target_odds=float(sc["target_odds"]),
            intercept=float(payload["calibration"]["calibrated_intercept"]),
            features=list(self.selected_features),
            points_card=self.points_card,
            round_points=bool(sc["round_points"]),
            master_scale=master_scale,
        )

    @classmethod
    def from_config(cls, config: Config) -> ScoringModel:
        return cls(load_payload(config))

    # ---- core scoring --------------------------------------------------- #
    def codes_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.selected_features if f not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")
        data = {f: assign_codes(self.binning_specs[f], df[f]) for f in self.selected_features}
        return pd.DataFrame(data, index=df.index)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scorecard.score_codes(self.codes_frame(df))

    def score_one(self, applicant: dict[str, Any], top_k: int = 4) -> dict[str, Any]:
        row = pd.DataFrame([applicant])
        codes = self.codes_frame(row)
        scored = self.scorecard.score_codes(codes)
        feature_points = {f: float(scored[f].iloc[0]) for f in self.selected_features}
        feature_codes = {f: int(codes[f].iloc[0]) for f in self.selected_features}
        reasons: list[ReasonCode] = compute_reasons(
            feature_points, feature_codes, self.points_card, self.labels, top_k=top_k
        )
        return {
            "score": int(scored["total_score"].iloc[0]),
            "pd": float(scored["pd"].iloc[0]),
            "rating_grade": str(scored["rating_grade"].iloc[0]),
            "points_breakdown": {f: round(v, 2) for f, v in feature_points.items()},
            "reason_codes": [r.to_dict() for r in reasons],
        }

    def explain_one(self, applicant: dict[str, Any], top_n: int = 4) -> dict[str, Any]:
        """Score + points reason codes + exact linear-SHAP reasons + parity (§5.8)."""
        base = self.score_one(applicant, top_k=top_n)
        codes = self.codes_frame(pd.DataFrame([applicant]))
        woe_values = {
            f: float(self.woe_maps[f].get(int(codes[f].iloc[0]), 0.0))
            for f in self.selected_features
        }
        shap_reasons = explain_applicant(self.coefficients, woe_values, self.woe_means, top_n=top_n)
        points_feats = [r["feature"] for r in base["reason_codes"]]
        shap_feats = [r["feature"] for r in shap_reasons]
        return {
            **base,
            "shap_reasons": shap_reasons,
            "points_vs_shap_agreement": points_vs_shap_agreement(points_feats, shap_feats),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": self.target,
            "positive_class": "Good (WoE = ln(%Good/%Bad)); model estimates P(Bad)",
            "selected_features": self.selected_features,
            "scaling": {
                "factor": self.scorecard.factor,
                "offset": self.scorecard.offset,
                "pdo": self.scorecard.pdo,
                "target_score": self.scorecard.target_score,
                "target_odds": self.scorecard.target_odds,
            },
            "rating_grades": self.scorecard.master_scale.grades
            if self.scorecard.master_scale
            else [],
        }
