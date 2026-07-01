"""Adverse-action reason codes.

A reason code identifies a characteristic where the applicant lost the most
points relative to the best attainable (neutral/max) baseline for that
characteristic. Larger shortfall == stronger adverse reason. This mirrors
regulatory adverse-action explanations (e.g. ECOA/Reg B).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReasonCode:
    feature: str
    actual_points: float
    max_points: float
    shortfall: float
    bin_label: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "actual_points": round(self.actual_points, 2),
            "max_points": round(self.max_points, 2),
            "shortfall": round(self.shortfall, 2),
            "bin_label": self.bin_label,
        }


def max_points_per_feature(points_card: dict[str, dict[int, float]]) -> dict[str, float]:
    """Best attainable points per characteristic (over normal bins)."""
    out: dict[str, float] = {}
    for feat, card in points_card.items():
        normal = [p for c, p in card.items() if c >= 0]
        out[feat] = float(max(normal)) if normal else 0.0
    return out


def compute_reasons(
    feature_points: dict[str, float],
    feature_codes: dict[str, int],
    points_card: dict[str, dict[int, float]],
    labels: dict[str, dict[int, str]],
    top_k: int = 4,
) -> list[ReasonCode]:
    """Return the top-k characteristics by points shortfall vs their best bin."""
    maxima = max_points_per_feature(points_card)
    reasons: list[ReasonCode] = []
    for feat, actual in feature_points.items():
        mx = maxima.get(feat, actual)
        shortfall = max(mx - actual, 0.0)
        code = feature_codes.get(feat, -999)
        label = labels.get(feat, {}).get(code, str(code))
        reasons.append(
            ReasonCode(
                feature=feat,
                actual_points=float(actual),
                max_points=float(mx),
                shortfall=float(shortfall),
                bin_label=label,
            )
        )
    reasons.sort(key=lambda r: r.shortfall, reverse=True)
    return reasons[:top_k]
