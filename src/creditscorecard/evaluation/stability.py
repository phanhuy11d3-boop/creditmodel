"""Population/Characteristic Stability Index (PSI/CSI) with FROZEN references.

Constraint 11: the score-distribution bin edges and per-characteristic bin
references are established **once at development** and stored as artifacts.
Monitoring on new data **reuses these frozen edges** via ``np.digitize``.

There is deliberately **no** function in this module that re-derives score bins
from new data (no ``qcut``/quantile on the scoring side). :func:`freeze_reference`
is the only entry that computes edges and it is called exactly once on the
development sample. ``assert_is_reference`` guards against passing a new-data
frame where a frozen reference is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


@dataclass
class StabilityReference:
    """Frozen reference distributions established at development time."""

    score_edges: list[float]  # interior score-bin cut points (len = psi_bins - 1)
    score_ref_pct: list[float]  # reference distribution over score bins
    csi_ref_pct: dict[str, dict[int, float]]  # feature -> {code -> reference pct}
    developed: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["csi_ref_pct"] = {
            f: {str(k): v for k, v in m.items()} for f, m in self.csi_ref_pct.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StabilityReference:
        return cls(
            score_edges=list(d["score_edges"]),
            score_ref_pct=list(d["score_ref_pct"]),
            csi_ref_pct={
                f: {int(k): float(v) for k, v in m.items()} for f, m in d["csi_ref_pct"].items()
            },
            developed=bool(d.get("developed", True)),
        )


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    e = np.clip(expected_pct, _EPS, None)
    a = np.clip(actual_pct, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _score_bin_counts(scores: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.digitize(np.asarray(scores, dtype=float), np.asarray(edges), right=False)
    n_bins = len(edges) + 1
    return np.bincount(idx, minlength=n_bins).astype(float)


def freeze_reference(
    dev_scores: np.ndarray, dev_codes: pd.DataFrame, config: Config
) -> StabilityReference:
    """Establish frozen references ONCE on the development sample.

    This is the only place score-bin edges are derived (via quantiles). New data
    never reaches this function.
    """
    n_bins = config.monitoring.psi_bins
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(np.asarray(dev_scores, dtype=float), quantiles)).tolist()
    counts = _score_bin_counts(dev_scores, edges)
    score_ref_pct = (counts / counts.sum()).tolist()

    csi_ref: dict[str, dict[int, float]] = {}
    for col in dev_codes.columns:
        vc = dev_codes[col].value_counts(normalize=True)
        csi_ref[col] = {
            int(code): float(pct)
            for code, pct in zip(vc.index.tolist(), vc.to_numpy().tolist(), strict=True)
        }

    logger.info(
        "Froze stability reference: %d score bins, %d characteristics.",
        len(edges) + 1,
        len(csi_ref),
    )
    return StabilityReference(score_edges=edges, score_ref_pct=score_ref_pct, csi_ref_pct=csi_ref)


def assert_is_reference(ref: StabilityReference) -> None:
    """Guard: PSI/CSI on new data must use a development-frozen reference."""
    if not isinstance(ref, StabilityReference) or not ref.developed:
        raise ValueError("PSI/CSI requires a development-frozen StabilityReference")


def population_stability_index(ref: StabilityReference, new_scores: np.ndarray) -> float:
    """PSI of new scores vs the frozen reference (reuses frozen edges)."""
    assert_is_reference(ref)
    counts = _score_bin_counts(new_scores, ref.score_edges)
    actual_pct = counts / counts.sum()
    return _psi(np.asarray(ref.score_ref_pct), actual_pct)


def characteristic_stability_index(
    ref: StabilityReference, new_codes: pd.DataFrame
) -> dict[str, float]:
    """CSI per characteristic vs the frozen per-characteristic references."""
    assert_is_reference(ref)
    out: dict[str, float] = {}
    for feat, ref_map in ref.csi_ref_pct.items():
        if feat not in new_codes.columns:
            continue
        codes = sorted(set(ref_map) | {int(c) for c in new_codes[feat].unique()})
        expected = np.array([ref_map.get(c, 0.0) for c in codes])
        actual_vc = new_codes[feat].value_counts(normalize=True)
        actual = np.array([float(actual_vc.get(c, 0.0)) for c in codes])
        out[feat] = _psi(expected, actual)
    return out


def psi_status(value: float, warn: float, alert: float) -> str:
    """Standard PSI banding: OK ≤ warn < WARN ≤ alert < ALERT."""
    if value > alert:
        return "ALERT"
    if value > warn:
        return "WARN"
    return "OK"


def split_psi(
    ref: StabilityReference, scores_by_split: dict[str, np.ndarray], config: Config
) -> dict[str, dict]:
    """PSI of each development split's score distribution vs the frozen train reference.

    Answers the validator's question "did the score distribution shift from train to
    test/OOT?" — computed at development time (distinct from post-deployment monitoring on
    new data). Reuses the frozen reference edges; the train split scores ~0 PSI by construction.
    """
    warn, alert = config.monitoring.psi_warn, config.monitoring.psi_alert
    out: dict[str, dict] = {}
    for name, scores in scores_by_split.items():
        psi = population_stability_index(ref, np.asarray(scores, dtype=float))
        out[name] = {"psi": psi, "status": psi_status(psi, warn, alert)}
    return out


def herfindahl_hirschman_index(labels: pd.Series | np.ndarray) -> float:
    """Portfolio concentration across rating grades/segments.

    ``HHI = sum(share_i^2)`` over the population share of each distinct label.
    A well-diversified book of ``k`` equally-sized grades has ``HHI = 1/k``;
    HHI rises toward 1.0 as the book concentrates into fewer grades.
    """
    counts = pd.Series(labels).value_counts()
    shares = counts.to_numpy(dtype=float) / counts.sum()
    return float(np.sum(shares**2))
