"""MDD rendering context: the artifacts every chapter reads.

Deterministic-by-construction (DoD §9.10): the context carries only serialized artifacts
(payload, tables, figures, model card) — no wall-clock time — so regenerating the MDD from
the same artifacts yields byte-identical output. The "generated" stamp uses the payload's
frozen ``created_at``/``version``, not ``datetime.now``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from creditscorecard.config import Config


@dataclass
class MddContext:
    payload: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    figures: dict[str, str]
    model_card: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def load_context(config: Config) -> MddContext:
    """Rebuild the MDD context purely from on-disk artifacts (no retraining).

    Powers ``scorecard report`` and proves deterministic regeneration.
    """
    from creditscorecard.registry import load_payload

    payload = load_payload(config)
    artifacts = config.artifacts_path()
    model_card = {}
    card_path = artifacts / "model_card.json"
    if card_path.exists():
        model_card = json.loads(card_path.read_text(encoding="utf-8"))

    tables: dict[str, pd.DataFrame] = {}
    tables_dir = artifacts / "tables"
    if tables_dir.exists():
        for csv in sorted(tables_dir.glob("*.csv")):
            tables[csv.stem] = pd.read_csv(csv)

    fig_dir = config.reports_path() / "figures"
    figures = {}
    if fig_dir.exists():
        for name in (
            "roc",
            "cap",
            "calibration",
            "score_distribution",
            "reliability_curve",
            "shap_summary",
        ):
            candidate = fig_dir / _figure_filename(name)
            if candidate.exists():
                figures[name] = str(candidate)

    return MddContext(payload=payload, tables=tables, figures=figures, model_card=model_card)


def _figure_filename(name: str) -> str:
    mapping = {
        "roc": "roc_curve.png",
        "cap": "cap_curve.png",
        "calibration": "calibration.png",
        "score_distribution": "score_distribution.png",
        "reliability_curve": "reliability_curve.png",
        "shap_summary": "shap_summary.png",
    }
    return mapping.get(name, f"{name}.png")


def generated_stamp(payload: dict[str, Any]) -> str:
    """Deterministic provenance line from the frozen payload (never wall-clock time)."""
    return (
        f"*Version:* `{payload.get('version', 'n/a')}`  ·  "
        f"*Built:* {payload.get('created_at', 'n/a')}  ·  "
        f"*Package:* {payload.get('package_version', '')}"
    )
