"""Chapter 3 — Reject inference: method, sensitivity, KGB baseline (§5.2, §6.3)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 3, "Reject Inference", "03_reject_inference"


def render(ctx: MddContext) -> str:
    ri = ctx.payload.get("reject_inference", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if not ri or ri.get("enabled") is False or "methods" not in ri:
        parts.append(
            "*Reject inference is **disabled** (no reject data). The development sample is "
            "Known-Good-Bad (KGB) only; through-the-door population selection bias is "
            "uncorrected — see the limitations register.*\n"
        )
        return "".join(parts)
    rows = []
    for name, m in ri.get("methods", {}).items():
        rows.append(
            {
                "method": name,
                "coef_shift_L2": round(m.get("coef_shift_l2", float("nan")), 4),
                "gini_kgb": round(m.get("gini_kgb", float("nan")), 4),
                "gini_method": round(m.get("gini_method", float("nan")), 4),
                "gini_shift": round(m.get("gini_shift", float("nan")), 4),
            }
        )
    parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
    parts.append(
        "- Sensitivity is reported across methods (Banasik & Crook; Bücker et al.): simpler "
        "methods often match complex ones — no single method is presented as definitive.\n"
    )
    return "".join(parts)
