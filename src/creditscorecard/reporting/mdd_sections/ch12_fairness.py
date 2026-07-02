"""Chapter 12 — Fairness: AIR/SMD/SPD/EOD, proxy scan, verdict (§5.6, §6.12)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 12, "Fairness — Disparate Impact", "12_fairness"


def render(ctx: MddContext) -> str:
    fair = ctx.payload.get("fairness", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if not fair or not fair.get("enabled") or not fair.get("attributes"):
        parts.append(
            "*Fairness N/A: no protected attributes configured/present. ECOA/Reg B "
            "disparate-impact testing should run whenever a protected attribute is available "
            "(see limitations register).*\n"
        )
        return "".join(parts)
    rows = [
        {
            "attribute": a["attribute"],
            "AIR": round(a["adverse_impact_ratio"], 3),
            "SMD": round(a["standardized_mean_difference"], 3),
            "SPD": round(a["statistical_parity_difference"], 3),
            "EOD": round(a["equal_opportunity_difference"], 3),
            "status": a["air_status"],
        }
        for a in fair["attributes"]
    ]
    parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
    parts.append(
        "- AIR < 0.80 breaches the **80% rule** (ECOA/Reg B). "
        f"{'Failure acknowledged in config.' if fair.get('acknowledged_failure') else ''}\n"
    )
    for attr, scan in (fair.get("proxies") or {}).items():
        flagged = [s["feature"] for s in scan if s["flagged"]]
        if flagged:
            parts.append(f"- **Proxy scan ({attr})**: flagged {flagged}\n")
    if fair.get("note"):
        parts.append(f"- {fair['note']}\n")
    return "".join(parts)
