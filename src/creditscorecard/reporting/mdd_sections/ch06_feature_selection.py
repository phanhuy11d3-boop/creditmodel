"""Chapter 6 — Feature selection: IV filter, VIF trail, forward selection (§6.6)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 6, "Feature Selection Trail", "06_feature_selection"


def render(ctx: MddContext) -> str:
    sel = ctx.payload.get("selection_trail", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    parts.append(f"- **Dropped (IV < min):** {sel.get('dropped_low_iv') or 'none'}\n")
    parts.append(
        f"- **Flagged for leakage review (IV > suspicious, kept):** "
        f"{sel.get('suspicious_iv') or 'none'}\n"
    )
    vif_dropped = sel.get("vif_dropped", [])
    parts.append(
        "- **VIF drops:** "
        + (", ".join(f"{f} (VIF={v:.2f})" for f, v in vif_dropped) if vif_dropped else "none")
        + "\n"
    )
    fwd = pd.DataFrame(sel.get("forward_trail", []), columns=["feature_added", "cv_gini"])
    parts.append("### Forward-selection order (CV Gini)\n")
    parts.append(df_to_md(fwd) + "\n")
    return "".join(parts)
