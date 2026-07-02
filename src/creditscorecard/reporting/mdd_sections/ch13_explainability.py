"""Chapter 13 — Explainability: global importance, points-vs-SHAP parity (§5.8, §6.13)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md, fig_rel
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 13, "Explainability", "13_explainability"


def render(ctx: MddContext) -> str:
    ex = ctx.payload.get("explainability", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if not ex:
        parts.append("*Explainability artifacts unavailable.*\n")
        return "".join(parts)
    parts.append(f"- **Method:** {ex.get('method')}\n")
    rep = ex.get("reportable_importance", {})
    if rep:
        top = sorted(rep.items(), key=lambda kv: kv[1], reverse=True)[:10]
        tbl = pd.DataFrame([{"feature": f, "mean_abs_shap": round(v, 4)} for f, v in top])
        parts.append("### Reportable model — global importance (mean |linear SHAP|)\n")
        parts.append(df_to_md(tbl) + "\n")
    if "shap_summary" in ctx.figures:
        parts.append(f"\n![shap summary]({fig_rel(ctx.figures['shap_summary'])})\n")
    parts.append(
        "- Reportable SHAP is exact (linear); challenger SHAP via TreeExplainer. Serving "
        "`/explain` returns points-based and SHAP-based reasons plus their agreement rate.\n"
    )
    return "".join(parts)
