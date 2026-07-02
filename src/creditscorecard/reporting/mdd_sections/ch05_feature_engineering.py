"""Chapter 5 — Feature engineering: binning tables, WoE/IV, monotonicity (§6.5)."""

from __future__ import annotations

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 5, "Feature Engineering — Binning, WoE & IV", "05_feature_engineering"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    tables = ctx.tables
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if "iv" in tables:
        parts.append("### Information Value summary\n")
        parts.append(df_to_md(tables["iv"]) + "\n")
    for feat in p.get("selected_features", []):
        key = f"woe_{feat}"
        if key in tables:
            iv = p.get("iv", {}).get(feat, float("nan"))
            parts.append(f"### {feat}  (IV = {iv:.4f})\n")
            parts.append(df_to_md(tables[key]) + "\n")
    parts.append(
        "- WoE = ln(%Good/%Bad); bins are monotonic in event rate (OptBinning "
        "`auto_asc_desc`), frozen at development and reused verbatim in scoring/monitoring.\n"
    )
    return "".join(parts)
