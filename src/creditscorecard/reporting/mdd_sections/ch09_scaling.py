"""Chapter 9 — Scorecard scaling: Factor/Offset/PDO + Master Scale (§6.9)."""

from __future__ import annotations

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 9, "Scorecard Scaling & Master Scale", "09_scaling"


def render(ctx: MddContext) -> str:
    sc = ctx.payload.get("scaling", {})
    factor = sc.get("factor", float("nan"))
    parts = [f"## {NUMBER}. {TITLE}\n"]
    parts.append(
        f"- `Factor = PDO/ln(2)` = **{factor:.4f}** (PDO={sc.get('pdo')}).\n"
        f"- `Offset = TargetScore − Factor·ln(TargetOdds)` = "
        f"**{sc.get('offset', float('nan')):.4f}** "
        f"(TargetScore={sc.get('target_score')}, TargetOdds={sc.get('target_odds')} Good:Bad).\n"
    )
    if "master_scale" in ctx.tables:
        parts.append(df_to_md(ctx.tables["master_scale"]) + "\n")
    return "".join(parts)
