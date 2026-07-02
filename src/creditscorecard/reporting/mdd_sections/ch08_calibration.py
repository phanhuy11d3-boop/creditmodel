"""Chapter 8 — Calibration: anchor, per-grade Jeffreys backtest, traffic light, HHI (§5.5, §6.8)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md, fig_rel
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 8, "Calibration & Backtest", "08_calibration"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    cal = p.get("calibration", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    parts.append(
        f"- Anchor method: **{cal.get('method')}**, anchor rate = "
        f"{cal.get('anchor_rate', float('nan')):.4f}.\n"
        f"- Mean PD before → after: {cal.get('mean_pd_before', float('nan')):.4f} → "
        f"{cal.get('mean_pd_after', float('nan')):.4f} "
        f"(intercept shift = {cal.get('intercept_shift', float('nan')):.4f}).\n"
    )
    cb = p.get("calibration_backtest", {})
    if cb:
        parts.append(
            f"- **Brier:** {cb.get('brier', float('nan')):.4f}  ·  "
            f"**ECE:** {cb.get('ece', float('nan')):.4f}  ·  "
            f"**HL:** stat={cb.get('hosmer_lemeshow_stat', float('nan')):.2f}, "
            f"p={cb.get('hosmer_lemeshow_pvalue', float('nan')):.3f}  ·  "
            f"**Grade HHI:** {cb.get('hhi_grades', float('nan')):.4f}\n"
        )
        per_grade = cb.get("per_grade", [])
        if per_grade:
            tbl = pd.DataFrame(
                [
                    {
                        "grade": g["grade"],
                        "n": g["n"],
                        "forecast_pd": round(g["forecast_pd"], 4),
                        "observed_rate": round(g["observed_rate"], 4),
                        "jeffreys_upper": round(g["jeffreys_upper"], 4),
                        "binom_p": round(g["binomial_pvalue"], 4),
                        "light": g["traffic_light"].upper(),
                    }
                    for g in per_grade
                ]
            )
            parts.append(df_to_md(tbl) + "\n")
        parts.append(
            "- Traffic light uses the binomial default count under the forecast PD "
            "(EBA IRB Validation Handbook 2023): green < green-quantile, yellow up to "
            "yellow-quantile, red above.\n"
        )
    if "reliability_curve" in ctx.figures:
        parts.append(f"\n![reliability]({fig_rel(ctx.figures['reliability_curve'])})\n")
    return "".join(parts)
