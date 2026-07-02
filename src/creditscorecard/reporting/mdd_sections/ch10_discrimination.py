"""Chapter 10 — Discrimination: AUC/Gini/KS with CIs, partial AUC, optimism (§5.3, §6.10)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import ci, df_to_md, fig_rel
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 10, "Discrimination Performance", "10_discrimination"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if "metrics" in ctx.tables:
        parts.append("### Point performance (train / test / OOT)\n")
        parts.append(df_to_md(ctx.tables["metrics"]) + "\n")

    disc = p.get("discrimination", {})
    if disc:
        rows = []
        for split, m in disc.get("per_split", {}).items():
            row = {
                "split": split,
                "n": m.get("n"),
                "AUC [CI]": ci(m["auc"]),
                "Gini [CI]": ci(m["gini"]),
                "KS [CI]": ci(m["ks"]),
            }
            if "partial_auc" in m:
                row["pAUC"] = f"{m['partial_auc']:.4f}"
            if "somers_d" in m:
                row["Somers' D"] = f"{m['somers_d']:.4f}"
            rows.append(row)
        if rows:
            parts.append("### With bootstrap confidence intervals (§5.3)\n")
            parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
        opt = disc.get("optimism", {})
        if opt:
            parts.append(
                "### In-sample optimism (.632+ bootstrap, Efron & Gong 1983)\n"
                f"- Apparent AUC **{opt.get('apparent_auc', float('nan')):.4f}** → "
                f"optimism-corrected **{opt.get('corrected_auc', float('nan')):.4f}** "
                f"(optimism {opt.get('optimism', float('nan')):.4f}; "
                f"OOB {opt.get('oob_auc', float('nan')):.4f}).\n"
            )
    parts.append("### Figures\n")
    for label in ("roc", "cap", "calibration", "score_distribution"):
        if label in ctx.figures:
            parts.append(f"**{label}**\n\n![{label}]({fig_rel(ctx.figures[label])})\n")
    return "".join(parts)
