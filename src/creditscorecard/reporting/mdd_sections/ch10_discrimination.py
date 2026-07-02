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

    ss = p.get("split_stability", {})
    overlap = ss.get("gini_train_vs_oot", {})
    if overlap:
        a = overlap["train_gini_ci"]
        b = overlap["oot_gini_ci"]
        flag = "✅ within sampling noise" if overlap["ci_overlap"] else "⚠️ significant drop"
        parts.append(
            "### Train → OOT Gini stability (CI overlap)\n"
            f"- Train Gini CI [{a[0]:.4f}, {a[1]:.4f}] vs OOT CI [{b[0]:.4f}, {b[1]:.4f}] "
            f"(point drop {overlap['point_drop']:+.4f}) — **{flag}**.\n"
            f"- {overlap['verdict']}\n"
        )
        # Cross-read with PSI: disjoint CI + stable input ⇒ overfitting, not covariate drift.
        oot_psi = ss.get("psi", {}).get("oot", {})
        if not overlap["ci_overlap"] and oot_psi.get("status") == "OK":
            parts.append(
                f"- **Interpretation:** PSI(oot vs train)={oot_psi['psi']:.4f} [OK] → the input "
                "distribution is stable, so the discrimination drop reflects **overfitting to "
                "train / small-sample variance**, not covariate shift.\n"
            )
    parts.append("### Figures\n")
    for label in ("roc", "cap", "calibration", "score_distribution"):
        if label in ctx.figures:
            parts.append(f"**{label}**\n\n![{label}]({fig_rel(ctx.figures[label])})\n")
    return "".join(parts)
