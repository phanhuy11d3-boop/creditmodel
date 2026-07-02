"""Chapter 11 — Champion vs challenger: DeLong, parity, verdict (§5.4, §6.11)."""

from __future__ import annotations

from creditscorecard.reporting._helpers import ci
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 11, "Champion vs Challenger", "11_benchmark"


def render(ctx: MddContext) -> str:
    bm = ctx.payload.get("benchmark", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if not bm or bm.get("enabled") is False:
        parts.append("*Benchmark disabled.*\n")
        return "".join(parts)
    rep = bm.get("reportable_gini_oot", {})
    chal = bm.get("challenger_gini_oot", {})
    dl = bm.get("delong", {})
    parts.append(
        f"- **Challenger:** {bm.get('challenger')}\n"
        f"- **OOT Gini** — reportable {ci(rep) if rep else 'n/a'} · "
        f"challenger {ci(chal) if chal else 'n/a'}\n"
    )
    if dl:
        parts.append(
            f"- **DeLong test** (AUC challenger vs reportable): "
            f"z={dl.get('z', float('nan')):.3f}, p={dl.get('pvalue', float('nan')):.4f}\n"
        )
    parity = bm.get("interpretability_parity", {})
    if parity:
        parts.append(
            f"- **Interpretability parity** (top-{parity.get('top_k')}): "
            f"Jaccard={parity.get('jaccard', float('nan')):.3f}; "
            f"reportable {parity.get('reportable_top')}, "
            f"challenger {parity.get('challenger_top')}\n"
        )
    flag = "⚠️ **UNDER-SPECIFIED**" if bm.get("under_specified") else "✅ adequate"
    parts.append(f"- **Verdict:** {flag} — {bm.get('verdict', '')}\n")
    return "".join(parts)
