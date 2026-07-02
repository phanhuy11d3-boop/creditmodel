"""Chapter 16 — Regulatory & academic references (§6.16)."""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 16, "References", "16_references"

_REFERENCES = [
    "Federal Reserve / OCC. **SR 11-7**: Guidance on Model Risk Management (2011).",
    "EBA. **Supervisory Handbook on the Validation of IRB Rating Systems** (2023).",
    "ECB. **Guide to Internal Models** (2024/2025 revisions).",
    "Basel Committee. **CRE36 / definition of default (90 days past due)**.",
    "Siddiqi, N. **Intelligent Credit Scoring**, 2nd ed. (2017).",
    "DeLong, DeLong & Clarke-Pearson (1988); Sun & Xu (2014) — DeLong test for correlated AUCs.",
    "Efron & Gong (1983); Efron & Tibshirani (1997) — .632+ bootstrap optimism correction.",
    "Banasik & Crook (2007); Bücker, van Kampen & Krämer (2013) — reject inference evidence.",
    "Lundberg & Lee (2017) — SHAP additive feature attributions.",
    "ECOA / Regulation B; CFPB adverse-action and disparate-impact guidance (80% rule).",
]


def render(ctx: MddContext) -> str:  # noqa: ARG001 - static chapter
    return f"## {NUMBER}. {TITLE}\n" + "".join(f"- {r}\n" for r in _REFERENCES)
