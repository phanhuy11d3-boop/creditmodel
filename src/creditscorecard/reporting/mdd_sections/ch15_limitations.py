"""Chapter 15 — Limitations & assumptions register (§5.9, §6.15).

Sourced from the model card (config-declared entries plus module auto-detections such as the
KGB limitation when reject inference is disabled).
"""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 15, "Limitations & Assumptions Register", "15_limitations"


def render(ctx: MddContext) -> str:
    card = ctx.model_card
    assumptions = card.get("assumptions", []) or ctx.payload.get("governance", {}).get(
        "assumptions", []
    )
    limitations = card.get("limitations", []) or ctx.payload.get("governance", {}).get(
        "known_limitations", []
    )
    parts = [f"## {NUMBER}. {TITLE}\n"]
    parts.append("### Assumptions\n")
    if assumptions:
        parts += [f"- {a}\n" for a in assumptions]
    else:
        parts.append("- _None recorded._\n")
    parts.append("\n### Limitations register\n")
    if limitations:
        parts += [f"- {lim}\n" for lim in limitations]
    else:
        parts.append("- _None recorded._\n")
    return "".join(parts)
