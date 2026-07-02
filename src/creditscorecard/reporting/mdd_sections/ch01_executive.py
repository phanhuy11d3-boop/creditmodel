"""Chapter 1 — Executive summary & governance metadata (§5.9, §6.1)."""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext, generated_stamp

NUMBER, TITLE, SLUG = 1, "Executive Summary & Governance Metadata", "01_executive"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    gov = p.get("governance", {})
    conv = p.get("convention", {})
    parts = [
        f"## {NUMBER}. {TITLE}\n",
        generated_stamp(p) + "\n",
    ]
    overall = p.get("validation_summary", {}).get("overall", {})
    if overall:
        verdict = overall.get("verdict", "?")
        badge = {"APPROVED": "✅", "CONDITIONAL": "⚠️", "NOT APPROVED": "⛔"}.get(verdict, "")
        failed = overall.get("failed_checks", [])
        parts.append(
            f"> **Validation verdict: {badge} {verdict}**"
            + (
                f" — failed checks: {', '.join(failed)}."
                if failed
                else " — all threshold checks pass."
            )
            + "\n"
        )
    if gov:
        parts.append(
            f"- **Model ID / name:** `{gov.get('model_id')}` — {gov.get('model_name')}  ·  "
            f"**Tier:** {gov.get('model_tier')}\n"
            f"- **Purpose:** {gov.get('model_purpose')}\n"
            f"- **Intended use:** {gov.get('intended_use')}\n"
            f"- **Owner / developer / validator:** {gov.get('owner')} / "
            f"{gov.get('developer')} / {gov.get('validator')}\n"
            f"- **Approval / next review:** {gov.get('approval_date')} / "
            f"{gov.get('next_review_date')}\n"
        )
    prov = ctx.model_card.get("provenance", {})
    if prov:
        parts.append(
            f"- **Provenance:** git `{str(prov.get('git_sha'))[:8]}`, "
            f"Python {prov.get('python_version')}, seed {prov.get('seed')}, "
            f"dataset hash `{ctx.model_card.get('dataset_hash')}`.\n"
        )
    parts.append(
        f"- **Orientation:** {conv.get('woe')}; expected coefficient sign "
        f"{conv.get('expected_coef_sign')} (more Good ⇒ lower P(Bad)).\n"
        "- **Points formula:** `points_i = -(WoE_i·β_i + α/n)·Factor + Offset/n`; "
        "higher score ⇔ lower PD.\n"
    )
    return "".join(parts)
