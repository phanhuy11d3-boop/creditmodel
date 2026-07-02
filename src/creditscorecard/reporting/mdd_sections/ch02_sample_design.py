"""Chapter 2 — Sample design: vintages, exclusions, default definition, window (§5.1, §6.2)."""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 2, "Sample Design", "02_sample_design"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    sample = p.get("sample", {})
    parts = [f"## {NUMBER}. {TITLE}\n"]
    if sample:
        parts.append(
            f"- Splits — train: **{sample.get('train')}**, test: **{sample.get('test')}**, "
            f"OOT: **{sample.get('oot')}**.\n"
            f"- Bad rates — train: {sample.get('train_bad_rate', float('nan')):.4f}, "
            f"test: {sample.get('test_bad_rate', float('nan')):.4f}, "
            f"OOT: {sample.get('oot_bad_rate', float('nan')):.4f}.\n"
        )
    if p.get("temporal_validation") and sample.get("date_ranges"):
        dr = sample["date_ranges"]
        parts.append(
            f"- **Temporal split** on `{p.get('date_column')}`: "
            f"dev [{dr.get('dev_min')} .. {dr.get('dev_max')}], "
            f"OOT [{dr.get('oot_min')} .. {dr.get('oot_max')}].\n"
        )
    else:
        parts.append(
            "- **Temporal validation unavailable**; OOT is a stratified random hold-out.\n"
        )

    design = p.get("sample_design", {})
    if design:
        flag_desc = (
            "constructed from DPD/status"
            if design.get("constructed_default")
            else "pass-through of supplied binary target (flat sample)"
        )
        parts.append(
            "### Definition of default & vintages (Basel / EBA §5.1)\n"
            f"- **Default flag:** {flag_desc}.\n"
            f"- **Cohort key:** `{design.get('cohort_col') or 'n/a'}`  ·  "
            f"raw **{design.get('n_raw')}** → post-exclusions "
            f"**{design.get('n_after_exclusions')}** → post-seasoning "
            f"**{design.get('n_after_seasoning')}** "
            f"(seasoning dropped {design.get('seasoning_dropped', 0)}).\n"
        )
        excl = design.get("exclusion_counts") or {}
        if excl:
            parts.append(
                "- **Exclusions:** " + ", ".join(f"{k} ({v})" for k, v in excl.items()) + "\n"
            )
        for note in design.get("notes", []):
            parts.append(f"  - _{note}_\n")
    return "".join(parts)
