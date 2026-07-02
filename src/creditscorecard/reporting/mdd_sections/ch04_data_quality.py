"""Chapter 4 — Data quality & schema (pandera contract) (§6.4)."""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 4, "Data Quality & Schema", "04_data_quality"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    conv = p.get("convention", {})
    return (
        f"## {NUMBER}. {TITLE}\n"
        f"- **Target:** `{p.get('target')}` — {conv.get('event')}. Validated by the pandera "
        "data contract at ingestion (binary non-null target; non-empty frame). Contract "
        "violations fail the pipeline before any modelling.\n"
        "- Missing values are handled by an explicit Missing bin per characteristic; unseen "
        "categorical levels map to a reserved Other bin (see binning specs). No row-level "
        "imputation is performed prior to binning.\n"
    )
