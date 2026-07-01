"""Auto-generate the Model Development Document (MDD) in Markdown + HTML.

Reads only the serialized payload and the auxiliary tables/figures so the MDD is
a faithful, reproducible description of the persisted model (validator-ready).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import markdown as md_lib
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Model Development Document</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 960px; color: #1a1a1a; line-height: 1.5; }}
 h1, h2, h3 {{ color: #0b3d91; }}
 table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 th {{ background: #eef2fb; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border: 1px solid #eee; }}
 code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>
{body}
</body></html>
"""


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def generate_mdd(
    payload: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, str],
    config: Config,
) -> dict[str, str]:
    md = _build_markdown(payload, tables, figures)
    mdd_dir = config.reports_path() / "mdd"
    mdd_dir.mkdir(parents=True, exist_ok=True)

    md_path = mdd_dir / "model_development_document.md"
    md_path.write_text(md, encoding="utf-8")

    html_body = md_lib.markdown(md, extensions=["tables", "fenced_code", "toc"])
    html_path = mdd_dir / "model_development_document.html"
    html_path.write_text(_HTML_TEMPLATE.format(body=html_body), encoding="utf-8")

    logger.info("Generated MDD: %s / %s", md_path.name, html_path.name)
    return {"md": str(md_path), "html": str(html_path)}


def _fig_rel(path: str) -> str:
    return "../figures/" + Path(path).name


def _build_markdown(payload, tables, figures) -> str:  # noqa: PLR0915
    conv = payload["convention"]
    m = payload["model"]
    cal = payload["calibration"]
    sc = payload["scaling"]
    sample = payload.get("sample", {})
    sel = payload["selection_trail"]
    parts: list[str] = []

    parts.append("# Model Development Document — PD Application Scorecard\n")
    parts.append(
        f"*Version:* `{payload.get('version', 'n/a')}`  ·  "
        f"*Generated:* {datetime.now(UTC):%Y-%m-%d %H:%M UTC}  ·  "
        f"*Package:* {payload.get('package_version', '')}\n"
    )

    parts.append("## 1. Target, Event Definition & Orientation Convention\n")
    parts.append(
        f"- **Target column:** `{payload['target']}` — event `{conv['event']}`.\n"
        f"- **Positive class:** {conv['positive_class']}.\n"
        f"- **WoE orientation:** {conv['woe']}.\n"
        f"- **Expected coefficient sign:** {conv['expected_coef_sign']} "
        "(more Good ⇒ lower P(Bad)).\n"
        "- **Points formula:** `points_i = -(WoE_i·β_i + α/n)·Factor + Offset/n`; "
        "`TotalScore = Σ points_i`. Higher score ⇔ lower PD.\n"
    )

    parts.append("## 2. Data Source & Sample Design\n")
    if sample:
        parts.append(
            f"- Splits — train: **{sample.get('train')}**, test: **{sample.get('test')}**, "
            f"OOT: **{sample.get('oot')}**.\n"
            f"- Bad rates — train: {sample.get('train_bad_rate', float('nan')):.4f}, "
            f"test: {sample.get('test_bad_rate', float('nan')):.4f}, "
            f"OOT: {sample.get('oot_bad_rate', float('nan')):.4f}.\n"
        )
    temporal = payload.get("temporal_validation", False)
    if temporal and sample.get("date_ranges"):
        dr = sample["date_ranges"]
        parts.append(
            f"- **Temporal split** on `{payload['date_column']}`: "
            f"dev [{dr.get('dev_min')} .. {dr.get('dev_max')}], "
            f"OOT [{dr.get('oot_min')} .. {dr.get('oot_max')}].\n"
        )
    else:
        parts.append(
            "- **Temporal validation unavailable**; OOT is a stratified random hold-out.\n"
        )

    parts.append("## 3. Binning, WoE & IV per Characteristic\n")
    if "iv" in tables:
        parts.append("### Information Value summary\n")
        parts.append(df_to_md(tables["iv"]) + "\n")
    for feat in payload["selected_features"]:
        key = f"woe_{feat}"
        if key in tables:
            parts.append(f"### {feat}  (IV = {payload['iv'].get(feat, float('nan')):.4f})\n")
            parts.append(df_to_md(tables[key]) + "\n")

    parts.append("## 4. Feature Selection Trail\n")
    parts.append(f"- **Dropped (IV < min):** {sel['dropped_low_iv'] or 'none'}\n")
    parts.append(
        f"- **Flagged for leakage review (IV > suspicious, kept):** "
        f"{sel['suspicious_iv'] or 'none'}\n"
    )
    vif_dropped = sel["vif_dropped"]
    parts.append(
        "- **VIF drops:** "
        + (", ".join(f"{f} (VIF={v:.2f})" for f, v in vif_dropped) if vif_dropped else "none")
        + "\n"
    )
    fwd = pd.DataFrame(sel["forward_trail"], columns=["feature_added", "cv_gini"])
    parts.append("### Forward-selection order (CV Gini)\n")
    parts.append(df_to_md(fwd) + "\n")

    parts.append("## 5. Model Summary\n")
    coef_rows = [
        {
            "feature": f,
            "coefficient": m["coefficients"][f],
            "std_error": m["std_errors"][f],
            "p_value": m["p_values"][f],
            "sign_ok": m["coefficients"][f] <= 0,
        }
        for f in payload["selected_features"]
    ]
    coef_df = pd.DataFrame([{"feature": "intercept", "coefficient": m["intercept"]}, *coef_rows])
    parts.append(df_to_md(coef_df) + "\n")
    parts.append(
        f"- **Sign check passed:** {m['sign_ok']}  ·  "
        f"**Excluded wrong-sign:** {m['excluded_wrong_sign'] or 'none'}\n"
        f"- **statsmodels/sklearn parity:** passed={m['parity_passed']} "
        f"(max abs diff = {m['parity_max_abs_diff']:.2e})\n"
    )

    parts.append("## 6. Calibration\n")
    parts.append(
        f"- Method: **{cal['method']}**, anchor default rate = {cal['anchor_rate']:.4f}.\n"
        f"- Mean PD before → after: {cal['mean_pd_before']:.4f} → {cal['mean_pd_after']:.4f} "
        f"(intercept shift = {cal['intercept_shift']:.4f}).\n"
    )

    parts.append("## 7. Scaling Parameters & Master Scale\n")
    parts.append(
        f"- `Factor = PDO/ln(2)` = **{sc['factor']:.4f}** (PDO={sc['pdo']}).\n"
        f"- `Offset = TargetScore − Factor·ln(TargetOdds)` = **{sc['offset']:.4f}** "
        f"(TargetScore={sc['target_score']}, TargetOdds={sc['target_odds']} Good:Bad).\n"
    )
    if "master_scale" in tables:
        parts.append(df_to_md(tables["master_scale"]) + "\n")

    parts.append("## 8. Performance (train / test / OOT)\n")
    if "metrics" in tables:
        parts.append(df_to_md(tables["metrics"]) + "\n")
    parts.append("### Figures\n")
    for label, fig in figures.items():
        parts.append(f"**{label}**\n\n![{label}]({_fig_rel(fig)})\n")

    parts.append("## 9. Validation Summary\n")
    parts.append(_validation_summary_md(payload.get("validation_summary", {})))

    parts.append("## 10. Monitoring — Frozen Reference & Escalation\n")
    thr = payload["monitoring_thresholds"]
    ref = payload["stability_reference"]
    parts.append(
        f"- PSI/CSI use **frozen** development reference bins "
        f"({len(ref['score_edges']) + 1} score bins).\n"
        f"- Thresholds: PSI < {thr['psi_warn']} OK · "
        f"{thr['psi_warn']}–{thr['psi_alert']} **WARN** · "
        f"> {thr['psi_alert']} **ALERT**.\n"
        "- New data is scored with these frozen edges (no re-binning).\n"
        "- Rating-grade HHI is checked against `validation.hhi_max` on every monitoring run.\n"
    )

    return "\n".join(parts)


def _validation_summary_md(summary: dict[str, Any]) -> str:
    if not summary:
        return "*No validation summary available.*\n"
    parts: list[str] = []

    def row(m: dict[str, Any]) -> str:
        return (
            f"- **{m['metric']}**: {m['value']:.4f} "
            f"(threshold {m['threshold']:.4f}) → **{m['status']}**\n"
        )

    parts.append("### Discriminatory power\n")
    for m in summary.get("discriminatory_power", {}).values():
        parts.append(row(m))

    parts.append("\n### Stability & concentration\n")
    for m in summary.get("stability_concentration", {}).values():
        parts.append(row(m))

    parts.append("\n### Calibration & accuracy\n")
    cal = summary.get("calibration_accuracy", {})
    for key in ("mape", "anchor_gap"):
        if key in cal:
            parts.append(row(cal[key]))
    curve = cal.get("curve_shape")
    if curve:
        parts.append(
            f"- **curve_shape** (±{curve['n_se']} SE band, rank-monotonic): "
            f"monotonic={curve['monotonic']}, all bands within SE band="
            f"{curve['all_within_band']} → **{curve['status']}**\n"
        )
    return "".join(parts) + "\n"
