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

    gov = payload.get("governance", {})
    if gov:
        parts.append("## 0. Governance Metadata (SR 11-7 §III)\n")
        parts.append(
            f"- **Model ID / name:** `{gov.get('model_id')}` — {gov.get('model_name')}  ·  "
            f"**Tier:** {gov.get('model_tier')}\n"
            f"- **Purpose:** {gov.get('model_purpose')}\n"
            f"- **Intended use:** {gov.get('intended_use')}\n"
            f"- **Owner / developer / validator:** {gov.get('owner')} / "
            f"{gov.get('developer')} / {gov.get('validator')}\n"
            f"- **Approval / next review:** {gov.get('approval_date')} / "
            f"{gov.get('next_review_date')}\n"
            "- Full provenance (git SHA, artifact hashes, package versions, dataset hash) "
            "and the assumptions/limitations register are in `artifacts/model_card.json`.\n"
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

    design = payload.get("sample_design", {})
    if design:
        flag_desc = (
            "constructed from DPD/status"
            if design.get("constructed_default")
            else "pass-through of supplied binary target (flat sample)"
        )
        parts.append(
            "### Sample design (Basel / EBA §5.1)\n"
            f"- **Default flag:** {flag_desc}.\n"
            f"- **Cohort key:** `{design.get('cohort_col') or 'n/a'}`  ·  "
            f"raw **{design.get('n_raw')}** → post-exclusions "
            f"**{design.get('n_after_exclusions')}** "
            f"→ post-seasoning **{design.get('n_after_seasoning')}** "
            f"(seasoning dropped {design.get('seasoning_dropped', 0)}).\n"
        )
        excl = design.get("exclusion_counts") or {}
        if excl:
            parts.append(
                "- **Exclusions:** " + ", ".join(f"{name} ({n})" for name, n in excl.items()) + "\n"
            )
        for note in design.get("notes", []):
            parts.append(f"  - _{note}_\n")

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

    parts.append(_discrimination_md(payload.get("discrimination", {})))
    parts.append(_calibration_backtest_md(payload.get("calibration_backtest", {})))
    parts.append(_benchmark_md(payload.get("benchmark", {})))
    parts.append(_explainability_md(payload.get("explainability", {})))
    parts.append(_reject_inference_md(payload.get("reject_inference", {})))
    parts.append(_fairness_md(payload.get("fairness", {})))

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


def _ci(metric: dict[str, Any]) -> str:
    return (
        f"{metric['point']:.4f} "
        f"[{metric['lower']:.4f}, {metric['upper']:.4f}] "
        f"({metric['method']} {int(metric['level'] * 100)}%)"
    )


def _discrimination_md(disc: dict[str, Any]) -> str:
    """§5.3 — discrimination with bootstrap CIs, partial AUC, Somers' D, .632+ optimism."""
    if not disc:
        return ""
    parts = ["## 8a. Discrimination with Uncertainty (§5.3)\n"]
    rows = []
    for split, m in disc.get("per_split", {}).items():
        row = {
            "split": split,
            "n": m.get("n"),
            "AUC [CI]": _ci(m["auc"]),
            "Gini [CI]": _ci(m["gini"]),
            "KS [CI]": _ci(m["ks"]),
        }
        if "partial_auc" in m:
            row["pAUC"] = f"{m['partial_auc']:.4f}"
        if "somers_d" in m:
            row["Somers' D"] = f"{m['somers_d']:.4f}"
        rows.append(row)
    if rows:
        parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
    opt = disc.get("optimism", {})
    if opt:
        parts.append(
            "### In-sample optimism (.632+ bootstrap, Efron & Gong 1983)\n"
            f"- Apparent AUC **{opt.get('apparent_auc', float('nan')):.4f}** → "
            f"optimism-corrected **{opt.get('corrected_auc', float('nan')):.4f}** "
            f"(optimism {opt.get('optimism', float('nan')):.4f}; "
            f"OOB AUC {opt.get('oob_auc', float('nan')):.4f}).\n"
        )
    return "".join(parts)


def _calibration_backtest_md(cb: dict[str, Any]) -> str:
    """§5.5 — Brier/ECE/HL + per-grade Jeffreys traffic-light table + grade HHI."""
    if not cb:
        return ""
    parts = ["## 8b. Calibration Backtest (§5.5)\n"]
    parts.append(
        f"- **Brier:** {cb.get('brier', float('nan')):.4f}  ·  "
        f"**ECE:** {cb.get('ece', float('nan')):.4f}  ·  "
        f"**Hosmer-Lemeshow:** stat={cb.get('hosmer_lemeshow_stat', float('nan')):.2f}, "
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
        "- Traffic light per grade uses the binomial default count under the forecast PD "
        "(EBA IRB Validation Handbook 2023): green < green-quantile, yellow up to "
        "yellow-quantile, red above.\n"
    )
    return "".join(parts)


def _benchmark_md(bm: dict[str, Any]) -> str:
    """§5.4 — champion vs challenger: Gini±CI, DeLong, interpretability parity, verdict."""
    if not bm or bm.get("enabled") is False:
        return "## 11. Champion vs Challenger (§5.4)\n\n*Benchmark disabled.*\n"
    parts = ["## 11. Champion vs Challenger (§5.4)\n"]
    rep = bm.get("reportable_gini_oot", {})
    chal = bm.get("challenger_gini_oot", {})
    dl = bm.get("delong", {})
    parts.append(
        f"- **Challenger:** {bm.get('challenger')}\n"
        f"- **OOT Gini** — reportable {_ci(rep) if rep else 'n/a'} · "
        f"challenger {_ci(chal) if chal else 'n/a'}\n"
    )
    if dl:
        parts.append(
            f"- **DeLong test** (AUC challenger vs reportable): z={dl.get('z', float('nan')):.3f}, "
            f"p={dl.get('pvalue', float('nan')):.4f}\n"
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


def _explainability_md(ex: dict[str, Any]) -> str:
    """§5.8 — global importance (reportable + challenger) and interpretability parity."""
    if not ex:
        return ""
    parts = ["## 13. Explainability (§5.8)\n"]
    parts.append(f"- **Method:** {ex.get('method')}\n")
    rep = ex.get("reportable_importance", {})
    if rep:
        top = sorted(rep.items(), key=lambda kv: kv[1], reverse=True)[:10]
        tbl = pd.DataFrame([{"feature": f, "mean_abs_shap": round(v, 4)} for f, v in top])
        parts.append("### Reportable model — global importance (mean |linear SHAP|)\n")
        parts.append(df_to_md(tbl) + "\n")
    parts.append("- SHAP beeswarm / importance figure: `../figures/shap_summary.png`.\n")
    return "".join(parts)


def _reject_inference_md(ri: dict[str, Any]) -> str:
    """§5.2 — reject inference method, sensitivity, or KGB limitation if disabled."""
    parts = ["## 3. Reject Inference (§5.2)\n"]
    if not ri or ri.get("enabled") is False or "methods" not in ri:
        parts.append(
            "*Reject inference is **disabled** (no reject data). The development sample is "
            "Known-Good-Bad (KGB) only; through-the-door population selection bias is "
            "uncorrected — see the limitations register in the model card.*\n"
        )
        return "".join(parts)
    rows = []
    for name, m in ri.get("methods", {}).items():
        rows.append(
            {
                "method": name,
                "coef_shift_L2": round(m.get("coef_shift_l2", float("nan")), 4),
                "gini_kgb": round(m.get("gini_kgb", float("nan")), 4),
                "gini_method": round(m.get("gini_method", float("nan")), 4),
                "gini_shift": round(m.get("gini_shift", float("nan")), 4),
            }
        )
    parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
    parts.append(
        "- Sensitivity is reported across methods (Banasik & Crook; Bücker et al.): the "
        "simpler methods often match complex ones — no single method is presented as "
        "definitive.\n"
    )
    return "".join(parts)


def _fairness_md(fair: dict[str, Any]) -> str:
    """§5.6 — AIR / SMD / SPD / EOD per protected attribute + proxy scan + verdict."""
    parts = ["## 12. Fairness — Disparate Impact (§5.6)\n"]
    if not fair or not fair.get("enabled") or not fair.get("attributes"):
        parts.append(
            "*Fairness N/A: no protected attributes configured/present. ECOA/Reg B "
            "disparate-impact testing should be run whenever a protected attribute is "
            "available (see limitations register).*\n"
        )
        return "".join(parts)
    rows = []
    for a in fair["attributes"]:
        rows.append(
            {
                "attribute": a["attribute"],
                "AIR": round(a["adverse_impact_ratio"], 3),
                "SMD": round(a["standardized_mean_difference"], 3),
                "SPD": round(a["statistical_parity_difference"], 3),
                "EOD": round(a["equal_opportunity_difference"], 3),
                "status": a["air_status"],
            }
        )
    parts.append(df_to_md(pd.DataFrame(rows)) + "\n")
    parts.append(
        "- AIR < 0.80 breaches the **80% rule** (ECOA/Reg B). "
        f"{'Failure acknowledged in config.' if fair.get('acknowledged_failure') else ''}\n"
    )
    for attr, scan in (fair.get("proxies") or {}).items():
        flagged = [s["feature"] for s in scan if s["flagged"]]
        if flagged:
            parts.append(f"- **Proxy scan ({attr})**: flagged {flagged}\n")
    if fair.get("note"):
        parts.append(f"- {fair['note']}\n")
    return "".join(parts)


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
