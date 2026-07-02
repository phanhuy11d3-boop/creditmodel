"""Chapter 14 — Stability & monitoring plan: frozen refs, thresholds, run-log (§5.7, §6.14)."""

from __future__ import annotations

from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 14, "Stability & Monitoring Plan", "14_monitoring"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    thr = p.get("monitoring_thresholds", {})
    ref = p.get("stability_reference", {})
    n_bins = len(ref.get("score_edges", [])) + 1 if ref else "?"

    split_psi = p.get("split_stability", {}).get("psi", {})
    psi_block = ""
    if split_psi:
        rows = "".join(
            f"  - **{name} vs train:** PSI = {v['psi']:.4f} [{v['status']}]\n"
            for name, v in split_psi.items()
        )
        psi_block = (
            "- **Development-time score stability** (PSI of each split vs the frozen train "
            "reference):\n" + rows
        )

    return (
        f"## {NUMBER}. {TITLE}\n"
        f"- PSI/CSI use **frozen** development reference bins ({n_bins} score bins); new data "
        "is scored with these frozen edges (no re-binning).\n"
        f"- Thresholds: PSI < {thr.get('psi_warn')} OK · "
        f"{thr.get('psi_warn')}–{thr.get('psi_alert')} **WARN** · > {thr.get('psi_alert')} "
        "**ALERT**. Rating-grade HHI checked against `validation.hhi_max` each run.\n"
        "- **Multi-period run-log** (§5.7): `scorecard monitor --period-id …` appends per-metric "
        "rows; `scorecard monitor-report` computes PSI/CSI **trend** (rising drift flagged before "
        "any single breach), per-grade **AvE** Jeffreys backtest, and score **migration** matrix.\n"
        "- Escalation: any metric ALERT, or a rising trend once the minimum period count is met.\n"
        + psi_block
    )
