"""Chapter 7 — Model: coefficients, SEs, p-values, sign check, parity (§6.7)."""

from __future__ import annotations

import pandas as pd

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.context import MddContext

NUMBER, TITLE, SLUG = 7, "Model Summary", "07_model"


def render(ctx: MddContext) -> str:
    p = ctx.payload
    m = p["model"]
    coef_rows = [
        {
            "feature": f,
            "coefficient": m["coefficients"][f],
            "std_error": m["std_errors"][f],
            "p_value": m["p_values"][f],
            "sign_ok": m["coefficients"][f] <= 0,
        }
        for f in p.get("selected_features", [])
    ]
    coef_df = pd.DataFrame([{"feature": "intercept", "coefficient": m["intercept"]}, *coef_rows])
    parts = [f"## {NUMBER}. {TITLE}\n", df_to_md(coef_df) + "\n"]
    parts.append(
        f"- **Sign check passed:** {m['sign_ok']}  ·  "
        f"**Excluded wrong-sign:** {m['excluded_wrong_sign'] or 'none'}\n"
        f"- **statsmodels/sklearn parity:** passed={m['parity_passed']} "
        f"(max abs diff = {m['parity_max_abs_diff']:.2e})\n"
    )
    return "".join(parts)
