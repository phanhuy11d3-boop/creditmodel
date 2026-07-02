"""Shared MDD rendering helpers (tables, figure links, CI formatting, HTML shell)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
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


def fig_rel(path: str) -> str:
    """Figure link relative to the MDD directory (figures live in ../figures)."""
    return "../figures/" + Path(path).name


def ci(metric: dict[str, Any]) -> str:
    return (
        f"{metric['point']:.4f} "
        f"[{metric['lower']:.4f}, {metric['upper']:.4f}] "
        f"({metric['method']} {int(metric['level'] * 100)}%)"
    )
