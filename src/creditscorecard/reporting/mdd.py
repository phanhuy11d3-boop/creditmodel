"""MDD assembly (§6): 16 per-chapter files + an index, plus a combined document.

Deterministic (DoD §9.10): every chapter reads only serialized artifacts via
:class:`MddContext`; there is no wall-clock time in the output (the provenance line uses the
payload's frozen ``created_at``/``version``). Regenerating from the same artifacts therefore
produces byte-identical files — proven by ``test_regression_baseline`` /
``test_reporting_deterministic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import markdown as md_lib
import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger
from creditscorecard.reporting._helpers import HTML_TEMPLATE
from creditscorecard.reporting.context import MddContext, load_context
from creditscorecard.reporting.mdd_sections import CHAPTERS

logger = get_logger(__name__)

_TITLE = "Model Development Document — PD Application Scorecard"


def _write_html(md: str, path: Path, title: str) -> None:
    body = md_lib.markdown(md, extensions=["tables", "fenced_code", "toc"])
    path.write_text(HTML_TEMPLATE.format(title=title, body=body), encoding="utf-8")


def _render_all(ctx: MddContext) -> list[tuple[Any, str]]:
    return [(chapter, chapter.render(ctx)) for chapter in CHAPTERS]


def generate_mdd_from_context(ctx: MddContext, config: Config) -> dict[str, str]:
    """Write per-chapter files + index + a combined document from an in-memory context."""
    mdd_dir = config.reports_path() / "mdd"
    mdd_dir.mkdir(parents=True, exist_ok=True)
    rendered = _render_all(ctx)

    # Per-chapter markdown + HTML.
    chapter_files: dict[str, str] = {}
    for chapter, md in rendered:
        stem = chapter.SLUG
        (mdd_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        _write_html(md, mdd_dir / f"{stem}.html", f"{chapter.NUMBER}. {chapter.TITLE}")
        chapter_files[stem] = str(mdd_dir / f"{stem}.md")

    # Index linking every chapter.
    index_lines = [f"# {_TITLE} — Index\n"]
    from creditscorecard.reporting.context import generated_stamp

    index_lines.append(generated_stamp(ctx.payload) + "\n")
    for chapter, _ in rendered:
        index_lines.append(f"- [{chapter.NUMBER}. {chapter.TITLE}]({chapter.SLUG}.html)\n")
    index_md = "\n".join(index_lines)
    (mdd_dir / "index.md").write_text(index_md, encoding="utf-8")
    _write_html(index_md, mdd_dir / "index.html", f"{_TITLE} — Index")

    # Combined document (backward-compatible filename).
    combined = f"# {_TITLE}\n\n" + "\n\n".join(md for _, md in rendered)
    md_path = mdd_dir / "model_development_document.md"
    md_path.write_text(combined, encoding="utf-8")
    html_path = mdd_dir / "model_development_document.html"
    _write_html(combined, html_path, _TITLE)

    logger.info("Generated MDD: %d chapters + index + combined document.", len(rendered))
    return {
        "md": str(md_path),
        "html": str(html_path),
        "index": str(mdd_dir / "index.html"),
        "chapters": chapter_files,  # type: ignore[dict-item]
    }


def generate_mdd(
    payload: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, str],
    config: Config,
    model_card: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Assemble the MDD from an in-pipeline payload/tables/figures (+ optional model card)."""
    ctx = MddContext(payload=payload, tables=tables, figures=figures, model_card=model_card or {})
    return generate_mdd_from_context(ctx, config)


def regenerate_mdd(config: Config) -> dict[str, str]:
    """Rebuild the MDD purely from on-disk artifacts (no retraining) — powers `scorecard report`."""
    ctx = load_context(config)
    return generate_mdd_from_context(ctx, config)
