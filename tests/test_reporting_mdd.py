"""MDD assembly (§6): 16 per-chapter files + index, deterministic regeneration (DoD §9.10)."""

from __future__ import annotations

import hashlib

from creditscorecard.reporting import regenerate_mdd
from creditscorecard.reporting.mdd_sections import CHAPTERS


def test_sixteen_chapters_registered():
    assert len(CHAPTERS) == 16
    numbers = [c.NUMBER for c in CHAPTERS]
    assert numbers == list(range(1, 17))  # contiguous 1..16 in order


def test_pipeline_writes_all_chapter_files(config, pipeline_payload):
    mdd = config.reports_path() / "mdd"
    for chapter in CHAPTERS:
        assert (mdd / f"{chapter.SLUG}.md").exists()
        assert (mdd / f"{chapter.SLUG}.html").exists()
    assert (mdd / "index.html").exists()
    assert (mdd / "model_development_document.md").exists()


def test_regeneration_is_byte_identical(config, pipeline_payload):
    """Regenerating from artifacts (no retraining) must be deterministic (DoD §9.10)."""
    paths = regenerate_mdd(config)
    combined = config.reports_path() / "mdd" / "model_development_document.md"
    first = hashlib.sha256(combined.read_bytes()).hexdigest()
    regenerate_mdd(config)
    second = hashlib.sha256(combined.read_bytes()).hexdigest()
    assert first == second
    assert paths["md"].endswith("model_development_document.md")


def test_limitations_chapter_carries_kgb_entry(config, pipeline_payload):
    # Reject inference is disabled on the offline config → KGB limitation must appear.
    text = (config.reports_path() / "mdd" / "15_limitations.md").read_text(encoding="utf-8")
    assert "KGB" in text
