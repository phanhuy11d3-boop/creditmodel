"""MDD generation (refactor §6): validator-ready Model Development Document.

One Python module per chapter under ``mdd_sections/``; the document is assembled
deterministically from serialized artifacts so regenerating without retraining is
byte-identical (DoD §9.10).
"""

from __future__ import annotations

from creditscorecard.reporting._helpers import df_to_md
from creditscorecard.reporting.mdd import (
    generate_mdd,
    generate_mdd_from_context,
    regenerate_mdd,
)

__all__ = ["generate_mdd", "generate_mdd_from_context", "regenerate_mdd", "df_to_md"]
