"""Structured logging used across the pipeline. No bare prints in library code."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Idempotently configure root logging to stderr with a consistent format."""
    global _CONFIGURED  # noqa: PLW0603 - module-level singleton guard
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
