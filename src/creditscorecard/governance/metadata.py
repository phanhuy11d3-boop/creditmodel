"""Governance metadata & model card (refactor §5.9).

SR 11-7 §III expects a model inventory record: identity (id/name/tier/owner),
intended use, assumptions log, and a limitations register — the parts an examiner
looks for first. This module augments the static ``governance`` config with
build-time provenance (git SHA, package/interpreter versions, dataset hash,
artifact SHA-256 hashes, run timestamp, seed) and auto-detects model-level
assumptions/limitations (e.g. a KGB-only sample when reject inference is off).

Determinism (diagnostic risk R6): every provenance field is either constant for a
given commit+dataset (git SHA, dataset hash, artifact hashes, seed) or an explicit
timestamp. The dataset hash uses ``pandas.util.hash_pandas_object`` so it is stable
across runs on identical data; only ``run_timestamp`` varies run-to-run.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

MODEL_CARD_FILE = "model_card.json"

# Packages whose versions materially affect model outputs (pinned in the card).
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "optbinning",
    "pydantic",
)

# Auto-detected assumption/limitation text (kept as constants so tests can assert them).
KGB_LIMITATION = (
    "KGB-only sample: reject inference disabled; the model is trained on approved "
    "applicants only and population (through-the-door) selection bias is uncorrected."
)


def git_sha(project_root: str | Path) -> str:
    """Current commit SHA, or 'unknown' outside a git work tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        pass
    return "unknown"


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover - env dependent
            out[pkg] = "unknown"
    return out


def dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash of a dataframe (columns + row values)."""
    h = hashlib.sha256()
    h.update("|".join(str(c) for c in df.columns).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h.update(row_hashes.tobytes())
    return h.hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes (first 16 hex chars), or 'missing'."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_artifacts(artifacts_dir: str | Path) -> dict[str, str]:
    """Hash every serialized artifact under ``artifacts_dir`` (excluding the card
    itself, so the card never needs to hash a file that contains its own hash)."""
    root = Path(artifacts_dir)
    if not root.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MODEL_CARD_FILE:
            out[p.relative_to(root).as_posix()] = hash_file(p)
    return out


def collect_assumptions(config: Config) -> tuple[list[str], list[str]]:
    """Return ``(assumptions, limitations)`` — config-declared plus auto-detected.

    Auto-detections encode model-level facts a validator must see (§5.9): the KGB
    limitation when reject inference is disabled, and the missing-fairness note when
    no protected attribute is configured.
    """
    assumptions = list(config.governance.assumptions)
    limitations = list(config.governance.known_limitations)

    if not config.reject_inference.enabled and KGB_LIMITATION not in limitations:
        limitations.append(KGB_LIMITATION)

    if config.fairness.enabled and not config.fairness.protected_attributes:
        note = (
            "Fairness metrics not computed: no protected attributes configured "
            "(fairness.protected_attributes is empty)."
        )
        if note not in limitations:
            limitations.append(note)

    return assumptions, limitations


@dataclass
class ModelCard:
    """Serializable governance record attached to the model (§5.9)."""

    governance: dict[str, Any]
    provenance: dict[str, Any]
    dataset_hash: str
    artifact_hashes: dict[str, str]
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_card(
    config: Config,
    dataset: pd.DataFrame,
    artifacts_dir: str | Path | None = None,
) -> ModelCard:
    """Assemble the model card from config + build-time provenance.

    ``artifacts_dir`` is hashed if provided (call after artifacts are written so the
    hashes cover the persisted model); otherwise ``artifact_hashes`` is left empty
    and can be filled by a later :func:`hash_artifacts` pass.
    """
    root = config.project_root
    provenance: dict[str, Any] = {
        "git_sha": git_sha(root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "run_timestamp": datetime.now(UTC).isoformat(),
        "seed": config.seed,
    }
    assumptions, limitations = collect_assumptions(config)
    artifact_hashes = hash_artifacts(artifacts_dir) if artifacts_dir is not None else {}

    card = ModelCard(
        governance=config.governance.model_dump(),
        provenance=provenance,
        dataset_hash=dataset_hash(dataset),
        artifact_hashes=artifact_hashes,
        assumptions=assumptions,
        limitations=limitations,
    )
    logger.info(
        "Built model card: id=%s tier=%d git=%s dataset=%s (%d limitations).",
        card.governance.get("model_id"),
        card.governance.get("model_tier"),
        provenance["git_sha"][:8],
        card.dataset_hash,
        len(limitations),
    )
    return card


def save_model_card(card: ModelCard, config: Config) -> Path:
    """Persist the model card to ``artifacts/model_card.json``."""
    import json

    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / MODEL_CARD_FILE
    with path.open("w", encoding="utf-8") as fh:
        json.dump(card.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Saved model card to %s", path)
    return path
