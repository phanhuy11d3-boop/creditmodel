"""Artifact registry: serialize/load everything that *is* the model.

The serialized ``model.json`` fully defines scoring, monitoring, and reporting.
Serving loads only these artifacts (no training libraries required). A content
hash over the model payload (excluding timestamps) provides a stable version id.
Optional MLflow logging is flag-gated and never required.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from creditscorecard.config import Config
from creditscorecard.logging import get_logger

logger = get_logger(__name__)

MODEL_FILE = "model.json"


def compute_version(payload: dict[str, Any]) -> str:
    """Deterministic content hash of the model payload (timestamps excluded)."""
    stable = {k: v for k, v in payload.items() if k not in {"created_at", "version"}}
    blob = json.dumps(stable, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_artifacts(
    payload: dict[str, Any],
    config: Config,
    tables: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Persist the model payload and any auxiliary tables (CSV) to artifacts_dir."""
    artifacts_dir = config.artifacts_path()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(payload)
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["version"] = compute_version(payload)

    model_path = artifacts_dir / MODEL_FILE
    with model_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)

    if tables:
        tables_dir = artifacts_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(tables_dir / f"{name}.csv", index=False)

    logger.info("Saved artifacts to %s (version=%s).", model_path, payload["version"])
    _maybe_log_mlflow(payload, config, artifacts_dir)
    return model_path


def load_payload(config: Config) -> dict[str, Any]:
    """Load the model payload dict from artifacts_dir."""
    model_path = config.artifacts_path() / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"No artifacts found at {model_path}. Run `scorecard run` first.")
    with model_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _maybe_log_mlflow(payload: dict[str, Any], config: Config, artifacts_dir: Path) -> None:
    if not config.tracking.mlflow_enabled:
        return
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_enabled but mlflow not installed; skipping tracking.")
        return
    if config.tracking.mlflow_uri:
        mlflow.set_tracking_uri(config.tracking.mlflow_uri)
    mlflow.set_experiment("credit-scorecard")
    with mlflow.start_run():
        mlflow.log_param("version", payload["version"])
        mlflow.log_param("n_features", len(payload["selected_features"]))
        for split, metrics in payload.get("performance", {}).items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{split}_{metric}", float(value))
        mlflow.log_artifact(str(artifacts_dir / MODEL_FILE))
    logger.info("Logged run to MLflow.")
