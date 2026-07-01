"""Typed configuration loaded from YAML via pydantic-settings.

``base.yaml`` provides defaults; a named config (e.g. ``home_credit.yaml``)
is deep-merged on top. Invalid configuration fails fast with a clear message.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
BASE_CONFIG = CONFIGS_DIR / "base.yaml"


class DataConfig(BaseModel):
    adapter: Literal["csv", "synthetic"] = "synthetic"
    path: str | None = None
    target: str = "default"
    date_column: str | None = None

    @model_validator(mode="after")
    def _csv_requires_path(self) -> DataConfig:
        if self.adapter == "csv" and not self.path:
            raise ValueError("data.path is required when data.adapter == 'csv'")
        return self


class SplitConfig(BaseModel):
    test_size: float = Field(0.2, gt=0, lt=1)
    oot_size: float = Field(0.2, ge=0, lt=1)
    stratify: bool = True

    @model_validator(mode="after")
    def _sizes_leave_train(self) -> SplitConfig:
        if self.test_size + self.oot_size >= 1.0:
            raise ValueError("split.test_size + split.oot_size must be < 1.0")
        return self


class BinningConfig(BaseModel):
    min_bin_pct: float = Field(0.05, gt=0, lt=0.5)
    monotonic_trend: str = "auto"
    max_n_bins: int = Field(8, ge=2, le=20)


class SelectionConfig(BaseModel):
    iv_min: float = Field(0.02, ge=0)
    iv_suspicious: float = Field(0.5, gt=0)
    vif_threshold: float = Field(5.0, gt=1)
    forward_metric: Literal["gini", "auc"] = "gini"
    cv_folds: int = Field(5, ge=2, le=20)

    @model_validator(mode="after")
    def _iv_order(self) -> SelectionConfig:
        if self.iv_suspicious <= self.iv_min:
            raise ValueError("selection.iv_suspicious must be > selection.iv_min")
        return self


class ModelConfig(BaseModel):
    engine: Literal["statsmodels", "sklearn"] = "statsmodels"
    enforce_sign_check: bool = True
    sign_overrides: list[str] = Field(default_factory=list)
    parity_tol: float = Field(1e-4, gt=0)


class CalibrationConfig(BaseModel):
    anchor_default_rate: float | None = Field(None)

    @field_validator("anchor_default_rate")
    @classmethod
    def _rate_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("calibration.anchor_default_rate must be in (0, 1)")
        return v


class ScalingConfig(BaseModel):
    pdo: float = Field(20, gt=0)
    target_score: float = 600
    target_odds: float = Field(50, gt=0)
    rating_grades: int = Field(7, ge=2, le=30)
    round_points: bool = True


class MonitoringConfig(BaseModel):
    psi_bins: int = Field(10, ge=2, le=50)
    psi_warn: float = Field(0.10, gt=0)
    psi_alert: float = Field(0.25, gt=0)

    @model_validator(mode="after")
    def _thresholds_order(self) -> MonitoringConfig:
        if self.psi_alert <= self.psi_warn:
            raise ValueError("monitoring.psi_alert must be > monitoring.psi_warn")
        return self


class ValidationConfig(BaseModel):
    """Thresholds for the discrimination / stability / calibration checks."""

    gini_min: float = Field(0.40, ge=0, lt=1)
    ks_min: float = Field(0.35, ge=0, lt=1)
    hhi_max: float = Field(0.15, gt=0, le=1)
    mape_max: float = Field(0.10, gt=0)
    anchor_gap_max: float = Field(0.10, gt=0)
    curve_shape_n_se: float = Field(2.0, gt=0)


class TrackingConfig(BaseModel):
    mlflow_enabled: bool = False
    mlflow_uri: str | None = None


class PathsConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    data_dir: str = "data"


class Config(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    binning: BinningConfig = Field(default_factory=BinningConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # Resolved at load time to the repository root so relative paths work anywhere.
    project_root: str = Field(default_factory=lambda: str(Path.cwd()))

    def artifacts_path(self) -> Path:
        return self._resolve(self.paths.artifacts_dir)

    def reports_path(self) -> Path:
        return self._resolve(self.paths.reports_dir)

    def data_path(self) -> Path:
        return self._resolve(self.paths.data_dir)

    def _resolve(self, sub: str) -> Path:
        p = Path(sub)
        return p if p.is_absolute() else Path(self.project_root) / p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


def load_config(config_path: str | Path | None = None) -> Config:
    """Load ``base.yaml`` and deep-merge the named config on top of it."""
    merged = _read_yaml(BASE_CONFIG)
    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute() and not override_path.exists():
            candidate = CONFIGS_DIR / override_path.name
            if candidate.exists():
                override_path = candidate
        merged = _deep_merge(merged, _read_yaml(override_path))
    merged.setdefault("project_root", str(Path(__file__).resolve().parents[2]))
    try:
        return Config.model_validate(merged)
    except ValidationError as exc:  # fail fast with a readable message
        raise SystemExit(f"Invalid configuration:\n{exc}") from exc
