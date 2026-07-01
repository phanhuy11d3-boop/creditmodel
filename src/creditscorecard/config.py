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


# --------------------------------------------------------------------------- #
# Refactor §4 — expanded config blocks (Basel/EBA/ECB/SR 11-7 aligned)
# --------------------------------------------------------------------------- #


class DefaultDefinition(BaseModel):
    """Basel default definition (90 DPD) with cure / re-default treatment."""

    dpd_threshold: int = Field(90, ge=1)  # Basel default = 90 days past due
    cure_period_months: int = Field(3, ge=0)
    re_default_treatment: Literal["separate", "merge"] = "separate"


class ExclusionRule(BaseModel):
    name: str
    rule: str  # pandas query() expression evaluated against the raw frame


class SampleDesignConfig(BaseModel):
    """Sample design & performance window (§5.1). Backward-compatible: when the
    frame lacks DPD/status/origination columns the pipeline falls back to the
    legacy temporal/stratified split (diagnostic risk R1)."""

    observation_window_months: int = Field(12, ge=1)
    performance_window_months: int = Field(12, ge=1)
    default_definition: DefaultDefinition = Field(default_factory=DefaultDefinition)
    exclusions: list[ExclusionRule] = Field(default_factory=list)
    cohort_key: str = "origination_month"
    minimum_seasoning_months: int = Field(6, ge=0)
    # Column names that drive default-flag construction. When none of these columns
    # exist in the frame, the module runs in pass-through mode (keeps the supplied
    # binary target) and records the limitation — see §5.1 / diagnostic risk R1.
    dpd_column: str | None = None  # per-account days-past-due (flat frame)
    status_column: str | None = None  # loan-status column (flat frame)
    default_statuses: list[str] = Field(default_factory=list)  # statuses counted as default
    origination_date_column: str | None = None  # source of the vintage; falls back to date_column
    reference_date: str | None = None  # as-of date for seasoning; else max origination date


class RejectInferenceConfig(BaseModel):
    """Reject inference (§5.2). Off by default; requires reject data."""

    enabled: bool = False
    reject_data_path: str | None = None
    method: Literal["parceling", "reweighting", "fuzzy_augmentation", "none"] = "parceling"
    bad_rate_multiplier: float = Field(3.0, gt=0)  # rejects ~2-5x accepts bad rate
    parcels: int = Field(10, ge=2)

    @model_validator(mode="after")
    def _enabled_requires_path(self) -> RejectInferenceConfig:
        if self.enabled and not self.reject_data_path:
            raise ValueError("reject_inference.reject_data_path is required when enabled")
        return self


class DiscriminationConfig(BaseModel):
    """Discrimination with uncertainty (§5.3)."""

    bootstrap_iterations: int = Field(1000, ge=1)
    bootstrap_method: Literal["bca", "percentile", "basic"] = "bca"
    confidence_level: float = Field(0.95, gt=0, lt=1)
    compute_partial_auc: bool = True
    partial_auc_range: tuple[float, float] = (0.0, 0.4)
    compute_somers_d: bool = True

    @model_validator(mode="after")
    def _range_ordered(self) -> DiscriminationConfig:
        lo, hi = self.partial_auc_range
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("discrimination.partial_auc_range must satisfy 0 <= lo < hi <= 1")
        return self


class BenchmarkConfig(BaseModel):
    """Champion vs challenger (§5.4)."""

    enabled: bool = True
    challenger: Literal["gradient_boosting", "random_forest", "xgboost"] = "gradient_boosting"
    challenger_params: dict[str, Any] = Field(
        default_factory=lambda: {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    )
    delong_test: bool = True
    interpretability_parity: bool = True
    # Verdict thresholds (§5.4): challenger "materially beats" the reportable model
    # when the OOT Gini gap exceeds this AND the DeLong p-value is below its threshold.
    gini_gap_threshold: float = Field(0.03, gt=0)
    delong_p_threshold: float = Field(0.05, gt=0, lt=1)


class TrafficLightConfig(BaseModel):
    green_upper_quantile: float = Field(0.95, gt=0, lt=1)
    yellow_upper_quantile: float = Field(0.99, gt=0, lt=1)

    @model_validator(mode="after")
    def _order(self) -> TrafficLightConfig:
        if self.yellow_upper_quantile <= self.green_upper_quantile:
            raise ValueError("yellow_upper_quantile must be > green_upper_quantile")
        return self


class PerGradeBacktestConfig(BaseModel):
    method: Literal["jeffreys", "binomial", "correlated_binomial"] = "jeffreys"
    alpha: float = Field(0.05, gt=0, lt=1)
    traffic_light: TrafficLightConfig = Field(default_factory=TrafficLightConfig)


class CalibrationExtendedConfig(BaseModel):
    """Calibration & backtesting (§5.5) — distinct from the anchor calibration."""

    hosmer_lemeshow_groups: int = Field(10, ge=2)
    compute_brier: bool = True
    compute_ece: bool = True
    reliability_curve_bins: int = Field(10, ge=2)
    per_grade_backtest: PerGradeBacktestConfig = Field(default_factory=PerGradeBacktestConfig)


class FairnessConfig(BaseModel):
    """Fairness / ECOA / Reg B disparate-impact testing (§5.6)."""

    enabled: bool = True
    protected_attributes: list[str] = Field(default_factory=list)
    favourable_outcome: str = "approved"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "adverse_impact_ratio",
            "standardized_mean_difference",
            "statistical_parity_difference",
            "equal_opportunity_difference",
        ]
    )
    air_threshold_warn: float = Field(0.90, gt=0, le=1)
    air_threshold_alert: float = Field(0.80, gt=0, le=1)  # 80% rule
    proxy_scan: bool = True
    # §5.6: AIR < alert is BUILD-FAILING unless explicitly acknowledged.
    acknowledge_failure: bool = False

    @model_validator(mode="after")
    def _thresholds_order(self) -> FairnessConfig:
        if self.air_threshold_alert >= self.air_threshold_warn:
            raise ValueError("fairness.air_threshold_alert must be < air_threshold_warn")
        return self


class ExplainabilityConfig(BaseModel):
    """SHAP / reason-code explainability (§5.8)."""

    shap_enabled: bool = True
    shap_sample_size: int = Field(1000, ge=1)
    reason_codes_top_n: int = Field(4, ge=1)  # ECOA adverse-action typically 4 reasons
    global_importance_method: Literal["mean_abs_shap"] = "mean_abs_shap"
    interpretability_parity_top_k: int = Field(10, ge=1)


class MonitoringExtendedConfig(BaseModel):
    """Multi-period monitoring run-log (§5.7)."""

    runlog_backend: Literal["sqlite", "jsonl"] = "sqlite"
    runlog_path: str = "monitoring/runlog.db"
    cadence: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"
    ave_backtest_enabled: bool = True
    vintage_analysis_enabled: bool = True
    psi_history_min_periods: int = Field(3, ge=1)


class GovernanceConfig(BaseModel):
    """SR 11-7 §III model-inventory fields (§5.9)."""

    model_config = {"protected_namespaces": ()}  # allow model_* field names

    model_id: str = "PD-APP-001"
    model_name: str = "Application PD Scorecard"
    model_tier: int = Field(2, ge=1, le=3)  # 1=highest risk, 3=lowest
    model_purpose: str = "Application-stage PD estimate for consumer lending decisioning."
    owner: str | None = None
    developer: str | None = None
    validator: str | None = None
    approval_date: str | None = None
    next_review_date: str | None = None
    intended_use: str = "Application-stage approve/decline and interest-rate tier assignment."
    known_limitations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


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
    # --- refactor §4 expanded blocks ---
    sample_design: SampleDesignConfig = Field(default_factory=SampleDesignConfig)
    reject_inference: RejectInferenceConfig = Field(default_factory=RejectInferenceConfig)
    discrimination: DiscriminationConfig = Field(default_factory=DiscriminationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    calibration_extended: CalibrationExtendedConfig = Field(
        default_factory=CalibrationExtendedConfig
    )
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    monitoring_extended: MonitoringExtendedConfig = Field(default_factory=MonitoringExtendedConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
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
