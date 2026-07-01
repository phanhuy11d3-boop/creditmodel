"""End-to-end development pipeline (the `scorecard run` one command).

ingest -> validate -> split -> bin -> WoE/IV -> select -> VIF -> forward-select
-> train -> sign-check -> calibrate -> scale -> Master Scale -> evaluate
(train/test/OOT) -> freeze PSI/CSI reference -> persist artifacts -> MDD + figures.

Leakage control: binning, WoE/IV, selection, VIF, calibration, scaling and the
frozen references are all fit on **train** only; test and OOT receive transforms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from creditscorecard import __version__
from creditscorecard.config import Config
from creditscorecard.data.adapters import load_dataset
from creditscorecard.data.definition_of_default import save_sample_design
from creditscorecard.data.schema import validate_dataframe
from creditscorecard.data.split import SplitData, split_data
from creditscorecard.evaluation import curves
from creditscorecard.evaluation.calibration import (
    compute_calibration_backtest,
    plot_reliability_curve,
    save_calibration_backtest,
)
from creditscorecard.evaluation.calibration_checks import (
    anchor_gap,
    curve_shape_check,
    mape_by_grade,
)
from creditscorecard.evaluation.discrimination import (
    compute_discrimination,
    save_discrimination,
)
from creditscorecard.evaluation.metrics import metrics_table
from creditscorecard.evaluation.stability import freeze_reference, herfindahl_hirschman_index
from creditscorecard.features.binning import BinningModel
from creditscorecard.features.selection import run_selection
from creditscorecard.features.woe import WoETransformer
from creditscorecard.governance.metadata import build_model_card, save_model_card
from creditscorecard.logging import get_logger
from creditscorecard.model.calibrate import calibrate
from creditscorecard.model.scorecard import build_master_scale, build_scorecard
from creditscorecard.model.train import train_model
from creditscorecard.registry import save_artifacts

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    payload: dict[str, Any]
    metrics: pd.DataFrame
    artifacts_path: str
    figures: dict[str, str]
    mdd_paths: dict[str, str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _feature_columns(df: pd.DataFrame, config: Config) -> list[str]:
    exclude = {config.data.target}
    if config.data.date_column:
        exclude.add(config.data.date_column)
    return [c for c in df.columns if c not in exclude]


def run_pipeline(config: Config) -> PipelineResult:
    _set_seed(config.seed)
    logger.info("=== Scorecard development pipeline start (seed=%d) ===", config.seed)

    df = load_dataset(config)
    df = validate_dataframe(df, config)
    split: SplitData = split_data(df, config)
    logger.info("Split sizes: %s (temporal=%s)", split.describe(), split.temporal)
    if split.sample_design is not None:
        save_sample_design(split.sample_design, config)

    target = config.data.target
    feat_cols = _feature_columns(df, config)
    Xtr, ytr = split.train[feat_cols], split.train[target]

    # --- features (fit on TRAIN only) ---
    binning = BinningModel(config).fit(Xtr, ytr)
    woe = WoETransformer(binning).fit(Xtr, ytr)
    Xtr_woe = woe.transform(Xtr)

    selection = run_selection(woe.iv, Xtr_woe, ytr, config)
    selected = selection.selected_features

    # --- model / calibration / scaling ---
    model = train_model(Xtr_woe, ytr, selected, config)
    calibration = calibrate(model, Xtr_woe, ytr, config)
    scorecard = build_scorecard(model, calibration, woe.woe_maps, config)

    codes_tr = binning.transform(Xtr)[model.features]
    build_master_scale(scorecard, codes_tr, ytr, config)

    # --- evaluate on train/test/OOT (transform only) ---
    split_frames = {"train": split.train, "test": split.test, "oot": split.oot}
    probs: dict[str, tuple] = {}
    scores: dict[str, np.ndarray] = {}
    train_grades: np.ndarray = np.array([])
    for name, frame in split_frames.items():
        codes = binning.transform(frame[feat_cols])[model.features]
        scored = scorecard.score_codes(codes)
        probs[name] = (frame[target].to_numpy(), scored["pd"].to_numpy())
        scores[name] = scored["total_score"].to_numpy()
        if name == "train":
            train_grades = scored["rating_grade"].to_numpy()
    metrics = metrics_table(probs)
    logger.info("Performance:\n%s", metrics.to_string(index=False))

    # --- freeze stability reference on the development (train) sample ---
    reference = freeze_reference(scores["train"], codes_tr, config)

    # --- validation summary: discrimination / concentration / calibration checks ---
    validation = _build_validation_summary(config, metrics, scorecard, calibration, train_grades)
    logger.info(
        "Validation summary: %s",
        {k: v["status"] for k, v in validation.items() if isinstance(v, dict) and "status" in v},
    )

    # --- §5.3 discrimination with uncertainty (bootstrap CIs + .632+ optimism) ---
    discrimination = compute_discrimination(
        probs, config, optimism_inputs=(Xtr_woe, ytr, model.woe_columns)
    )
    save_discrimination(discrimination, config)

    # --- §5.5 calibration backtest (Brier/ECE/HL + per-grade Jeffreys traffic light) ---
    calibration_bt = compute_calibration_backtest(
        probs["train"][0], probs["train"][1], train_grades, config
    )
    save_calibration_backtest(calibration_bt, config)

    # --- figures ---
    fig_dir = config.reports_path() / "figures"
    y_test, p_test = probs["test"]
    figures = {
        "roc": str(curves.plot_roc(probs, fig_dir)),
        "cap": str(curves.plot_cap(probs, fig_dir)),
        "calibration": str(curves.plot_calibration(y_test, p_test, fig_dir)),
        "score_distribution": str(
            curves.plot_score_distribution(scores["train"], probs["train"][0], fig_dir)
        ),
        "reliability_curve": str(plot_reliability_curve(calibration_bt, fig_dir)),
    }

    # --- payload / artifacts ---
    payload = _build_payload(
        config, model, calibration, scorecard, woe, selection, reference, metrics, split, validation
    )
    payload["discrimination"] = discrimination.to_dict()
    payload["calibration_backtest"] = calibration_bt.to_dict()
    tables = _build_tables(woe, selection, scorecard, metrics, model.features)
    artifacts_path = save_artifacts(payload, config, tables)

    # --- governance model card (§5.9): built AFTER artifacts exist so it can hash
    # them; volatile provenance (timestamp/git SHA) lives here, NOT in model.json,
    # keeping the model version deterministic (diagnostic risk R6). ---
    card = build_model_card(config, df, artifacts_dir=config.artifacts_path())
    save_model_card(card, config)

    payload = _reload_version(config, payload)

    # --- MDD (imported lazily to avoid a cycle) ---
    from creditscorecard.reporting import generate_mdd

    mdd_paths = generate_mdd(payload, tables, figures, config)

    logger.info("=== Pipeline complete (version=%s) ===", payload["version"])
    return PipelineResult(
        payload=payload,
        metrics=metrics,
        artifacts_path=str(artifacts_path),
        figures=figures,
        mdd_paths=mdd_paths,
    )


def _build_payload(
    config, model, calibration, scorecard, woe, selection, reference, metrics, split, validation
) -> dict[str, Any]:
    return {
        "package_version": __version__,
        "target": config.data.target,
        "date_column": config.data.date_column,
        "temporal_validation": split.temporal,
        "convention": {
            "positive_class": "Good",
            "woe": "ln(%Good/%Bad); positive WoE => better applicant",
            "event": "default == 1 (Bad); model estimates P(Bad)",
            "expected_coef_sign": "negative",
        },
        "sample": _sample_info(config, split),
        "sample_design": (
            split.sample_design.summary_dict() if split.sample_design is not None else {}
        ),
        "governance": config.governance.model_dump(),
        "selected_features": model.features,
        "binning_specs": {f: binning_spec(woe, f) for f in model.features},
        "woe_maps": {f: {str(k): v for k, v in woe.woe_maps[f].items()} for f in model.features},
        "iv": {f: float(v) for f, v in woe.iv.items()},
        "model": {
            "engine": config.model.engine,
            "intercept": model.intercept,
            "coefficients": model.coefficients,
            "std_errors": model.std_errors,
            "p_values": model.p_values,
            "excluded_wrong_sign": model.excluded_wrong_sign,
            "sign_ok": all(s.ok for s in model.sign_checks),
            "parity_passed": model.parity.passed,
            "parity_max_abs_diff": model.parity.max_abs_diff,
            "n_obs": model.n_obs,
        },
        "calibration": {
            "method": calibration.method,
            "anchor_rate": calibration.anchor_rate,
            "intercept_shift": calibration.intercept_shift,
            "calibrated_intercept": calibration.calibrated_intercept(model),
            "mean_pd_before": calibration.mean_pd_before,
            "mean_pd_after": calibration.mean_pd_after,
        },
        "scaling": {
            "factor": scorecard.factor,
            "offset": scorecard.offset,
            "pdo": scorecard.pdo,
            "target_score": scorecard.target_score,
            "target_odds": scorecard.target_odds,
            "round_points": scorecard.round_points,
        },
        "points_card": scorecard.points_card_serialisable(),
        "master_scale": {
            "grades": scorecard.master_scale.grades,
            "score_edges": scorecard.master_scale.score_edges,
            "table": scorecard.master_scale.table,
        },
        "stability_reference": reference.to_dict(),
        "monitoring_thresholds": {
            "psi_warn": config.monitoring.psi_warn,
            "psi_alert": config.monitoring.psi_alert,
        },
        "selection_trail": {
            "dropped_low_iv": selection.dropped_low_iv,
            "suspicious_iv": selection.suspicious_iv,
            "vif_dropped": selection.vif_dropped,
            "forward_trail": selection.forward_trail,
        },
        "performance": {
            row["split"]: {"auc": row["auc"], "gini": row["gini"], "ks": row["ks"]}
            for _, row in metrics.iterrows()
        },
        "validation_summary": validation,
    }


def binning_spec(woe: WoETransformer, feature: str) -> dict:
    return woe.binning.specs[feature].to_dict()


def _threshold_row(name: str, value: float, threshold: float, op: str) -> dict[str, Any]:
    """op: 'min' -> pass if value >= threshold; 'max' -> pass if value <= threshold."""
    passed = value >= threshold if op == "min" else value <= threshold
    return {
        "metric": name,
        "value": value,
        "threshold": threshold,
        "status": "PASS" if passed else "FAIL",
    }


def _build_validation_summary(
    config: Config,
    metrics: pd.DataFrame,
    scorecard,
    calibration,
    train_grades: np.ndarray,
) -> dict[str, Any]:
    """Discrimination / concentration / calibration checks (validation framework).

    Discrimination (Gini/KS) is checked on the out-of-time split, since that is
    the sample that matters for generalisation. Concentration (HHI) and
    calibration accuracy (MAPE/anchor/curve-shape) are checked on the
    development (train) sample, matching where the Master Scale was built.
    """
    v = config.validation
    oot_row = metrics.loc[metrics["split"] == "oot"].iloc[0]

    table = scorecard.master_scale.table
    hhi = herfindahl_hirschman_index(train_grades)
    mape = mape_by_grade(table)
    gap = anchor_gap(calibration.mean_pd_after, calibration.anchor_rate)
    curve = curve_shape_check(table, n_se=v.curve_shape_n_se)

    return {
        "discriminatory_power": {
            "gini": _threshold_row("gini_oot", float(oot_row["gini"]), v.gini_min, "min"),
            "ks": _threshold_row("ks_oot", float(oot_row["ks"]), v.ks_min, "min"),
        },
        "stability_concentration": {
            "hhi": _threshold_row("hhi_train_grades", hhi, v.hhi_max, "max"),
        },
        "calibration_accuracy": {
            "mape": _threshold_row("mape_by_grade", mape, v.mape_max, "max"),
            "anchor_gap": _threshold_row("anchor_gap", abs(gap), v.anchor_gap_max, "max"),
            "curve_shape": {
                "monotonic": curve.monotonic,
                "n_se": curve.n_se,
                "all_within_band": curve.all_within_band,
                "status": "PASS" if curve.monotonic and curve.all_within_band else "FAIL",
                "bands": [b.__dict__ for b in curve.bands],
            },
        },
    }


def _sample_info(config: Config, split: SplitData) -> dict[str, Any]:
    target = config.data.target
    info: dict[str, Any] = {
        "train": len(split.train),
        "test": len(split.test),
        "oot": len(split.oot),
        "temporal": split.temporal,
        "train_bad_rate": float(split.train[target].mean()),
        "test_bad_rate": float(split.test[target].mean()),
        "oot_bad_rate": float(split.oot[target].mean()),
    }
    date_col = config.data.date_column
    if split.temporal and date_col:
        dev_dates = pd.concat([split.train[date_col], split.test[date_col]])
        info["date_ranges"] = {
            "dev_min": str(dev_dates.min()),
            "dev_max": str(dev_dates.max()),
            "oot_min": str(split.oot[date_col].min()),
            "oot_max": str(split.oot[date_col].max()),
        }
    return info


def _build_tables(woe, selection, scorecard, metrics, selected_features) -> dict[str, pd.DataFrame]:
    tables = {
        "iv": woe.iv_frame(),
        "selection_iv": selection.iv_table,
        "metrics": metrics,
        "master_scale": pd.DataFrame(scorecard.master_scale.table),
    }
    for feat in selected_features:
        tables[f"woe_{feat}"] = _woe_detail(woe, feat)
    return tables


def _woe_detail(woe: WoETransformer, feature: str) -> pd.DataFrame:
    """Per-characteristic binning + WoE + IV table with human-readable labels."""
    labels = woe.binning.specs[feature].labels
    tbl = woe.tables[feature].copy()
    tbl.insert(0, "bin", tbl["code"].map(lambda c: labels.get(int(c), str(c))))
    return tbl.drop(columns=["code"])


def _reload_version(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    from creditscorecard.registry import load_payload

    return load_payload(config)
