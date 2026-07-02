"""Shared fixtures. Uses the offline synthetic adapter for deterministic tests."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from creditscorecard.config import Config, load_config
from creditscorecard.data.adapters import load_dataset
from creditscorecard.data.schema import validate_dataframe
from creditscorecard.data.split import SplitData, split_data
from creditscorecard.features.binning import BinningModel
from creditscorecard.features.selection import SelectionResult, run_selection
from creditscorecard.features.woe import WoETransformer
from creditscorecard.model.calibrate import CalibrationResult, calibrate
from creditscorecard.model.scorecard import ScorecardModel, build_master_scale, build_scorecard
from creditscorecard.model.train import TrainedModel, train_model

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Home-credit config but offline (synthetic) and writing to a temp dir."""
    cfg = load_config("configs/home_credit.yaml")
    cfg.data.adapter = "synthetic"
    cfg.data.target = "default"  # the synthetic fallback frame always uses this column
    cfg.data.date_column = "application_date"
    # Keep bootstrap/SHAP/challenger work small so the suite stays fast (diagnostic risk R3);
    # correctness of the CI/benchmark machinery is pinned by its own dedicated unit tests.
    cfg.discrimination.bootstrap_iterations = 120
    cfg.explainability.shap_sample_size = 80
    cfg.benchmark.challenger_params = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}
    tmp = tmp_path_factory.mktemp("run")
    cfg.paths.artifacts_dir = str(tmp / "artifacts")
    cfg.paths.reports_dir = str(tmp / "reports")
    cfg.paths.data_dir = str(tmp / "data")
    return cfg


@pytest.fixture(scope="session")
def dataset(config: Config) -> pd.DataFrame:
    return validate_dataframe(load_dataset(config), config)


@pytest.fixture(scope="session")
def split(dataset: pd.DataFrame, config: Config) -> SplitData:
    return split_data(dataset, config)


@dataclass
class Fitted:
    config: Config
    split: SplitData
    feat_cols: list[str]
    binning: BinningModel
    woe: WoETransformer
    Xtr_woe: pd.DataFrame
    selection: SelectionResult
    model: TrainedModel
    calibration: CalibrationResult
    scorecard: ScorecardModel
    ytr: pd.Series


@pytest.fixture(scope="session")
def fitted(config: Config, split: SplitData) -> Fitted:
    target = config.data.target
    feat_cols = [c for c in split.train.columns if c not in (target, config.data.date_column)]
    Xtr, ytr = split.train[feat_cols], split.train[target]
    binning = BinningModel(config).fit(Xtr, ytr)
    woe = WoETransformer(binning).fit(Xtr, ytr)
    Xtr_woe = woe.transform(Xtr)
    selection = run_selection(woe.iv, Xtr_woe, ytr, config)
    model = train_model(Xtr_woe, ytr, selection.selected_features, config)
    calibration = calibrate(model, Xtr_woe, ytr, config)
    scorecard = build_scorecard(model, calibration, woe.woe_maps, config)
    codes_tr = binning.transform(Xtr)[model.features]
    build_master_scale(scorecard, codes_tr, ytr, config)
    return Fitted(
        config=config,
        split=split,
        feat_cols=feat_cols,
        binning=binning,
        woe=woe,
        Xtr_woe=Xtr_woe,
        selection=selection,
        model=model,
        calibration=calibration,
        scorecard=scorecard,
        ytr=ytr,
    )


@pytest.fixture(scope="session")
def pipeline_payload(config: Config) -> dict[str, Any]:
    from creditscorecard.pipeline import run_pipeline

    return run_pipeline(config).payload
