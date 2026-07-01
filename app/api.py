"""FastAPI scoring service. Loads frozen artifacts only (no training deps).

Endpoints:
* ``POST /score``       — one applicant -> score, pd, rating grade, breakdown, reasons.
* ``POST /batch-score`` — JSON list of applicants.
* ``GET  /model-info``  — feature list, scaling constants, model version.
* ``GET  /health``      — liveness + loaded model version.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from creditscorecard.config import load_config
from creditscorecard.logging import configure_logging, get_logger
from creditscorecard.scoring import ScoringModel

configure_logging()
logger = get_logger(__name__)

CONFIG_ENV = "CREDITSCORECARD_CONFIG"
DEFAULT_CONFIG = "configs/home_credit.yaml"

app = FastAPI(title="Credit PD Scorecard", version="0.1.0")


@lru_cache(maxsize=1)
def get_model() -> ScoringModel:
    config_path = os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)
    cfg = load_config(config_path)
    model = ScoringModel.from_config(cfg)
    logger.info("Loaded scoring model version=%s", model.version)
    return model


class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Applicant characteristic values.")
    top_k_reasons: int = Field(4, ge=1, le=20)


class BatchScoreRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class ReasonCodeOut(BaseModel):
    feature: str
    actual_points: float
    max_points: float
    shortfall: float
    bin_label: str


class ScoreResponse(BaseModel):
    score: int
    pd: float
    rating_grade: str
    points_breakdown: dict[str, float]
    reason_codes: list[ReasonCodeOut]


def _score_applicant(model: ScoringModel, applicant: dict[str, Any], top_k: int) -> dict[str, Any]:
    missing = [f for f in model.selected_features if f not in applicant]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")
    try:
        return model.score_one(applicant, top_k=top_k)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    try:
        version = get_model().version
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found") from None
    return {"status": "ok", "version": version}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return get_model().model_info()


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> dict[str, Any]:
    model = get_model()
    return _score_applicant(model, request.features, request.top_k_reasons)


@app.post("/batch-score")
def batch_score(request: BatchScoreRequest) -> dict[str, Any]:
    model = get_model()
    results = [_score_applicant(model, rec, top_k=4) for rec in request.records]
    return {"version": model.version, "count": len(results), "results": results}
