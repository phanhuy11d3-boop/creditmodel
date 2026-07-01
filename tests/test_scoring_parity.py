"""Scoring parity: API score == artifact score == pipeline score for the same input."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from creditscorecard.scoring import ScoringModel


def _records(dataset, features, n=5):
    sub = dataset.iloc[:n][features]
    return json.loads(sub.to_json(orient="records"))


def test_artifact_matches_inmemory_payload(config, pipeline_payload, dataset):
    from_artifacts = ScoringModel.from_config(config)  # loaded from disk
    in_memory = ScoringModel(pipeline_payload)  # from the returned payload
    rows = dataset.iloc[:20][from_artifacts.selected_features]

    a = from_artifacts.score_frame(rows)
    b = in_memory.score_frame(rows)
    np.testing.assert_array_equal(a["total_score"].to_numpy(), b["total_score"].to_numpy())
    np.testing.assert_allclose(a["pd"].to_numpy(), b["pd"].to_numpy(), rtol=1e-12)
    assert list(a["rating_grade"]) == list(b["rating_grade"])


@pytest.fixture
def client(config, pipeline_payload, monkeypatch):
    from app import api

    model = ScoringModel.from_config(config)
    api.get_model.cache_clear()
    monkeypatch.setattr(api, "get_model", lambda: model)
    return TestClient(api.app), model


def test_api_score_matches_pipeline(client, dataset):
    tc, model = client
    row = _records(dataset, model.selected_features, 1)[0]
    resp = tc.post("/score", json={"features": row})
    assert resp.status_code == 200
    data = resp.json()
    ref = model.score_one(row)
    assert data["score"] == ref["score"]
    assert abs(data["pd"] - ref["pd"]) < 1e-9
    assert data["rating_grade"] == ref["rating_grade"]
    assert len(data["reason_codes"]) >= 1


def test_api_health_and_info(client):
    tc, model = client
    assert tc.get("/health").json()["status"] == "ok"
    info = tc.get("/model-info").json()
    assert info["version"] == model.version
    assert info["selected_features"] == model.selected_features


def test_api_batch_score(client, dataset):
    tc, model = client
    records = _records(dataset, model.selected_features, 4)
    resp = tc.post("/batch-score", json={"records": records})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert len(body["results"]) == 4


def test_api_missing_feature_returns_422(client):
    tc, _ = client
    resp = tc.post("/score", json={"features": {"not_a_real_feature": 1}})
    assert resp.status_code == 422
