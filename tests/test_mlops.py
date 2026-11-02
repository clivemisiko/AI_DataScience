"""Track L tests for experiment logging, packaging, and API validation."""

import json

import joblib
import numpy as np
from fastapi.testclient import TestClient

from src.api import create_app


class StubModel:
    def predict_proba(self, rows):
        return np.array([[0.2, 0.8] for _ in range(len(rows))])


def test_api_returns_prediction_for_valid_record(tmp_path):
    model_path = tmp_path / "model.joblib"
    threshold_path = tmp_path / "threshold.json"
    joblib.dump(StubModel(), model_path)
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    with TestClient(create_app(str(model_path), str(threshold_path))) as client:
        response = client.post(
            "/predict",
            json={
                "age": 48,
                "workclass": "Private",
                "education_num": 16,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "capital_gain": 15000,
                "capital_loss": 0,
                "hours_per_week": 50,
                "native_country": "United-States",
            },
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["label"] == ">50K"


def test_api_rejects_bad_input(tmp_path):
    model_path = tmp_path / "model.joblib"
    threshold_path = tmp_path / "threshold.json"
    joblib.dump(StubModel(), model_path)
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    with TestClient(create_app(str(model_path), str(threshold_path))) as client:
        response = client.post("/predict", json={"age": 12})

    assert response.status_code == 422
