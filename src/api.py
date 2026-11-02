"""FastAPI prediction service for the packaged Adult Income model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_income


class IncomeRecord(BaseModel):
    age: int = Field(ge=16, le=100)
    workclass: str = Field(min_length=1)
    education_num: int = Field(ge=1, le=16)
    marital_status: str = Field(min_length=1)
    occupation: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    capital_gain: float = Field(ge=0)
    capital_loss: float = Field(ge=0)
    hours_per_week: int = Field(ge=1, le=100)
    native_country: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    label: str
    threshold: float


def create_app(model_path: str | None = None, threshold_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Adult Income Prediction API", version="1.0.0")
    configured_model_path = Path(
        model_path or os.getenv("MODEL_PATH", "mlops_artifacts/production_model.joblib")
    )
    configured_threshold_path = Path(
        threshold_path or os.getenv("THRESHOLD_PATH", "mlops_artifacts/threshold.json")
    )
    state: dict[str, Any] = {"model": None, "threshold": None, "error": None}

    @app.on_event("startup")
    def load_model() -> None:
        try:
            state["model"] = joblib.load(configured_model_path)
            state["threshold"] = float(
                json.loads(configured_threshold_path.read_text(encoding="utf-8"))["threshold"]
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            state["error"] = f"Model bundle is unavailable: {exc}"

    @app.get("/health")
    def health() -> dict[str, str]:
        if state["error"] is not None:
            raise HTTPException(status_code=503, detail=state["error"])
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: IncomeRecord) -> dict[str, Any]:
        if state["error"] is not None or state["model"] is None:
            raise HTTPException(status_code=503, detail=state["error"] or "Model is not loaded.")
        return predict_income(state["model"], state["threshold"], record.model_dump())

    return app


app = create_app()
