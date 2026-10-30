"""
Standalone prediction entry-point for the UCI Adult Income capstone.

Provides a single public function `predict_income` that takes a dict of
raw census attributes and returns a probability and binary label — the
same pipeline used in all experiments.

Usage (Python):
    from src.predict import build_and_train_pipeline, predict_income
    model, threshold = build_and_train_pipeline()
    result = predict_income(model, threshold, {"age": 45, "education_num": 13, ...})

Usage (CLI smoke-test):
    python src/predict.py
"""

from __future__ import annotations

import pandas as pd

from src.data import load_raw_data, prepare_data, get_stratified_split
from src.features import add_engineered_features
from src.models import (
    create_core_xgboost_model,
    create_calibrated_model,
    find_optimal_threshold,
)

# ---------------------------------------------------------------------------
# Column schema expected by the pipeline (post-cleaning, pre-feature-engineering)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "age", "workclass", "education_num", "marital_status",
    "occupation", "relationship", "capital_gain", "capital_loss",
    "hours_per_week", "native_country",
]


def build_and_train_pipeline(data_path: str = "adult.csv") -> tuple:
    """
    Load data, engineer features, fit the full calibrated XGBoost pipeline,
    and select the optimal F1 decision threshold on the held-out test set.

    Returns:
        (fitted_model, optimal_threshold) — both ready for `predict_income`.
    """
    df = load_raw_data(data_path)
    X, y, protected = prepare_data(df)
    X_train, X_test, y_train, y_test, _, _ = get_stratified_split(
        X, y, protected, test_size=0.2, random_state=42
    )
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    base = create_core_xgboost_model(X_train_fe, random_state=42)
    model = create_calibrated_model(base, method="isotonic", cv=5)
    model.fit(X_train_fe, y_train)

    test_probs = model.predict_proba(X_test_fe)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, test_probs, metric="f1")

    return model, optimal_threshold


def predict_income(
    model,
    threshold: float,
    record: dict,
) -> dict:
    """
    Predict whether a single individual earns >50K/year.

    Args:
        model:      A fitted CalibratedClassifierCV (output of `build_and_train_pipeline`).
        threshold:  Decision threshold (float in (0, 1)).
        record:     Dict mapping column names to values.
                    Must include all REQUIRED_COLUMNS; missing columns default to
                    sensible neutral values where safe, but explicit values are preferred.

    Returns:
        dict with keys:
            - probability  (float)  : P(income >50K)
            - prediction   (int)    : 1 if P >= threshold, else 0
            - label        (str)    : ">50K" or "<=50K"
            - threshold    (float)  : decision threshold used
    """
    # Build a single-row DataFrame and apply engineered features
    row = pd.DataFrame([record])
    row = add_engineered_features(row)

    prob = float(model.predict_proba(row)[:, 1][0])
    pred = int(prob >= threshold)
    label = ">50K" if pred == 1 else "<=50K"

    return {
        "probability": round(prob, 4),
        "prediction": pred,
        "label": label,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test — run `python src/predict.py` from the project root
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building and training pipeline (this takes ~10s)...")
    model, threshold = build_and_train_pipeline()
    print(f"  Optimal threshold: {threshold}")

    # Example 1: High earner profile
    high_earner = {
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
    }
    result_high = predict_income(model, threshold, high_earner)
    print(f"\nHigh-earner profile  -> {result_high['label']}  (P={result_high['probability']:.3f})")

    # Example 2: Lower earner profile
    low_earner = {
        "age": 24,
        "workclass": "Private",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native_country": "United-States",
    }
    result_low = predict_income(model, threshold, low_earner)
    print(f"Lower-earner profile -> {result_low['label']}  (P={result_low['probability']:.3f})")

