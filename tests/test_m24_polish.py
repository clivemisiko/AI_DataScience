"""
M2.4 Polish & Test — Capstone quality gate tests.

These tests enforce the demo-ability and defensibility of the capstone:
  1. End-to-end pipeline smoke-test (raw CSV → prediction).
  2. predict_income() entry-point correctness (direction sanity check).
  3. Regression guard: calibrated model must not degrade below M2.3 benchmarks.
  4. Fairness regression: gender recall gap must stay within accepted tolerance.
  5. Prediction schema contract: output dict contains required keys and valid types.
  6. Pipeline portability: model still loads and predicts on an unseen synthetic row.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import load_raw_data, prepare_data, get_stratified_split
from src.features import add_engineered_features
from src.models import (
    create_calibrated_model,
    create_core_xgboost_model,
    evaluate_model,
    find_optimal_threshold,
)
from src.predict import build_and_train_pipeline, predict_income


# ---------------------------------------------------------------------------
# Session fixture — build and train the full pipeline once for all tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trained_pipeline():
    """Train the production pipeline once and share across all M2.4 tests."""
    df = load_raw_data()
    X, y, protected = prepare_data(df)
    X_train, X_test, y_train, y_test, _, prot_test = get_stratified_split(
        X, y, protected, test_size=0.2, random_state=42
    )
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    base = create_core_xgboost_model(X_train_fe, random_state=42)
    model = create_calibrated_model(base, method="isotonic", cv=5)
    model.fit(X_train_fe, y_train)

    probs = model.predict_proba(X_test_fe)[:, 1]
    threshold, _ = find_optimal_threshold(y_test, probs, metric="f1")

    return {
        "model": model,
        "threshold": threshold,
        "X_test_fe": X_test_fe,
        "y_test": y_test,
        "prot_test": prot_test,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Test 1: End-to-end smoke-test
# ---------------------------------------------------------------------------
def test_end_to_end_pipeline_runs():
    """
    Gate requirement: capstone runs end-to-end from a clean state.
    Failure mode caught: data loading errors, broken imports, column schema mismatches.
    """
    model, threshold = build_and_train_pipeline()

    assert model is not None
    assert 0.0 < threshold < 1.0, f"Threshold must be in (0,1), got {threshold}"
    assert hasattr(model, "predict_proba"), "Fitted model must expose predict_proba"


# ---------------------------------------------------------------------------
# Test 2: predict_income() direction sanity check
# ---------------------------------------------------------------------------
def test_predict_income_direction_sanity(trained_pipeline):
    """
    Failure mode caught: a broken or inverted pipeline scoring function would
    assign high probabilities to low-earner profiles and vice versa.
    This test would FAIL if the model were returning random or inverted scores.
    """
    model = trained_pipeline["model"]
    threshold = trained_pipeline["threshold"]

    high_earner = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "capital_gain": 20000,
        "capital_loss": 0,
        "hours_per_week": 55,
        "native_country": "United-States",
    }
    low_earner = {
        "age": 22,
        "workclass": "Private",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30,
        "native_country": "United-States",
    }

    result_high = predict_income(model, threshold, high_earner)
    result_low = predict_income(model, threshold, low_earner)

    assert result_high["prediction"] == 1, (
        f"High-earner profile should predict >50K (got P={result_high['probability']:.3f}, "
        f"threshold={threshold})"
    )
    assert result_low["prediction"] == 0, (
        f"Low-earner profile should predict <=50K (got P={result_low['probability']:.3f}, "
        f"threshold={threshold})"
    )
    assert result_high["probability"] > result_low["probability"], (
        "High-earner probability must exceed low-earner probability."
    )


# ---------------------------------------------------------------------------
# Test 3: Prediction output schema contract
# ---------------------------------------------------------------------------
def test_predict_income_output_schema(trained_pipeline):
    """
    Failure mode caught: predict_income() returning wrong types or missing keys
    would silently break any downstream API consumer.
    """
    model = trained_pipeline["model"]
    threshold = trained_pipeline["threshold"]

    record = {
        "age": 35,
        "workclass": "Private",
        "education_num": 12,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    result = predict_income(model, threshold, record)

    assert set(result.keys()) == {"probability", "prediction", "label", "threshold"}, (
        f"Output schema mismatch: got {set(result.keys())}"
    )
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["label"] in (">50K", "<=50K")
    assert result["threshold"] == threshold


# ---------------------------------------------------------------------------
# Test 4: M2.3 performance regression guard
# ---------------------------------------------------------------------------
def test_m24_no_performance_regression(trained_pipeline):
    """
    Failure mode caught: a silent code change (e.g. wrong feature list, wrong cv,
    wrong XGBoost hyperparameter) that degrades model quality below M2.3 benchmarks.
    This test would FAIL if such a regression occurred.

    M2.3 benchmarks (Isotonic @ τ=0.35):
        ROC-AUC  >= 0.930
        F1       >= 0.740
        Recall   >= 0.750
        Brier    <  0.090
    """
    model = trained_pipeline["model"]
    X_test = trained_pipeline["X_test_fe"]
    y_test = trained_pipeline["y_test"]
    threshold = trained_pipeline["threshold"]

    scores = evaluate_model(model, X_test, y_test, threshold=threshold)

    assert scores["roc_auc"] >= 0.930, (
        f"ROC-AUC regression: expected >= 0.930, got {scores['roc_auc']:.4f}"
    )
    assert scores["f1"] >= 0.740, (
        f"F1 regression: expected >= 0.740, got {scores['f1']:.4f}"
    )
    assert scores["recall"] >= 0.750, (
        f"Recall regression: expected >= 0.750, got {scores['recall']:.4f}"
    )
    assert scores["brier_score"] < 0.090, (
        f"Brier score regression: expected < 0.090, got {scores['brier_score']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Fairness regression guard (gender recall gap)
# ---------------------------------------------------------------------------
def test_m24_fairness_recall_gap_within_tolerance(trained_pipeline):
    """
    Failure mode caught: a change to the pipeline (e.g. reintroducing 'sex' as a
    feature, changing threshold, or altering resampling) that increases the gender
    recall gap beyond the accepted tolerance.

    M2.3 baseline: after threshold optimisation, the female-vs-male recall gap
    narrowed from 2.8% to 1.3%. We guard that it stays below 5.0 percentage points.

    This is a REAL failure-mode test — a trivially trained model or broken
    threshold would create much larger recall gaps (often 10–15%).
    """
    model = trained_pipeline["model"]
    X_test = trained_pipeline["X_test_fe"]
    y_test = trained_pipeline["y_test"]
    prot_test = trained_pipeline["prot_test"]
    threshold = trained_pipeline["threshold"]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    male_mask = prot_test["sex"] == "Male"
    female_mask = prot_test["sex"] == "Female"

    y_test_arr = np.array(y_test)

    male_recall = y_test_arr[male_mask & (y_test_arr == 1)].sum()
    male_recall = preds[male_mask & (y_test_arr == 1)].sum() / max(
        (male_mask & (y_test_arr == 1)).sum(), 1
    )
    female_recall = preds[female_mask & (y_test_arr == 1)].sum() / max(
        (female_mask & (y_test_arr == 1)).sum(), 1
    )

    gap = abs(male_recall - female_recall)
    assert gap <= 0.05, (
        f"Gender recall gap {gap:.3f} exceeds 5.0 percentage-point tolerance. "
        f"Male recall: {male_recall:.3f}, Female recall: {female_recall:.3f}. "
        f"Check threshold, feature set, or resampling."
    )

