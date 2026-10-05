"""Unit and integration tests for Core Model Development (Track M - Gate M2.2)."""

import pytest
import numpy as np
import pandas as pd
from src.data import load_raw_data, prepare_data, get_stratified_split
from src.features import add_engineered_features
from src.models import (
    create_baseline_model,
    create_rf_model,
    create_core_xgboost_model,
    evaluate_model,
)


@pytest.fixture(scope="session")
def dataset():
    """Load and prepare train and test sets once for the test session."""
    df = load_raw_data()
    X, y, protected = prepare_data(df)
    X_train, X_test, y_train, y_test, prot_train, prot_test = get_stratified_split(
        X, y, protected, test_size=0.2, random_state=42
    )
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "y_train": y_train,
        "y_test": y_test,
        "prot_train": prot_train,
        "prot_test": prot_test,
    }


def test_data_preparation_contract(dataset):
    """Test data cleaning and contract."""
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    # Redundant and protected columns removed from X
    for col in ["education", "fnlwgt", "sex", "race", "income"]:
        assert col not in X_train.columns

    # Check stratification
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.005
    assert abs(train_rate - 0.2408) < 0.01


def test_feature_engineering(dataset):
    """Verify engineered features are present, non-null, and have correct properties."""
    X_train_fe = dataset["X_train_fe"]

    assert "log_capital_gain" in X_train_fe.columns
    assert "age_group" in X_train_fe.columns
    assert "edu_x_hours" in X_train_fe.columns

    assert X_train_fe["log_capital_gain"].isna().sum() == 0
    assert X_train_fe["age_group"].isna().sum() == 0
    assert X_train_fe["edu_x_hours"].isna().sum() == 0

    # log1p verification
    assert np.allclose(
        X_train_fe["log_capital_gain"], np.log1p(dataset["X_train"]["capital_gain"])
    )


def test_baseline_reproduction(dataset):
    """Confirm Baseline Logistic Regression matches M2.1 benchmark."""
    baseline = create_baseline_model(dataset["X_train"])
    baseline.fit(dataset["X_train"], dataset["y_train"])
    scores = evaluate_model(baseline, dataset["X_test"], dataset["y_test"])

    assert scores["roc_auc"] >= 0.90
    assert scores["accuracy"] >= 0.85
    assert scores["f1"] >= 0.66


def test_core_model_beats_m21_baseline(dataset):
    """Verify Iteration 2 (Core XGBoost) measurably beats M2.1 Baseline on ROC-AUC, F1, and Accuracy."""
    # M2.1 Baseline reference: ROC-AUC 0.9083, Accuracy 0.8548, F1 0.6704
    core_xgb = create_core_xgboost_model(dataset["X_train_fe"], random_state=42)
    core_xgb.fit(dataset["X_train_fe"], dataset["y_train"])
    scores = evaluate_model(core_xgb, dataset["X_test_fe"], dataset["y_test"])

    # Measurable improvement criteria
    assert scores["roc_auc"] > 0.920, f"Expected ROC-AUC > 0.920, got {scores['roc_auc']:.4f}"
    assert scores["accuracy"] > 0.870, f"Expected Accuracy > 0.870, got {scores['accuracy']:.4f}"
    assert scores["f1"] > 0.700, f"Expected F1 > 0.700, got {scores['f1']:.4f}"


def test_threshold_tuning_impact(dataset):
    """Test that adjusting decision threshold enables controllable precision/recall trade-off."""
    core_xgb = create_core_xgboost_model(dataset["X_train_fe"], random_state=42)
    core_xgb.fit(dataset["X_train_fe"], dataset["y_train"])

    scores_default = evaluate_model(core_xgb, dataset["X_test_fe"], dataset["y_test"], threshold=0.5)
    scores_high_recall = evaluate_model(core_xgb, dataset["X_test_fe"], dataset["y_test"], threshold=0.3)

    assert scores_high_recall["recall"] > scores_default["recall"]
    assert scores_high_recall["precision"] < scores_default["precision"]

