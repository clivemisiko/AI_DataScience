"""Model factories and evaluation pipelines for Core Development."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix
)


def get_preprocessor(X: pd.DataFrame, scale_numeric: bool = True):
    """Build a ColumnTransformer handling numeric and categorical columns."""
    numeric_cols = list(X.select_dtypes("number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)

    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, cat_cols),
    ])


def create_baseline_model(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Build M2.1 Baseline Logistic Regression Pipeline."""
    preprocess = get_preprocessor(X, scale_numeric=True)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def create_rf_model(
    X: pd.DataFrame,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Pipeline:
    """Build Iteration 1 Random Forest Pipeline."""
    preprocess = get_preprocessor(X, scale_numeric=False)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def create_core_xgboost_model(
    X: pd.DataFrame,
    scale_pos_weight: float = 1.0,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Pipeline:
    """Build Iteration 2 Core XGBoost Pipeline."""
    preprocess = get_preprocessor(X, scale_numeric=False)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Compute comprehensive performance metrics."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "brier_score": float(brier_score_loss(y, prob)),
    }

