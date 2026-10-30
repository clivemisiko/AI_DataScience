"""Model factories and evaluation pipelines for Core Development."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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




def create_calibrated_model(
    base_pipeline: Pipeline,
    method: str = "isotonic",
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Wrap an existing pipeline with CalibratedClassifierCV to calibrate posterior probabilities.
    method can be 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling).
    """
    return CalibratedClassifierCV(estimator=base_pipeline, method=method, cv=cv)


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1"
) -> tuple[float, float]:
    """
    Search thresholds from 0.05 to 0.95 with step 0.01 to find optimal score.
    Returns: (best_threshold, best_score)
    """
    best_threshold = 0.5
    best_score = -1.0

    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (probabilities >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds)
        elif metric == "balanced_accuracy":
            score = accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return round(best_threshold, 2), round(best_score, 4)


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Compute comprehensive performance metrics including calibration sharpness and PR-AUC."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "brier_score": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob)),
    }


