"""MLflow experiment tracking and model packaging for the Adult Income model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import get_stratified_split, load_raw_data, prepare_data
from src.features import add_engineered_features
from src.models import (
    create_baseline_model,
    create_calibrated_model,
    create_core_xgboost_model,
    create_rf_model,
    find_optimal_threshold,
)


def _data_split(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_raw_data(data_path)
    X, y, protected = prepare_data(df)
    X_train, X_test, y_train, y_test, _, _ = get_stratified_split(
        X, y, protected, test_size=0.2, random_state=42
    )
    return add_engineered_features(X_train), add_engineered_features(X_test), y_train, y_test


def _evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold, _ = find_optimal_threshold(y_test.to_numpy(), probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }, threshold


def train_and_log_experiments(
    data_path: str = "adult.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "adult-income-capstone",
    output_dir: str = "mlops_artifacts",
) -> pd.DataFrame:
    """Train three different models, log runs, and package the production model."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    X_train, X_test, y_train, y_test = _data_split(data_path)

    configurations = [
        ("baseline_logistic", create_baseline_model(X_train), {"model_type": "logistic_regression"}),
        (
            "random_forest",
            create_rf_model(X_train, max_depth=12, min_samples_leaf=2),
            {"model_type": "random_forest", "max_depth": 12, "min_samples_leaf": 2},
        ),
        (
            "calibrated_xgboost",
            create_calibrated_model(
                create_core_xgboost_model(X_train, n_estimators=150, max_depth=5, learning_rate=0.1),
                method="isotonic",
                cv=5,
            ),
            {
                "model_type": "calibrated_xgboost",
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.1,
                "calibration": "isotonic",
                "cv": 5,
            },
        ),
    ]

    rows = []
    production_model = None
    production_threshold = None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for run_name, model, params in configurations:
        with mlflow.start_run(run_name=run_name) as run:
            model.fit(X_train, y_train)
            metrics, threshold = _evaluate(model, X_test, y_test)
            mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_param("random_state", 42)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metric("decision_threshold", threshold)
            mlflow.log_metrics(metrics)

            metrics_path = output_path / f"{run_name}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(
                model,
                "model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.calibration._CalibratedClassifier",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )
            rows.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    **params,
                    **metrics,
                    "decision_threshold": threshold,
                }
            )
            if run_name == "calibrated_xgboost":
                production_model = model
                production_threshold = threshold

    if production_model is None or production_threshold is None:
        raise RuntimeError("The production configuration did not produce a trained model.")

    joblib.dump(production_model, output_path / "production_model.joblib")
    (output_path / "threshold.json").write_text(
        json.dumps({"threshold": production_threshold}, indent=2), encoding="utf-8"
    )
    results = pd.DataFrame(rows)
    results.to_csv(output_path / "runs_export.csv", index=False)
    results.to_csv("track_l_runs_export.csv", index=False)
    return results


if __name__ == "__main__":
    results = train_and_log_experiments()
    print(results[["run_name", "roc_auc", "f1", "accuracy", "decision_threshold"]].to_string(index=False))
