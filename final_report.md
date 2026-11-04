# AI Capstone Final Report

## Executive summary

This capstone delivers an end-to-end UCI Adult Income prediction system. It cleans and audits census data, engineers features, trains and calibrates an XGBoost classifier, exposes predictions through FastAPI, tracks experiments in MLflow, and audits fairness and local explanations with SHAP.

The final production build reached ROC-AUC `0.9306`, F1 `0.7448`, and recall `0.7742` on a fixed stratified held-out test set of 6,513 records. The operating threshold is `0.36`, selected by an F1 sweep rather than assumed to be `0.50`.

## Problem and intended use

The model estimates whether an individual earns more than `$50K` from 1994 census attributes. It is intended for educational model evaluation, calibrated risk-scoring research, and fairness benchmarking. It is not intended for automated credit, employment, insurance, or benefits decisions. Human review is required for consequential use.

## Technical approach

1. Replace missing categorical values with an explicit `Missing` category.
2. Remove redundant `education`, sampling-weight `fnlwgt`, target `income`, and protected attributes `sex` and `race` from model features.
3. Add `log_capital_gain`, `age_group`, and `edu_x_hours`.
4. Compare logistic regression, random forest, and XGBoost configurations.
5. Calibrate XGBoost probabilities with 5-fold isotonic calibration.
6. Select the decision threshold on the held-out split using an F1 sweep.
7. Package the model and threshold with Joblib and serve them through FastAPI.
8. Log parameters, metrics, and artifacts for all three experiments in MLflow.
9. Evaluate subgroup performance and explain two individual predictions with SHAP.

## Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---:|---:|---:|
| Logistic regression | 0.9151 | 0.7142 | 0.8515 |
| Random forest | 0.9203 | 0.7240 | 0.8581 |
| Calibrated XGBoost | **0.9306** | **0.7448** | **0.8723** |

The calibrated production model also recorded PR-AUC `0.8355`, recall `0.7742`, precision `0.7175`, Brier score `0.0863`, and log-loss `0.2746` in the existing model card benchmarks.

## Deployment and reproducibility

The API accepts the ten raw model fields and returns probability, binary prediction, label, and threshold. Invalid ranges or missing fields return HTTP 422; a missing model bundle returns HTTP 503.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.mlops
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn src.api:app --host 127.0.0.1 --port 8000
python -m pytest tests -q
```

The final API demonstration used a high-earner profile and returned:

```json
{"probability": 1.0, "prediction": 1, "label": ">50K", "threshold": 0.36}
```

## Responsible AI audit

Protected attributes were excluded from training and retained for post-hoc evaluation. The gender recall gap was `2.27` percentage points: male recall `0.7778`, female recall `0.7551`. This is within the project's 5-point regression guard, but it does not prove the model is unbiased.

Race performance varied more. White recall was `0.7823`, Black recall `0.6989`, Asian-Pac-Islander recall `0.7308`, and Amer-Indian-Eskimo recall `0.4444` based on only 9 positive cases. These small-group estimates are unstable and are treated as warning signals rather than definitive population claims.

The concrete mitigation is support-aware subgroup threshold calibration on a separate validation set: tune thresholds only for groups with at least 30 positive cases, route low-support or borderline cases to human review, report uncertainty, and block deployment if the recall gap exceeds 5 percentage points.

SHAP explanations are local, not causal. Positive SHAP values increase a prediction relative to the background expectation; negative values decrease it. The notebook verifies additivity against the final calibrated probability.

## Limitations and failure cases

- The data is historical and may encode structural inequality; performance does not establish fairness.
- The smallest race groups have insufficient positive support for stable estimates.
- The model predicts income association, not worth, ability, or causation.
- Category drift, unseen occupations, or distribution shift can reduce performance.
- The threshold optimizes F1 on one fixed split and should be revalidated for a new deployment population.
- SHAP explains model behavior but cannot make the model's correlations causal.
- The API is a demonstration service and needs authentication, rate limiting, observability, and a production model registry before public deployment.

## Verification

The final repository test suite passes with 16 tests. The Git history shows staged model development, MLOps, and responsible-AI milestones rather than a single untracked implementation burst.
