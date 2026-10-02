# AI Capstone — UCI Adult Income Prediction

Predicting whether an individual earns more than $50K a year from 1994 US census data, combining rigorous data-quality auditing, iterative feature engineering, and non-linear tree-based ensembles.

## Dataset
[UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult): 32,561 rows, 15 columns.

---

## Sprint Progression & Milestones

### M2.1 Baseline & Feature Engineering
- Audited missingness (`?`), class imbalance (24.1% >50K), and protected attribute sampling skew.
- Discarded redundant `education` and zero-signal `fnlwgt`. Separated `sex` and `race` for fairness audits.
- Implemented baseline Logistic Regression and 3 hypothesis-driven features (`log_capital_gain`, `age_group`, `edu_x_hours`).
- Reached **ROC-AUC 0.9151** and **F1 0.6857** with Logistic Regression.

### M2.2 Core Development Checkpoint
- **Iteration 1 (Random Forest & Real Problem Diagnosed):** Unconstrained tree depth (`max_depth=None`) led to severe memorization (Train ROC-AUC 0.9971 vs Test ROC-AUC 0.8971 — worse than baseline) and split dilution over 90+ sparse one-hot encoded categories.
- **Iteration 2 (Core Model - XGBoost):** Re-architected with regularized gradient boosting (shallow depth `max_depth=5`, learning rate $\eta=0.1$, row/col subsampling 0.8).
- **Measurable Improvement:** Decisively beats the baseline across all metrics, verified in **5 of 5 folds** in stratified cross-validation.
- **Imbalance & Threshold Tuning:** Adjusted classification threshold ($\tau=0.35$) and `scale_pos_weight` to lift minority class recall from 61.2% to **79.7% / 88.2%**.

---

## Model Performance Benchmarks

| Milestone / Model | Accuracy | F1 Score | ROC-AUC | Precision | Recall | 5-Fold CV ROC-AUC |
|---|---|---|---|---|---|---|
| **M2.1 Raw Baseline (LR)** | 0.8548 | 0.6704 | 0.9083 | 0.7410 | 0.6120 | 0.9043 |
| **M2.1 Engineered Baseline (LR)** | 0.8610 | 0.6857 | 0.9151 | 0.7557 | 0.6276 | 0.9104 |
| **Iteration 1a (RF Unconstrained)** | 0.8494 | 0.6707 | 0.8971 *(Overfit)* | 0.7080 | 0.6371 | — |
| **Iteration 1b (RF Regularized)** | 0.8626 | 0.6632 | 0.9190 | 0.8090 | 0.5619 | 0.9182 |
| **Iteration 2 (Core XGBoost)** | **0.8753** | **0.7194** | **0.9307** | **0.7851** | **0.6639** | **0.9270** |
| **Iteration 2b (XGBoost @ Threshold 0.35)** | 0.8617 | **0.7262** | **0.9307** | 0.6672 | **0.7966** | 0.9270 |
| **Change over M2.1 Baseline** | **+0.0205** | **+0.0490** | **+0.0224** | **+0.0441** | **+0.0519** | **+0.0166** |

---

## Project Structure
```text
├── src/
│   ├── __init__.py
│   ├── data.py                 # Data loading, cleaning, stratified split
│   ├── features.py             # Feature engineering transformations
│   └── models.py               # Model pipelines (LR, RF, XGBoost) & evaluation
├── tests/
│   └── test_model.py           # Automated pytest suite (contract, baseline, core model)
├── feature_engineering_and_data_quality.ipynb   # M2.1 Baseline notebook
├── core_model_development.ipynb                 # M2.2 Core Development notebook
├── mid_sprint_progress_report.md                # M2.2 Checkpoint progress report
├── pytest.ini                  # Pytest configuration
├── adult.csv                   # Cached dataset (git-ignored)
└── Readme.md                   # Project overview & documentation
```

---

## How to Run

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # or: pip install pandas numpy scikit-learn xgboost pytest jupyter
```

### 2. Run Automated Tests
```powershell
pytest tests/ -v
```

### 3. Run Core Development Notebook
Open and run all cells in `core_model_development.ipynb`:
```powershell
jupyter lab core_model_development.ipynb
```
