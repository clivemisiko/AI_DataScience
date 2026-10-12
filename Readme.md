# AI Capstone — UCI Adult Income Prediction

Predicting whether an individual earns more than $50K a year from 1994 US census data, combining rigorous data-quality auditing, iterative feature engineering, non-linear tree ensembles, post-hoc probability calibration, and cost-sensitive threshold optimization.

## Dataset
[UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult): 32,561 rows, 15 columns.

---

## Milestone Trajectory & Progression

### 1. M2.1 Baseline & Feature Engineering
- Audited hidden `"?"` missingness, class imbalance ($24.1\%$ positive), and sampling skews on protected attributes (`sex`, `race`).
- Dropped redundant `education` and uninformative `fnlwgt`; isolated protected attributes for auditing only.
- Implemented baseline Logistic Regression and 3 hypothesis-driven features (`log_capital_gain`, `age_group`, `edu_x_hours`).
- Reached **ROC-AUC 0.9151** and **F1 0.6857** with Logistic Regression.

### 2. M2.2 Core Development Checkpoint
- **Iteration 1 (Random Forest & Overfitting Hit):** Unconstrained tree depth caused severe memorization (Train ROC-AUC 0.9971 vs Test ROC-AUC 0.8971 — worse than baseline) due to 90+ sparse one-hot dummy columns diluting subspace splits.
- **Iteration 2 (Core Build - XGBoost):** Re-architected with regularized gradient boosting (`max_depth=5`, $\eta=0.1$, `subsample=0.8`).
- Decisively outperformed baseline across all metrics, confirmed in **5 of 5 folds** in stratified cross-validation (mean CV ROC-AUC: $0.9270$ vs $0.9104$).

### 3. M2.3 Checkpoint: Integrate & Iterate (Evaluation & Calibration)
- **Proper Evaluation Metrics:** Added Brier Score Loss, Log-Loss, PR-AUC, and Equal Opportunity group recall auditing.
- **Probability Calibration (Ablation 1):** Integrated `CalibratedClassifierCV` (5-fold Isotonic Regression). Improved Brier Score to **0.08627** and boosted default F1 to **0.7282**.
- **Threshold Optimization (Ablation 2):** Shifted operating cutoff from arbitrary $0.50$ to F1-optimal **$\tau = 0.35$**. Lifted positive recall from $66.4\%$ to **$78.2\%$** and peaked F1 at **$0.7446$**.
- **Fairness Impact:** Narrowed female vs male recall gap from $2.8\%$ to **$1.3\%$** and boosted Black recall from $61.3\%$ to **$71.0\%$**.

---

## Benchmark Progression Table

| Checkpoint / Model State | Accuracy | F1 Score | ROC-AUC | PR-AUC | Recall | Precision | Brier Score *(lower=better)* |
|---|---|---|---|---|---|---|---|
| **M2.1 Raw Baseline (LR)** | 0.8548 | 0.6704 | 0.9083 | 0.7712 | 0.6120 | 0.7410 | 0.10420 |
| **M2.1 Engineered Baseline (LR)** | 0.8610 | 0.6857 | 0.9151 | 0.7850 | 0.6276 | 0.7557 | 0.09910 |
| **M2.2 Core Build (XGBoost @ 0.50)** | 0.8753 | 0.7194 | **0.9307** | 0.8354 | 0.6639 | **0.7851** | 0.08646 |
| **M2.3 Calibrated Build (Isotonic @ 0.50)** | **0.8770** | 0.7282 | 0.9306 | **0.8355** | 0.6843 | 0.7781 | **0.08627** |
| **M2.3 Tuned Build (Isotonic @ 0.35)** | 0.8709 | **0.7446** | 0.9306 | **0.8355** | **0.7819** | 0.7107 | **0.08627** |
| **Total Gain over Baseline** | **+0.0161** | **+0.0742** | **+0.0223** | **+0.0643** | **+0.1699** | -0.0303 | **-0.01793** |

---

## Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── data.py                 # Data loading, cleaning, stratified split
│   ├── features.py             # Feature engineering transformations
│   └── models.py               # Model pipelines (LR, RF, XGBoost, Calibration, Thresholding)
├── tests/
│   └── test_model.py           # Automated test suite (7/7 tests passing)
├── core_model_development.ipynb              # M2.2 Core modeling experiments
├── feature_engineering_and_data_quality.ipynb# M2.1 Baseline notebook
├── mid_sprint_progress_report.md             # M2.2 Progress report
├── evaluation_and_ablation_report.md         # M2.3 Evaluation & ablation report
├── pytest.ini                                # Test runner configuration
├── requirements.txt                          # Dependencies
└── Readme.md                                 # Full documentation & benchmarks
```

---

## How to Run

### 1. Run Automated Test Suite
```powershell
pytest tests/ -v
```
All 7 tests validate data integrity, feature creation, baseline replication, core XGBoost superiority, calibration pipeline stability, and optimal threshold search.

### 2. Inspect Ablation & Evaluation Report
Read [`evaluation_and_ablation_report.md`](evaluation_and_ablation_report.md) for full before/after ablation metrics, calibration curves, and demographic fairness analysis.
