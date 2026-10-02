# Mid-Sprint Progress Report: Core Development (Gate M2.2)

**Track:** Track M: AI Capstone  
**Checkpoint:** M2.2 Core Development  
**Project:** UCI Adult Income Prediction (>\$50K)  
**Date:** October 2026  

---

## 1. Executive Summary

This mid-sprint progress report documents the transition from the M2.1 feature engineering and data-quality baseline to a production-grade core predictive model. 

Through two distinct experimental iterations, we:
1. Explored non-linear tree ensembles, diagnosing and documenting a critical problem during development: **severe unconstrained tree overfitting and split degradation caused by high-cardinality one-hot encoded features**.
2. Developed and tuned our **Core Model: Regularized Gradient Boosted Decision Trees (XGBoost)**, which achieves **decisive, statistically validated improvements** across every target metric over the M2.1 baseline, confirmed across all 5 folds in stratified cross-validation.
3. Addressed minority-class recall via **decision-threshold optimization** and **class weighting (`scale_pos_weight`)**.
4. Established a modular Python codebase (`src/data.py`, `src/features.py`, `src/models.py`), a comprehensive demonstration notebook (`core_model_development.ipynb`), and an automated test suite (`tests/test_model.py`) passing 100% of integration checks.

---

## 2. What Was Built

### 2.1 Modular Code Architecture
To move beyond monolithic notebooks and support explainable, unprompted line-by-line inspection:
- [`src/data.py`](file:///c:/Users/Misiko/AI_DataScience/src/data.py): Automated raw data ingestion with local caching, data cleaning according to M2.1 audit findings (`?` $\to$ `Missing`, removal of redundant `education` and zero-signal `fnlwgt`), separation of protected attributes (`sex`, `race`) for auditing, and stratified 80/20 train/test splitting.
- [`src/features.py`](file:///c:/Users/Misiko/AI_DataScience/src/features.py): Deterministic feature transformations for `log_capital_gain` ($\log(1 + x)$), career lifecycle binning `age_group`, and `edu_x_hours` interaction.
- [`src/models.py`](file:///c:/Users/Misiko/AI_DataScience/src/models.py): Scikit-learn and XGBoost pipelines integrating `ColumnTransformer`, standard scaling, one-hot encoding (`handle_unknown='ignore'`), and configurable estimators with evaluation utilities.
- [`tests/test_model.py`](file:///c:/Users/Misiko/AI_DataScience/tests/test_model.py): 5 automated tests verifying data contracts, feature transformations, baseline reproduction, core model performance superiority, and threshold tuning.
- [`core_model_development.ipynb`](file:///c:/Users/Misiko/AI_DataScience/core_model_development.ipynb): Fully executed interactive notebook recording all experiment iterations, metric tables, cross-validation outputs, and fairness audits.

---

## 3. Experiment Iterations & Progression

The development proceeded across two distinct iterations:

```
[M2.1 Baseline: Logistic Regression]
       ROC-AUC: 0.9083 | F1: 0.6704 | Acc: 0.8548
                     │
                     ▼
[Iteration 1a: Unconstrained Random Forest]
       Real Problem Hit: Severe Overfitting (Train AUC: 0.9971 vs Test AUC: 0.8971)
       One-hot sparse feature split dilution
                     │
                     ▼
[Iteration 1b: Regularized Random Forest (max_depth=12, leaf=5)]
       Partial Fix: Overfitting closed, Test AUC recovered to 0.9190, but low Recall (56.2%)
                     │
                     ▼
[Iteration 2: Core XGBoost Model (max_depth=5, lr=0.1, subsample=0.8)]
       Decisive Improvement: Test ROC-AUC: 0.9307 (+0.0224) | F1: 0.7194 (+0.0490)
       5-Fold CV ROC-AUC: 0.9270 (Beats baseline in 5 of 5 folds)
                     │
                     ▼
[Iteration 2b: Threshold Optimization (Threshold = 0.35)]
       Minority Recall boosted to 79.66% (+18.5% over baseline) while retaining 0.7262 F1
```

### Iteration Tracking Summary Table

| Model Stage | Model & Configuration | Accuracy | F1 Score | ROC-AUC | Precision | Recall | 5-Fold CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| **M2.1 Baseline** | Logistic Regression (Scaled + OHE) | 0.8548 | 0.6704 | 0.9083 | 0.7410 | 0.6120 | 0.9043 |
| **M2.1 Engineered** | LR + 3 Handcrafted Features | 0.8610 | 0.6857 | 0.9151 | 0.7557 | 0.6276 | 0.9104 |
| **Iteration 1a** | Random Forest (Unconstrained: `max_depth=None`) | 0.8494 | 0.6707 | **0.8971** *(Problem)* | 0.7080 | 0.6371 | — *(Overfit)* |
| **Iteration 1b** | Random Forest (Regularized: `depth=12`, `leaf=5`) | 0.8626 | 0.6632 | 0.9190 | **0.8090** | 0.5619 | 0.9182 |
| **Iteration 2 (Core)** | **XGBoost (`depth=5`, $\eta=0.1$, `sub=0.8`)** | **0.8753** | **0.7194** | **0.9307** | 0.7851 | **0.6639** | **0.9270** |
| **Iteration 2b** | Core XGBoost @ Decision Threshold = 0.35 | 0.8617 | **0.7262** | **0.9307** | 0.6672 | **0.7966** | 0.9270 |
| **Iteration 2c** | XGBoost with `scale_pos_weight = 3.15` | 0.8329 | 0.7177 | **0.9306** | 0.6050 | **0.8820** | 0.9268 |

---

## 4. Real Problem Hit During Development & Resolution

### Problem Diagnosed: Tree Overfitting & One-Hot Feature Dilution
When moving from linear models to an ensemble of decision trees in **Iteration 1a**, the unconstrained Random Forest produced:
- **Training Set:** Accuracy $97.58\%$, F1 $0.9490$, ROC-AUC $0.9971$.
- **Test Set:** Accuracy $84.94\%$, F1 $0.6707$, ROC-AUC **$0.8971$**.

The test ROC-AUC dropped **below the simple linear baseline ($0.9083$)** despite near-perfect training accuracy.

#### Root Causes:
1. **Unconstrained Tree Depth:** The default `max_depth=None` allowed individual decision trees to grow until leaf purity, memorizing unique combinations of noise (e.g. rare combinations of age, capital flow, and occupation).
2. **One-Hot Encoding Split Dilution:** One-hot encoding categorical variables (e.g., `native_country` with 42 categories, `occupation` with 15) inflated feature dimensionality to 90+ columns. In Random Forest, random subspace feature sampling ($\sqrt{p} \approx 9$ features per split) frequently sampled uninformative sparse dummy columns, causing shallow splits on unhelpful flags and fracturing the feature space.
3. **Class Imbalance Penalty:** With only 24% positive samples, unweighted Gini impurity splitting biased tree splits towards the majority class ($<=50\text{K}$).

### The Fix: Depth-Controlled Gradient Boosted Decision Trees
In **Iteration 2**, we re-architected the ensemble using regularized gradient boosting (XGBoost):
1. **Constrained Maximum Depth (`max_depth=5`):** Restricts the maximum interaction order per tree to 5 levels, preventing memorization of leaf noise.
2. **Shrinkage / Learning Rate ($\eta = 0.1$):** Rather than averaging deep trees, trees are sequentially fit to pseudo-residuals with conservative step sizes.
3. **Dual Subsampling (`subsample=0.8`, `colsample_bytree=0.8`):** Prevents dominant features from masking subtle signals while reducing tree variance.
4. **Result of Fix:**
   - Overfitting eliminated: Training ROC-AUC is $0.9399$ while Test ROC-AUC reaches **$0.9307$** (a narrow, healthy generalization margin).
   - Test ROC-AUC improves by **$+0.0224$** over the M2.1 baseline and **$+0.0336$** over the failed Random Forest.

---

## 5. What Improved Over the Baseline

1. **Overall Discrimination (ROC-AUC):**
   - M2.1 Baseline: $0.9083 \to$ Core XGBoost: **$0.9307$** (**$+0.0224$** absolute gain).
   - 5-Fold Cross-Validation: Baseline Mean $0.9104 \to$ Core XGBoost Mean **$0.9270$** (**$+0.0166$** gain). XGBoost outperformed the linear model in **5 out of 5 folds** ($p < 0.05$).
2. **Classification Quality (F1 Score & Accuracy):**
   - F1 Score increased from $0.6704$ to **$0.7194$** (**$+0.0490$**).
   - Accuracy increased from $85.48\%$ to **$87.53\%$** (**$+2.05\%$**).
   - Calibration (Brier Score Loss): improved from $0.1042$ to **$0.0908$** (lower is better, indicating sharper and more accurate probability estimates).
3. **Controllable Minority Recall:**
   - In M2.1, recall for $>50\text{K}$ individuals was capped at $61.2\%$ (or $62.8\%$ with feature engineering).
   - With threshold tuning ($\tau = 0.35$), recall reached **$79.66\%$** with an F1 score of **$0.7262$**. With `scale_pos_weight = 3.15`, recall reached **$88.20\%$**.

---

## 6. What's Still Broken & Remaining Challenges

1. **Persistent Group Disparities (Fairness Deficit):**
   - Auditing the Core XGBoost predictions reveals that withholding `sex` and `race` does not resolve outcome disparities:
     - **Sex:** Female test recall is $61.8\%$ vs Male test recall $67.5\%$ (a 5.7 percentage-point gap). Female predicted positive rate is $8.9\%$ vs actual $11.4\%$.
     - **Race:** Black test recall is $56.9\%$ vs White test recall $67.2\%$ (a 10.3 percentage-point gap).
     - *Cause:* Proxies like `relationship` (Husband/Wife) and historical income disparity remain encoded in occupational and capital features.
2. **Probability Extremes Calibration:**
   - While the Brier score improved to $0.0908$, gradient boosted trees can produce slightly uncalibrated probabilities in the tail regions ($p \in [0.0, 0.1]$ and $p \in [0.9, 1.0]$). Post-hoc isotonic or Platt calibration should be evaluated in M2.3 if exact probability values are needed for decision-making.
3. **Inference Latency & Deployment Footprint:**
   - Logistic regression evaluates in $< 1 \text{ ms}$ with minimal memory. The XGBoost model comprises 150 trees. While fast on modern CPUs, deployment memory constraints must be profiled.

---

## 7. Plan for Next Gate (M2.3 / Final Evaluation)

1. **Fairness Mitigation:** Implement group-specific threshold calibration or fair reweighing to close the recall gap between demographic subgroups.
2. **Probability Calibration:** Fit a `CalibratedClassifierCV` (isotonic regression) on top of the Core XGBoost model.
3. **Deployment Artifact:** Export the end-to-end pipeline via `joblib`/`ONNX` with an automated validation check for production readiness.
