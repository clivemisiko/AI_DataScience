# Track L2 Fairness and Explainability Audit

## Scope and method

This audit evaluated the calibrated XGBoost income model from the capstone. The train/test split is stratified by target with `random_state=42`. `sex` and `race` are excluded from training features by the existing data-preparation contract and retained only for post-hoc subgroup analysis.

SHAP permutation explanations were generated against the **full model prediction function**, including feature engineering, preprocessing, XGBoost, and isotonic calibration. Two test-set individuals were explained: the highest-probability and lowest-probability records. Categorical features were numerically encoded only inside the SHAP adapter and decoded before prediction. The notebook checks SHAP additivity, so the baseline plus contributions reconstructs each model probability.

Notebook: [l2_fairness_explainability.ipynb](l2_fairness_explainability.ipynb)

## What was checked

- Accuracy, precision, recall, and F1 by `sex`.
- Accuracy, precision, recall, and F1 by `race`.
- Support and number of positive cases for every subgroup.
- Two individual SHAP explanations with local contribution tables and waterfall plots.
- SHAP additivity against the final predicted probability.

## Findings

Metrics use the production threshold `0.36` on the 6,513-row held-out test set:

| Attribute | Group | Support | Positive cases | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| sex | Male | 4,355 | 1,323 | 0.8383 | 0.7151 | 0.7778 | 0.7451 |
| sex | Female | 2,158 | 245 | 0.9407 | 0.7312 | 0.7551 | 0.7430 |
| race | White | 5,533 | 1,410 | 0.8690 | 0.7252 | 0.7823 | 0.7526 |
| race | Black | 662 | 93 | 0.9109 | 0.6771 | 0.6989 | 0.6878 |
| race | Asian-Pac-Islander | 200 | 52 | 0.8200 | 0.6333 | 0.7308 | 0.6786 |
| race | Amer-Indian-Eskimo | 73 | 9 | 0.8493 | 0.4000 | 0.4444 | 0.4211 |
| race | Other | 45 | 4 | 0.9778 | 0.8000 | 1.0000 | 0.8889 |

The gender recall gap is `0.7778 - 0.7551 = 0.0227`, within the existing 5-point regression guard. This is not evidence that the model is unbiased: race recall varies more, and the smallest groups have too few positive cases for stable estimates. The Amer-Indian-Eskimo estimate is particularly uncertain because it is based on only 9 positive cases.

SHAP explanations are interpreted locally. Positive values raise the model probability relative to the background expectation; negative values lower it. They are not causal explanations and do not establish that a feature caused income.

## Concrete mitigation

Before production use, add subgroup threshold calibration on a separate validation set. For each subgroup with a minimum of 30 positive cases, choose the lowest threshold that meets a documented equal-opportunity recall target; do not tune a threshold for groups below that support. For under-supported groups, route borderline predictions to human review and report confidence intervals rather than claiming a precise fairness estimate.

Re-run the subgroup audit and SHAP spot checks for every model release. Block deployment if the absolute recall gap exceeds 5 percentage points, if a subgroup's minimum-support rule is violated without the human-review fallback, or if SHAP shows a new reliance on a sensitive proxy that has not been reviewed.

This mitigation is actionable: it specifies the validation data, threshold rule, support floor, fallback behavior, monitoring cadence, and deployment block conditions.
