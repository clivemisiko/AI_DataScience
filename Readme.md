# Feature Engineering & Data Quality — UCI Adult Income

Predicting whether a person earns more than $50K a year from 1994 US census data,
with a focus on auditing data quality before modelling and testing whether
hand-built features improve a simple baseline.

## Dataset
[UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult): 32,561 rows, 15 columns.
The notebook downloads it directly from the UCI archive, so no local data file is needed.

## What the notebook does
1. **Data-quality audit.** Checks for missing values, class imbalance, skew in
   protected attributes, and leaky, redundant or proxy columns.
2. **Preparation.** Replaces `?` with a `Missing` category and drops `education` and `fnlwgt`.
   `sex` and `race` are kept out of the model and used only for auditing.
   The train/test split is stratified 80/20.
3. **Baseline model.** Logistic regression inside a scikit-learn Pipeline
   (scaling and one-hot encoding are fitted on the training data only).
4. **Feature engineering.** Three features, each stated as a hypothesis first and then checked:
   - `log_capital_gain`: compresses a heavily skewed column
   - `age_group`: captures the inverted-U relationship between age and income
   - `edu_x_hours`: the interaction between education and hours worked
5. **Before/after comparison**, confirmed with 5-fold cross-validation.
6. **Per-group check** of recall and prediction rates by sex and race.

## Results
| | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline | 0.8548 | 0.6704 | 0.9083 |
| + 3 engineered features | 0.8610 | 0.6857 | 0.9151 |
| Change | +0.0063 | +0.0153 | +0.0068 |

5-fold CV ROC-AUC went from 0.9043 to 0.9104, and the engineered model was better in all 5 folds.

## Key findings
1. Missing values are hidden as `"?"` and are not random: rows with gaps earn >50K
   at 13.9% vs 24.9% for complete rows.
2. Classes are imbalanced (24% >50K), so accuracy alone is misleading.
3. The sample is skewed on sex (67% male) and race (85% White), and some race groups are very small.
4. `education` duplicates `education_num`, `fnlwgt` carries no signal,
   and `relationship` is a near-perfect proxy for `sex`.
5. Leaving out `sex` and `race` did **not** remove disparities. Recall is 57.6% for
   women vs 63.9% for men, and 53.8% for Black vs 63.8% for White individuals.
   This model shouldn't be used for decisions about individuals without a fairness mitigation step.

## How to run
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn jupyter
Then open `feature_engineering_and_data_quality.ipynb` and run all cells
(an internet connection is needed to download the data).
