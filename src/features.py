"""Feature engineering transformations for the UCI Adult Income dataset."""

import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applies three validated engineered features:
    1. log_capital_gain: log1p compression of heavily skewed capital_gain.
    2. age_group: binned age to capture non-linear career-earnings lifecycle peak (45-54).
    3. edu_x_hours: interaction term between education level and hours worked per week.
    """
    X_out = X.copy()
    X_out["log_capital_gain"] = np.log1p(X_out["capital_gain"])
    X_out["age_group"] = pd.cut(
        X_out["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        right=False,
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)
    X_out["edu_x_hours"] = X_out["education_num"] * X_out["hours_per_week"]
    return X_out

