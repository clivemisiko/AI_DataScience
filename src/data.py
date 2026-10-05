"""Data loading and preprocessing utilities for the UCI Adult Income dataset."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]


def load_raw_data(data_path="adult.csv"):
    """Load raw dataset from local cache if available, else download from UCI archive."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(URL, header=None, names=COLUMNS, skipinitialspace=True)
        df.to_csv(data_path, index=False)
    return df


def prepare_data(df):
    """
    Clean data according to M2.1 audit findings:
    - Replace '?' with explicit 'Missing' category.
    - Drop 'education' (redundant with education_num) and 'fnlwgt' (sampling weight, zero signal).
    - Separate protected attributes ('sex', 'race') for auditing only.
    - Convert income to binary target (1 if '>50K', 0 otherwise).
    """
    clean = df.replace("?", "Missing")
    clean = clean.drop(columns=["education", "fnlwgt"])

    y = (clean["income"] == ">50K").astype(int)
    protected = clean[["sex", "race"]]
    X = clean.drop(columns=["income", "sex", "race"])

    return X, y, protected


def get_stratified_split(X, y, protected, test_size=0.2, random_state=42):
    """Perform stratified 80/20 train/test split on target y."""
    return train_test_split(
        X, y, protected, test_size=test_size, stratify=y, random_state=random_state
    )

