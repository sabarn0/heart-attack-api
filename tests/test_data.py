import pandas as pd
from src.heart_disease.data import load_clean_data
from src.heart_disease.config import TARGET_COL


def test_clean_data_exists():
    """
    Verify the cleaned dataset file exists or can be generated.
    """
    df = load_clean_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_data_shape_and_columns():
    """
    Verify shape and expected column names of clean dataset.
    """
    df = load_clean_data()
    expected_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' missing from clean dataset"

    # Dataset should contain ~303 instances (original size)
    assert len(df) >= 290
    assert len(df) <= 310


def test_binary_target():
    """
    Verify target variable is binary (0 and 1 only).
    """
    df = load_clean_data()
    unique_targets = set(df[TARGET_COL].unique())
    assert unique_targets == {0, 1}, f"Target values must be exactly {{0, 1}}, got {unique_targets}"


def test_no_missing_values():
    """
    Verify there are no missing values remaining in the clean CSV.
    """
    df = load_clean_data()
    total_missing = df.isnull().sum().sum()
    assert total_missing == 0, f"Cleaned dataset contains {total_missing} missing values"
