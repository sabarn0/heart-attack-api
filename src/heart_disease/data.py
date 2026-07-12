import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.heart_disease.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, TARGET_COL


def download_and_clean_data(output_path: str = DATA_PATH):
    """
    Downloads raw Heart Disease dataset from UCI repo, applies cleaning steps
    similar to those in the EDA notebook, and saves the cleaned dataset.
    """
    from ucimlrepo import fetch_ucirepo

    # Fetch dataset (id=45 -> Heart Disease)
    heart_disease = fetch_ucirepo(id=45)

    # Extract features and targets
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine them
    df = pd.concat([X, y], axis=1)
    raw_target_col = y.columns[0]  # 'num'

    # Create binary target
    df["target"] = (df[raw_target_col] > 0).astype(int)
    df_clean = df.drop(columns=[raw_target_col]).copy()

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # Impute missing values (median for numeric, mode for categorical)
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Cast categoricals to integer
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(int)

    # Add back the binary target
    df_clean["target"] = df["target"]

    # Save cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned dataset downloaded and saved to: {output_path}")


def load_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the cleaned CSV. If it does not exist, triggers download and clean.
    """
    if not os.path.exists(path):
        print(f"Cleaned data not found at {path}. Re-running acquisition & cleaning...")
        download_and_clean_data(path)
    return pd.read_csv(path)


def get_train_test_split(df: pd.DataFrame):
    """
    Performs stratified train-test split.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
