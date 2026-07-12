from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import numpy as np
from src.heart_disease.config import SKEWED_COLS, OTHER_NUMERIC_COLS, CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a single ColumnTransformer that preprocesses skewed numeric features,
    other numeric features, and categorical features.
    """
    # 1. Skewed Numeric: Impute (median) -> log1p -> scale
    skewed_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
        ]
    )

    # 2. Other Numeric: Impute (median) -> scale
    other_numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # 3. Categorical: Impute (most_frequent) -> OneHotEncoder
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("skewed_num", skewed_pipeline, SKEWED_COLS),
            ("other_num", other_numeric_pipeline, OTHER_NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
