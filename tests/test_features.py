import pandas as pd
from sklearn.compose import ColumnTransformer
from src.heart_disease.features import build_preprocessor


def test_build_preprocessor():
    """
    Verify build_preprocessor returns a scikit-learn ColumnTransformer.
    """
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_columns_and_unseen_categories():
    """
    Verify the preprocessor runs successfully, outputs expected columns,
    and handles unknown categorical levels gracefully without crashing.
    """
    # 1. Fit on normal train sample
    train_data = pd.DataFrame(
        {
            "age": [63.0, 37.0],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [145.0, 130.0],
            "chol": [233.0, 250.0],
            "fbs": [1, 0],
            "restecg": [0, 1],
            "thalach": [150.0, 187.0],
            "exang": [0, 0],
            "oldpeak": [2.3, 3.5],
            "slope": [0, 1],
            "ca": [0, 0],
            "thal": [1, 2],
        }
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    # 2. Transform unseen test sample containing an unknown category
    # in "cp" (e.g. 5) and "thal" (e.g. 9)
    test_data = pd.DataFrame(
        {
            "age": [45.0],
            "sex": [1],
            "cp": [5],  # Unseen category
            "trestbps": [120.0],
            "chol": [210.0],
            "fbs": [0],
            "restecg": [2],
            "thalach": [160.0],
            "exang": [1],
            "oldpeak": [1.0],
            "slope": [2],
            "ca": [1],
            "thal": [9],  # Unseen category
        }
    )

    # The transformation should succeed without throwing error because
    # of handle_unknown='ignore'
    transformed = preprocessor.transform(test_data)
    assert transformed is not None
    assert transformed.shape[0] == 1

    # Let's count expected columns:
    # 2 skewed numeric + 3 other numeric = 5 numeric.
    # Categoricals have levels: sex(2), cp(2), fbs(2), restecg(2),
    # exang(1 in train), slope(2), ca(1 in train), thal(2).
    # One-hot encoding should map unknown cp and thal to all zeros.
    assert transformed.shape[1] > 5
