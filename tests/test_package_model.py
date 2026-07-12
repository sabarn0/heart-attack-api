import os
import joblib
import pandas as pd
import pytest


def test_production_model_existence():
    """
    Checks that the serialized pipeline exists in the models/production path.
    """
    pkl_path = "models/production/model.pkl"
    assert os.path.exists(pkl_path), f"Production model not found at {pkl_path}"


def test_production_model_prediction():
    """
    Checks that the production model can run predictions on raw sample input correctly.
    """
    pkl_path = "models/production/model.pkl"
    if not os.path.exists(pkl_path):
        pytest.skip("Model pickle file does not exist yet. Run experiments first.")

    model = joblib.load(pkl_path)

    sample = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        ]
    )

    pred = model.predict(sample)
    proba = model.predict_proba(sample)

    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0][0] <= 1.0
    assert 0.0 <= proba[0][1] <= 1.0
