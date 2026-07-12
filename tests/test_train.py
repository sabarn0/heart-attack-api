import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.heart_disease.train import get_lr_search, get_xgb_search


def test_model_training_pipeline_and_probabilities():
    """
    Verifies that hyperparameter tuning constructs a valid pipeline, trains
    properly on synthetic data, and outputs probabilities summing to 1.
    """
    # Create small synthetic dataframe
    np.random.seed(42)
    rows = 40
    dummy_df = pd.DataFrame(
        {
            "age": np.random.randint(30, 80, size=rows),
            "sex": np.random.choice([0, 1], size=rows),
            "cp": np.random.choice([1, 2, 3, 4], size=rows),
            "trestbps": np.random.randint(100, 180, size=rows),
            "chol": np.random.randint(150, 350, size=rows),
            "fbs": np.random.choice([0, 1], size=rows),
            "restecg": np.random.choice([0, 1, 2], size=rows),
            "thalach": np.random.randint(100, 200, size=rows),
            "exang": np.random.choice([0, 1], size=rows),
            "oldpeak": np.random.uniform(0.0, 5.0, size=rows),
            "slope": np.random.choice([0, 1, 2], size=rows),
            "ca": np.random.choice([0, 1, 2, 3], size=rows),
            "thal": np.random.choice([1, 2, 3], size=rows),
        }
    )
    dummy_y = np.random.choice([0, 1], size=rows)

    # Check Logistic Regression Search
    lr_search = get_lr_search()
    lr_search.fit(dummy_df, dummy_y)

    best_lr = lr_search.best_estimator_
    assert isinstance(best_lr, Pipeline)

    # Check predictions
    pred_probs = best_lr.predict_proba(dummy_df)
    assert pred_probs.shape == (rows, 2)
    assert np.allclose(pred_probs.sum(axis=1), 1.0)

    # Check XGBoost Search
    xgb_search = get_xgb_search()
    xgb_search.fit(dummy_df, dummy_y)

    best_xgb = xgb_search.best_estimator_
    assert isinstance(best_xgb, Pipeline)

    # Check predictions
    pred_probs_xgb = best_xgb.predict_proba(dummy_df)
    assert pred_probs_xgb.shape == (rows, 2)
    assert np.allclose(pred_probs_xgb.sum(axis=1), 1.0)
