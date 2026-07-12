from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from src.heart_disease.features import build_preprocessor
from src.heart_disease.config import RANDOM_STATE, CV_FOLDS


def get_lr_search():
    """
    Constructs GridSearchCV for Logistic Regression.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]}
    search = GridSearchCV(
        pipeline, param_grid=param_grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1
    )
    return search


def get_xgb_search():
    """
    Constructs RandomizedSearchCV for XGBoost.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        ]
    )
    param_distributions = {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__n_estimators": [50, 100, 150, 200],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return search
