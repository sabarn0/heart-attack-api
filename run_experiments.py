"""
Main entrypoint to run the end-to-end training, tracking, selection, and packaging pipeline.

### Model Selection & Tuning Narrative
- **Logistic Regression**: Serves as our linear, highly interpretable baseline. Using GridSearchCV, we tuned `C` and `penalty` (L1 vs L2) with a `liblinear` solver. L2 regularization helps mitigate collinearity between variables, while L1 regularization can perform feature selection by forcing coefficients of noisy/weak predictors (like `chol` and `fbs`) to zero.
- **XGBoost**: Serves as our non-linear tree ensemble baseline, which is robust to outliers and handles complex feature interactions (such as age vs. max heart rate) automatically. Using RandomizedSearchCV, we tuned tree depth, learning rate, and subsampling to prevent overfitting on this relatively small dataset.
"""

import os
import shutil
import pandas as pd
from src.heart_disease.data import load_clean_data, get_train_test_split
from src.heart_disease.train import get_lr_search, get_xgb_search
from src.heart_disease.evaluate import compute_cv_metrics, evaluate_test_set, save_plots
from src.heart_disease.experiment import log_experiment_run, init_experiment
from src.heart_disease.compare_and_select import compare_runs_and_select_best
from src.heart_disease.package_model import package_winning_model

def main():
    print("Starting MLOps pipeline for Heart Disease Classification...")
    
    # 1. Load data
    df = load_clean_data()
    row_count = len(df)
    print(f"Loaded clean dataset. Total rows: {row_count}")
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"Data split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")
    
    # Temporary directory for plots
    temp_plots_dir = os.path.abspath("temp_plots")
    os.makedirs(temp_plots_dir, exist_ok=True)
    
    # Initialize MLflow experiment
    init_experiment()
    
    # Dataset version tag based on row count and timestamp
    dataset_version = f"clean_v1_{row_count}_rows"
    
    # ------------------
    # RUN 1: Logistic Regression
    # ------------------
    print("\nTuning Logistic Regression model...")
    lr_search = get_lr_search()
    lr_search.fit(X_train, y_train)
    
    lr_best_pipeline = lr_search.best_estimator_
    lr_best_params = lr_search.best_params_
    print(f"Logistic Regression Best Params: {lr_best_params}")
    
    # Compute metrics
    cv_metrics_lr = compute_cv_metrics(lr_best_pipeline, X_train, y_train)
    test_metrics_lr, y_pred_lr, y_proba_lr = evaluate_test_set(lr_best_pipeline, X_test, y_test)
    
    # Generate plots
    cm_path_lr, roc_path_lr = save_plots(y_test, y_pred_lr, y_proba_lr, "Logistic Regression", temp_plots_dir)
    
    # Combine metrics
    all_metrics_lr = {**cv_metrics_lr, **test_metrics_lr}
    
    # Log to MLflow
    log_experiment_run(
        run_name="Logistic_Regression_Baseline",
        model_type="logistic_regression",
        best_params=lr_best_params,
        metrics=all_metrics_lr,
        pipeline=lr_best_pipeline,
        cm_path=cm_path_lr,
        roc_path=roc_path_lr,
        dataset_version=dataset_version
    )
    
    # ------------------
    # RUN 2: XGBoost
    # ------------------
    print("\nTuning XGBoost model...")
    xgb_search = get_xgb_search()
    xgb_search.fit(X_train, y_train)
    
    xgb_best_pipeline = xgb_search.best_estimator_
    xgb_best_params = xgb_search.best_params_
    print(f"XGBoost Best Params: {xgb_best_params}")
    
    # Compute metrics
    cv_metrics_xgb = compute_cv_metrics(xgb_best_pipeline, X_train, y_train)
    test_metrics_xgb, y_pred_xgb, y_proba_xgb = evaluate_test_set(xgb_best_pipeline, X_test, y_test)
    
    # Generate plots
    cm_path_xgb, roc_path_xgb = save_plots(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", temp_plots_dir)
    
    # Combine metrics
    all_metrics_xgb = {**cv_metrics_xgb, **test_metrics_xgb}
    
    # Log to MLflow
    log_experiment_run(
        run_name="XGBoost_Ensemble",
        model_type="xgboost",
        best_params=xgb_best_params,
        metrics=all_metrics_xgb,
        pipeline=xgb_best_pipeline,
        cm_path=cm_path_xgb,
        roc_path=roc_path_xgb,
        dataset_version=dataset_version
    )
    
    # ------------------
    # PROGRAMMATIC SELECTION & PACKAGING
    # ------------------
    print("\nComparing runs and selecting best model...")
    best_run = compare_runs_and_select_best()
    
    print("\nPackaging best model for serving...")
    package_winning_model(best_run, row_count)
    
    # Clean up temp plots directory
    if os.path.exists(temp_plots_dir):
        shutil.rmtree(temp_plots_dir)
        
    print("\nMLOps end-to-end pipeline finished successfully!")

if __name__ == "__main__":
    main()
