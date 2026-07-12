import os
import mlflow
import pandas as pd
from src.heart_disease.config import MLRUNS_DIR, MODELS_DIR


def compare_runs_and_select_best():
    """
    Queries MLflow, compares models, and selects the best run.

    Rationale:
    With ~300 total rows and a 20% test split (only ~60 rows in test set),
    single-split metrics (like test-set ROC-AUC) have high variance. Therefore,
    selection is driven by 5-fold cross-validated ROC-AUC (cv_roc_auc_mean).
    To account for statistical insignificance of tiny differences, an epsilon-based
    tie rule of 0.01 is applied: if the difference in cv_roc_auc_mean is less than 0.01,
    the tie is broken using the test-set F1-score.
    """
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    experiment = mlflow.get_experiment_by_name("heart-disease-classification")
    if experiment is None:
        raise ValueError("Experiment 'heart-disease-classification' not found. Run training first.")

    df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if df_runs.empty:
        raise ValueError("No logged runs found in experiment.")

    # Sort runs by metrics.cv_roc_auc_mean descending
    df_runs = df_runs.sort_values(by="metrics.cv_roc_auc_mean", ascending=False)

    # Check for tie/epsilon rule
    if len(df_runs) >= 2:
        top_run = df_runs.iloc[0]
        second_run = df_runs.iloc[1]

        cv_auc_top = top_run.get("metrics.cv_roc_auc_mean", 0.0)
        cv_auc_second = second_run.get("metrics.cv_roc_auc_mean", 0.0)

        if abs(cv_auc_top - cv_auc_second) < 0.01:
            f1_top = top_run.get("metrics.test_f1", 0.0)
            f1_second = second_run.get("metrics.test_f1", 0.0)

            print(
                "\n[WARNING] The top two models are statistically close "
                "(CV ROC-AUC difference < 0.01)."
            )
            print(
                f"Top model ({top_run.get('tags.mlflow.runName')}): "
                f"CV ROC-AUC = {cv_auc_top:.4f}, Test F1 = {f1_top:.4f}"
            )
            print(
                f"Second model ({second_run.get('tags.mlflow.runName')}): "
                f"CV ROC-AUC = {cv_auc_second:.4f}, Test F1 = {f1_second:.4f}"
            )

            if f1_second > f1_top:
                run_name = second_run.get("tags.mlflow.runName")
                print(
                    f"Applying tiebreaker: Selecting '{run_name}' "
                    "based on higher Test F1-score.\n"
                )
                best_run = second_run
            else:
                run_name = top_run.get("tags.mlflow.runName")
                print(
                    f"Applying tiebreaker: Selecting '{run_name}' "
                    "based on equal or higher Test F1-score.\n"
                )
                best_run = top_run
        else:
            best_run = top_run
    else:
        best_run = df_runs.iloc[0]

    comparison_data = []
    for _, row in df_runs.iterrows():
        comparison_data.append(
            {
                "Run Name": row.get("tags.mlflow.runName", "Unnamed"),
                "Model Type": row.get("tags.model_type", "Unknown"),
                "CV ROC-AUC (Mean)": f"{row.get('metrics.cv_roc_auc_mean', 0.0):.4f}",
                "Test ROC-AUC": f"{row.get('metrics.test_roc_auc', 0.0):.4f}",
                "Accuracy": f"{row.get('metrics.test_accuracy', 0.0):.4f}",
                "Precision": f"{row.get('metrics.test_precision', 0.0):.4f}",
                "Recall": f"{row.get('metrics.test_recall', 0.0):.4f}",
                "F1-Score": f"{row.get('metrics.test_f1', 0.0):.4f}",
            }
        )
    df_comparison = pd.DataFrame(comparison_data)

    # Print to stdout
    print("\n=== Model Comparison Table ===")
    print(df_comparison.to_string(index=False))
    print("==============================\n")

    # Write comparison markdown file
    os.makedirs(MODELS_DIR, exist_ok=True)
    comp_md_path = os.path.join(MODELS_DIR, "comparison.md")

    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write(
            "Below is the performance comparison showing both CV and test metrics side-by-side:\n\n"
        )
        f.write(df_comparison.to_markdown(index=False))
        f.write(
            "\n\nBest model selected is **{}** based on CV ROC-AUC "
            "(with F1-score tiebreaker if difference < 0.01).\n".format(
                best_run.get("tags.mlflow.runName", "Unnamed")
            )
        )

    print(f"Comparison report saved to: {comp_md_path}")
    return best_run
