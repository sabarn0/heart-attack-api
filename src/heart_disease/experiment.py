import mlflow
import mlflow.sklearn
from src.heart_disease.config import MLRUNS_DIR


def init_experiment():
    """
    Sets the tracking URI and defines the MLflow experiment.
    """
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("heart-disease-classification")


def log_experiment_run(
    run_name, model_type, best_params, metrics, pipeline, cm_path, roc_path, dataset_version
):
    """
    Logs hyperparameters, metrics, artifacts, and tags to MLflow.
    """
    init_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        # Log tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset_version", dataset_version)

        # Log hyperparams
        for p_name, p_val in best_params.items():
            # Clean parameter name (strip "model__" if present)
            clean_name = p_name.replace("model__", "")
            mlflow.log_param(clean_name, p_val)

        # Log metrics
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)

        # Log artifacts (plots)
        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(roc_path, "plots")

        mlflow.sklearn.log_model(pipeline, "model", serialization_format="pickle")

        print(f"Logged run '{run_name}' to MLflow. Run ID: {run.info.run_id}")
        return run.info.run_id
