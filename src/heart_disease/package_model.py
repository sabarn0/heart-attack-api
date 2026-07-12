import os
import json
import datetime
import joblib
import mlflow.sklearn
from src.heart_disease.config import MODELS_DIR, NUMERIC_COLS, CATEGORICAL_COLS


def package_winning_model(best_run, dataset_row_count):
    """
    Loads the winning pipeline from MLflow, packages it as model.pkl,
    generates model_card.json, and writes README.md.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    run_id = best_run["run_id"]
    model_type = best_run["tags.model_type"]

    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading winning model from: {model_uri}")
    winning_pipeline = mlflow.sklearn.load_model(model_uri)

    # Save using joblib
    pkl_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(winning_pipeline, pkl_path)
    print(f"Serialized pipeline saved to: {pkl_path}")

    # Gather hyperparameters and metrics
    # Best params in mlflow runs are prefix-free or start with params.*
    params = {}
    metrics = {}
    for col in best_run.index:
        if col.startswith("params."):
            params[col.replace("params.", "")] = best_run[col]
        elif col.startswith("metrics."):
            metrics[col.replace("metrics.", "")] = float(best_run[col])

    # Gather metadata for model_card.json
    model_card = {
        "model_type": model_type,
        "hyperparameters": params,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_row_count": int(dataset_row_count),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "feature_list": {"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS},
    }

    card_path = os.path.join(MODELS_DIR, "model_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=4)
    print(f"Model card written to: {card_path}")

    # Write README.md usage instructions
    readme_path = os.path.join(MODELS_DIR, "README.md")
    readme_content = f"""# Production Heart Disease Classifier

This is the serialized end-to-end model pipeline for the Heart Disease classification model.

## Model Metadata
- **Type**: {model_type}
- **Training Run ID**: `{run_id}`
- **Trained on**: {model_card['training_date']}
- **Dataset Size**: {dataset_row_count} rows
- **Test ROC-AUC**: {metrics.get('test_roc_auc', 0.0):.4f}
- **Test F1-Score**: {metrics.get('test_f1', 0.0):.4f}

## Usage Instructions

To load and run predictions with this model, use the following code:

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load("model.pkl")

# Prepare raw inputs
sample = pd.DataFrame([{{
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
    "thal": 1
}}])

# Predict binary target (0 = no disease, 1 = presence of disease)
pred = pipeline.predict(sample)
proba = pipeline.predict_proba(sample)

print("Prediction:", pred[0])
print("Probabilities:", proba[0])
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"README.md written to: {readme_path}")
