# Production Heart Disease Classifier

This is the serialized end-to-end model pipeline for the Heart Disease classification model.

## Model Metadata
- **Type**: logistic_regression
- **Training Run ID**: `02fef5025e5e49e79c3357e4779bc7b0`
- **Trained on**: 2026-07-10T04:09:36.354794+00:00
- **Dataset Size**: 303 rows
- **Test ROC-AUC**: 0.9545
- **Test F1-Score**: 0.8621

## Usage Instructions

To load and run predictions with this model, use the following code:

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load("model.pkl")

# Prepare raw inputs
sample = pd.DataFrame([{
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
}])

# Predict binary target (0 = no disease, 1 = presence of disease)
pred = pipeline.predict(sample)
proba = pipeline.predict_proba(sample)

print("Prediction:", pred[0])
print("Probabilities:", proba[0])
```
