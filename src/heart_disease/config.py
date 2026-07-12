import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "notebooks", "data", "heart_disease_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "production")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Columns
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SKEWED_COLS = ["chol", "oldpeak"]
OTHER_NUMERIC_COLS = ["age", "trestbps", "thalach"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"

# Hyperparams & seed
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
