import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from src.heart_disease.config import RANDOM_STATE


def compute_cv_metrics(pipeline, X_train, y_train, cv=5):
    """
    Computes cross-validation metrics using a stratified split.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring)

    cv_metrics = {}
    for metric in scoring:
        cv_metrics[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        cv_metrics[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
    return cv_metrics


def evaluate_test_set(pipeline, X_test, y_test):
    """
    Evaluates predictions on the held-out test set.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_pred, y_proba


def save_plots(y_test, y_pred, y_proba, model_name, output_dir="."):
    """
    Generates and saves Confusion Matrix and ROC Curve plots as PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.abspath(
        os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    )
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.abspath(
        os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    )
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    return cm_path, roc_path
