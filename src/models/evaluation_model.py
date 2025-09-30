# evaluate_model.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import mlflow

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def evaluate():
    model_dir = "models"
    model_path = os.path.join(model_dir, "stack_model.pkl")
    test_data_path = os.path.join(model_dir, "test_data.pkl")

    # Load model and test data
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\n ROC-AUC Score: {roc_auc:.4f}")

    # Save plots
    os.makedirs("artifacts_tmp", exist_ok=True)
    cm_path = os.path.join("artifacts_tmp", "confusion_matrix.png")
    roc_path = os.path.join("artifacts_tmp", "roc_curve.png")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_proba, roc_path)

    # Log to MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

    print("\n[INFO] Evaluation complete. Metrics and plots logged to MLflow.")

if __name__ == "__main__":
    evaluate()