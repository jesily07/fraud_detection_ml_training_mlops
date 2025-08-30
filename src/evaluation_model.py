# evaluate_model.py

import os
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred, cm_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()


def plot_roc_curve(y_test, y_proba, roc_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()


def evaluate():
    model_dir = "models"
    model_path = os.path.join(model_dir, "stack_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    test_data_path = os.path.join(model_dir, "test_data.pkl")

    # Load model, scaler, and test data
    stack_model = joblib.load(model_path)
    _ = joblib.load(scaler_path)  # not directly needed here
    X_test_scaled, y_test = joblib.load(test_data_path)

    # Predictions
    y_pred = stack_model.predict(X_test_scaled)
    y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print summary
    print("\n[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\n[INFO] Precision: {precision:.4f}")
    print(f"[INFO] Recall: {recall:.4f}")
    print(f"[INFO] F1 Score: {f1:.4f}")
    print(f"[INFO] ROC-AUC Score: {roc_auc:.4f}")

    # Start MLflow logging
    with mlflow.start_run(run_name="model_evaluation") as run:
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("ROC-AUC", roc_auc)

        # Save plots
        os.makedirs("artifacts_tmp", exist_ok=True)
        cm_path = "artifacts_tmp/confusion_matrix.png"
        roc_path = "artifacts_tmp/roc_curve.png"

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_proba, roc_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        print(f"[INFO] Metrics & plots logged to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    evaluate()