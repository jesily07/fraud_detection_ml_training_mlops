# re_evaluation_model.py

import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
import seaborn as sns


# ----------------------------
# Utility: plot confusion matrix
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


# ----------------------------
# Utility: plot ROC curve
# ----------------------------
def plot_roc(y_true, y_pred_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_pred_proba):.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()


# ----------------------------
# Main evaluation function
# ----------------------------
def evaluate():
    os.makedirs("artifacts_tmp", exist_ok=True)

    # Detect which model to use
    model_optuna_path = "models/stack_model_optuna.pkl"
    test_optuna_path = "models/test_data_optuna.pkl"
    model_base_path = "models/stack_model.pkl"
    test_base_path = "models/test_data.pkl"

    if os.path.exists(model_optuna_path) and os.path.exists(test_optuna_path):
        model_path = model_optuna_path
        test_data_path = test_optuna_path
        model_type = "optuna"
    elif os.path.exists(model_base_path) and os.path.exists(test_base_path):
        model_path = model_base_path
        test_data_path = test_base_path
        model_type = "base"
    else:
        raise FileNotFoundError(
            "No model/test data found in models/. Run train_model.py or optimize_model.py first."
        )

    print(f"[INFO] Using {model_type} model for evaluation: {model_path}")

    # Load model + test data
    model = joblib.load(model_path)
    X_test_scaled, y_test = joblib.load(test_data_path)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    print("\n Confusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(report)
    print("\n ROC-AUC Score:", round(roc_auc, 4))

    # Plots
    cm_path = os.path.join("artifacts_tmp", f"confusion_matrix_{model_type}.png")
    roc_path = os.path.join("artifacts_tmp", f"roc_curve_{model_type}.png")
    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc(y_test, y_pred_proba, roc_path)

    # MLflow logging
    with mlflow.start_run(run_name=f"re-evaluation_{model_type}_stack_model"):
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

    print(
        f"[INFO] Re-evaluation complete for {model_type} model. Metrics and plots logged to MLflow."
    )


if __name__ == "__main__":
    evaluate()