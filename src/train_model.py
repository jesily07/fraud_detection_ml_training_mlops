# train_model.py +  MLflow logging + confusion matrix artifact + ROC Curve

import os
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# =========================
# Helper Functions
# =========================

def log_confusion_matrix(y_true, y_pred, run_id):
    """Generate and log confusion matrix plot as MLflow artifact."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")

    cm_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Ensure MLflow artifact subdir exists
    artifact_dir = os.path.join("mlruns", "0", run_id, "artifacts", "plots")
    os.makedirs(artifact_dir, exist_ok=True)

    mlflow.log_artifact(cm_path, artifact_path="plots")
    os.remove(cm_path)


def log_roc_curve(y_true, y_pred_proba, run_id):
    """Generate and log ROC curve plot as MLflow artifact."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    roc_path = f"roc_curve_{run_id}.png"
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Ensure MLflow artifact subdir exists
    artifact_dir = os.path.join("mlruns", "0", run_id, "artifacts", "plots")
    os.makedirs(artifact_dir, exist_ok=True)

    mlflow.log_artifact(roc_path, artifact_path="plots")
    os.remove(roc_path)


# =========================
# Training Function
# =========================
def train():
    # Load dataset
    df = pd.read_csv("data/creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # MLflow logging
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"[INFO] Metrics logged: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
              f"Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        # Save model + scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print("[INFO] Model saved to: models/stack_model.pkl")
        print("[INFO] Scaler saved to: models/scaler.pkl")

        # Log artifacts
        log_confusion_matrix(y_test, y_pred, run_id)
        log_roc_curve(y_test, y_pred_proba, run_id)


if __name__ == "__main__":
    train()