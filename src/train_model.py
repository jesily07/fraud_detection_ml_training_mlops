# train_model.py +  MLflow logging + confusion matrix artifact + ROC Curve

import os
import warnings
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def log_confusion_matrix(y_test, y_pred, run_id):
    """Generate and log confusion matrix as MLflow artifact (PNG)."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fraud (0)", "Fraud (1)"],
                yticklabels=["Non-Fraud (0)", "Fraud (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Fraud Detection")

    cm_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path, artifact_path="plots")
    print(f"[INFO] Confusion matrix logged as artifact: {cm_path}")


def log_roc_curve(y_test, y_proba, run_id):
    """Generate and log ROC curve as MLflow artifact (PNG)."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection")
    plt.legend(loc="lower right")

    roc_path = f"roc_curve_{run_id}.png"
    plt.savefig(roc_path)
    plt.close()

    mlflow.log_artifact(roc_path, artifact_path="plots")
    print(f"[INFO] ROC curve logged as artifact: {roc_path}")


def train():
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="train_fraud_stack") as run:
        run_id = run.info.run_id

        stack_model.fit(X_train_res, y_train_res)

        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"\n[INFO] Metrics logged: "
              f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, "
              f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        # Save artifacts locally
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "stack_model.pkl")
        joblib.dump(stack_model, model_path)
        print(f"[INFO] Model saved to: {model_path}")

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved to: {scaler_path}")

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(stack_model, artifact_path="fraud_stack_model")

        # === NEW: log confusion matrix + ROC curve ===
        log_confusion_matrix(y_test, y_pred, run_id)
        log_roc_curve(y_test, y_proba, run_id)

        print("\n[INFO] Training complete. Metrics + artifacts logged to MLflow.")


if __name__ == "__main__":
    train()