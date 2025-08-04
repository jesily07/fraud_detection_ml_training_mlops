import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Set MLflow experiment
mlflow.set_experiment("fraud_detection_stack")


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def evaluate_model(y_test, y_pred, y_proba):
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n ROC-AUC Score: {roc_auc:.4f}")
    return roc_auc


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        commit_hash = "not_available"
    return commit_hash


def train():
    # Load and preprocess
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Scale only on training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE only after split
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    meta_clf = LogisticRegression(max_iter=1000)

    # Stacking Ensemble
    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    # MLflow run
    with mlflow.start_run():
        stack_model.fit(X_train_res, y_train_res)
        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate and log metrics
        roc_auc = evaluate_model(y_test, y_pred, y_proba)
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        # Log Git commit hash as a tag
        commit_hash = get_git_commit_hash()
        mlflow.set_tag("source_commit", commit_hash)

        # Log model safely
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Save artifacts locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")


if __name__ == "__main__":
    train()