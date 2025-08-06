import os
import subprocess
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def ensure_model_tags_folder(mlruns_dir="mlruns"):
    """
    Ensure 'tags/' folder exists under the latest registered model directory to avoid FileNotFoundError.
    """
    try:
        experiment_ids = [d for d in os.listdir(mlruns_dir) if d.isdigit()]
        if not experiment_ids:
            return
        latest_experiment = max(experiment_ids, key=int)

        model_dir = os.path.join(mlruns_dir, latest_experiment, "models")
        if not os.path.exists(model_dir):
            return

        model_subdirs = [d for d in os.listdir(model_dir) if re.match(r"m-[a-f0-9]{32}", d)]
        if not model_subdirs:
            return

        latest_model = sorted(model_subdirs)[-1]
        tags_path = os.path.join(model_dir, latest_model, "tags")
        os.makedirs(tags_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Error ensuring model tags folder: {e}")


def train():
    # Load and preprocess
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    with mlflow.start_run() as run:
        stack_model.fit(X_train_res, y_train_res)

        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        roc_auc = evaluate_model(y_test, y_pred, y_proba)

        # Ensure tags folder to avoid FileNotFoundError
        ensure_model_tags_folder()

        # Git commit tag
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        # Automatically register model to model registry
        mlflow.sklearn.log_model(
            stack_model,
            artifact_path="model",
            registered_model_name="fraud_detection_model"
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")
        print(f" MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train()