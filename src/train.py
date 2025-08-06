import os
import subprocess
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

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


def train():
    # Load and preprocess
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # Scale and apply SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Define models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    # Start MLflow run
    with mlflow.start_run() as run:
        stack_model.fit(X_train_res, y_train_res)
        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        roc_auc = evaluate_model(y_test, y_pred, y_proba)

        # Git commit tag
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        # Log artifacts
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        # Fix: Forcefully write missing commit tag file to model tags directory
        if commit_hash:
            experiment = mlflow.get_experiment_by_name("fraud_detection_stack")
            run_id = run.info.run_id
            if experiment:
                experiment_id = experiment.experiment_id
                model_root = os.path.join("mlruns", experiment_id, "models")
                if os.path.exists(model_root):
                    for model_dir in os.listdir(model_root):
                        tags_path = os.path.join(model_root, model_dir, "tags")
                        os.makedirs(tags_path, exist_ok=True)
                        commit_tag_path = os.path.join(tags_path, "mlflow.source.git.commit")
                        if not os.path.exists(commit_tag_path):
                            with open(commit_tag_path, "w", encoding="utf-8") as f:
                                f.write(commit_hash)

        # Register model to MLflow
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")


if __name__ == "__main__":
    train()