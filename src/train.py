import os
import subprocess
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

mlflow.set_experiment("fraud_detection_stack")

# -------------------- Utility Functions -------------------- #
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

def write_commit_tag(exp_id, model_uuid, commit_hash):
    """Write commit hash into the correct MLflow model tags folder."""
    tag_dir = os.path.join("mlruns", str(exp_id), "models", f"m-{model_uuid}", "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash or "")

# -------------------- Main Training -------------------- #
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

        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        client = MlflowClient()
        try:
            client.create_registered_model("fraud_stack_model")
        except RestException:
            pass  # Ignore if model already exists

        # Create a placeholder model version so MLflow generates the model UUID
        model_source = os.path.join("models", "stack_model.pkl")
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, model_source)

        mv = client.create_model_version(
            name="fraud_stack_model",
            source=model_source,
            run_id=run.info.run_id
        )

        # Write commit hash into exact MLflow folder
        write_commit_tag(run.info.experiment_id, mv.version_uuid, commit_hash)

        # Now log model (folder already exists → no FileNotFoundError)
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Save scaler locally
        joblib.dump(scaler, "models/scaler.pkl")

        print(f"\n Training complete. Model + Scaler saved.")
        print(f" Commit hash written to: mlruns/{run.info.experiment_id}/models/m-{mv.version_uuid}/tags")

if __name__ == "__main__":
    train()