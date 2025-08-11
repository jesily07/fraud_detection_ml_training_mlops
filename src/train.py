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

# -----------------------
# Utility Functions
# -----------------------

def load_data(filepath):
    """Load CSV data with existence check."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
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
    """Get current git commit hash if available."""
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

def find_model_folder(exp_id, model_name):
    """
    Search mlruns folder for the latest model directory for given experiment & model name.
    Returns full path or None.
    """
    base_path = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.exists(base_path):
        return None
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # Optionally could match on model_name, but MLflow's folder IDs are hashed
            return folder_path
    return None

def write_commit_tag(exp_id, model_id, commit_hash):
    """Ensure tags folder exists & write commit hash file."""
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_id, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)

# -----------------------
# Training Pipeline
# -----------------------

def train():
    # Load & preprocess
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    meta_clf = LogisticRegression(max_iter=1000)

    # Stacking ensemble
    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    client = MlflowClient()

    with mlflow.start_run() as run:
        # Fit
        stack_model.fit(X_train_res, y_train_res)

        # Predictions
        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluation
        roc_auc = evaluate_model(y_test, y_pred, y_proba)

        # Commit hash
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        # Log params & metrics
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        # Pre-register model if not exists
        try:
            client.create_registered_model("fraud_stack_model")
        except RestException:
            pass  # Already exists

        # Dynamically find or prepare tags folder before logging model
        model_folder_path = find_model_folder(run.info.experiment_id, "fraud_stack_model")
        model_id = None

        if model_folder_path:
            model_id = os.path.basename(model_folder_path)
        else:
            # If not found yet, create a placeholder using run_id
            model_id = f"m-{run.info.run_id[:8]}"

        if commit_hash:
            write_commit_tag(run.info.experiment_id, model_id, commit_hash)

        # Log model
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")

# -----------------------
# Main Entry
# -----------------------

if __name__ == "__main__":
    train()