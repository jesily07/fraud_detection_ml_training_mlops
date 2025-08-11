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


# ===============================
# Utility Functions
# ===============================
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Data file not found: {filepath}")
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


def find_model_folder(exp_id, model_version):
    """
    Dynamically finds the actual model folder name created by MLflow under:
    mlruns/<exp_id>/models/<model_folder>
    """
    models_path = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.exists(models_path):
        return None

    for folder in os.listdir(models_path):
        folder_path = os.path.join(models_path, folder)
        if os.path.isdir(folder_path):
            tags_file = os.path.join(folder_path, "version")
            if os.path.exists(tags_file):
                try:
                    with open(tags_file, "r", encoding="utf-8") as f:
                        version_in_file = f.read().strip()
                    if str(model_version) == version_in_file:
                        return folder
                except Exception:
                    pass
    return None


def write_commit_tag(exp_id, model_folder, commit_hash):
    """
    Writes the commit hash to the mlflow.source.git.commit file
    in the appropriate tags folder.
    """
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_folder, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)


# ===============================
# Main Training Function
# ===============================
def train():
    # Load and split
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    meta_clf = LogisticRegression(max_iter=1000)

    # Stacking ensemble
    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    client = MlflowClient()

    # Ensure model registry entry exists
    try:
        client.create_registered_model("fraud_stack_model")
    except RestException:
        pass  # already exists

    with mlflow.start_run() as run:
        # Train model
        stack_model.fit(X_train_res, y_train_res)

        # Predictions
        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        roc_auc = evaluate_model(y_test, y_pred, y_proba)

        # Log metrics and params
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_metric("roc_auc", roc_auc)

        # Git commit tracking
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        # Log model
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Register model version
        mv = client.create_model_version(
            name="fraud_stack_model",
            source=os.path.join(mlflow.get_artifact_uri(), "model"),
            run_id=run.info.run_id
        )

        # Dynamically find the actual model folder name
        model_folder = find_model_folder(run.info.experiment_id, mv.version)
        if model_folder and commit_hash:
            write_commit_tag(run.info.experiment_id, model_folder, commit_hash)

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")


if __name__ == "__main__":
    train()