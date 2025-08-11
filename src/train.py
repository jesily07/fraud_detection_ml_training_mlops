import os
import subprocess
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
import mlflow
import mlflow.sklearn

# Ignore warnings
warnings.filterwarnings("ignore")

# Ensure xgboost is installed
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    import xgboost as xgb


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)


def train():
    # Load dataset
    data = load_data("data/creditcard.csv")
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define base models
    estimators = [
        ("lr", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("xgb", xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)),
    ]

    # Stacking model
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )

    # Start MLflow run
    mlflow.set_experiment("fraud_detection_stack")
    with mlflow.start_run():
        stack_model.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = stack_model.predict(X_test)
        y_proba = stack_model.predict_proba(X_test)[:, 1]

        print("\n Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\n ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        # Log model
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Optional Git commit tagging
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)
        except Exception:
            print(" Git commit tagging skipped — not a valid git repo or git not available.")


if __name__ == "__main__":
    train()