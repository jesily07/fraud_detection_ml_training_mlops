# train_model.py

import os
import joblib
import mlflow
import mlflow.sklearn
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
EXPERIMENT_NAME = "fraud_detection_stack"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


def train():
    # Load dataset
    df = pd.read_csv("data/creditcard.csv")
    X, y = df.drop("Class", axis=1), df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # Base models
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

    # Meta-model
    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )

    # MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train_fraud_stack") as run:
        stack_model.fit(X_train_res, y_train_res)

        # Save locally
        model_path = os.path.join(MODEL_DIR, "stack_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(stack_model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"[INFO] Model saved to: {model_path}")
        print(f"[INFO] Scaler saved to: {scaler_path}")

        # Log model to MLflow
        mlflow.sklearn.log_model(stack_model, "model")

    # Return test set for evaluation
    return X_test, y_test, stack_model


if __name__ == "__main__":
    train()