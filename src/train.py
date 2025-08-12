import os
import subprocess
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

# Silence XGBoost use_label_encoder warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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


def find_latest_model_folder(exp_id):
    """Locate the most recent m-<uuid> model folder in mlruns/<exp_id>/models/"""
    models_path = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.exists(models_path):
        return None
    m_folders = [f for f in os.listdir(models_path) if f.startswith("m-")]
    if not m_folders:
        return None
    latest_folder = max(m_folders, key=lambda f: os.path.getctime(os.path.join(models_path, f)))
    return latest_folder


def write_commit_tag(exp_id, model_folder, commit_hash):
    """Write commit hash to mlflow.source.git.commit inside model's tags folder"""
    if not model_folder:
        print(" Could not determine model folder for commit tag.")
        return
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_folder, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    if os.path.exists(tag_file):
        print(f" Commit tag written successfully to {tag_file}")


def train():
    # Load and preprocess
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
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

        # Step 1: Create Registered Model (safe if exists)
        try:
            client.create_registered_model("fraud_stack_model")
        except Exception:
            pass  # model already exists

        # Step 2: Create Model Version → triggers folder creation
        try:
            client.create_model_version(
                name="fraud_stack_model",
                source=os.path.join(mlflow.get_artifact_uri(), "model"),
                run_id=run.info.run_id
            )
        except Exception as e:
            print(f" create_model_version failed: {e}")

        # Step 3: Find actual model folder from filesystem
        model_folder = find_latest_model_folder(run.info.experiment_id)

        # Step 4: Write commit tag before logging the model
        if commit_hash:
            write_commit_tag(run.info.experiment_id, model_folder, commit_hash)

        # Step 5: Log model
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Step 6: Post-run folder verification
        if model_folder:
            full_path = os.path.join("mlruns", str(run.info.experiment_id), "models", model_folder)
            if os.path.exists(full_path):
                print(f" Verified MLflow model folder exists: {full_path}")
            else:
                print(f" Model folder {full_path} not found after run.")
        else:
            print(" No model folder detected for verification.")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_model, "models/stack_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("\n Training complete. Model + Scaler saved.")


if __name__ == "__main__":
    train()