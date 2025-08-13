import os
import joblib
import tempfile
import mlflow
import mlflow.sklearn
from datetime import datetime
import subprocess

def get_git_commit_hash():
    """Return current git commit hash."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown"

def register_model():
    # === Setup ===
    experiment_name = "fraud_detection_stack"
    mlflow.set_experiment(experiment_name)
    commit_hash = get_git_commit_hash()

    # === Load model from train_model output ===
    model_path = "models/stack_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Could not find model at {model_path}")

    stack_model = joblib.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[INFO] Started run {run_id} in experiment {mlflow.get_experiment(run.info.experiment_id).name}")
        print(f"[INFO] Using commit hash: {commit_hash}")

        # === Log scaler if exists ===
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # === Step 1: Log model (NO registration yet) ===
        logged_model_info = mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="fraud-stack-model"  # will store under artifacts/
        )
        print(f"[INFO] Model logged at: {logged_model_info.model_uri}")

        # === Step 2: Register model from logged URI ===
        model_uri = f"runs:/{run_id}/fraud-stack-model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="fraud_stack_model"
        )
        print(f"[INFO] Created model version {registered_model.version} of 'fraud_stack_model'")

        # === Step 3: Inject commit hash tag AFTER folder exists ===
        try:
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                name="fraud_stack_model",
                version=registered_model.version,
                key="mlflow.source.git.commit",
                value=commit_hash
            )
            print(f"[OK] Commit hash tag ({commit_hash}) added to model version {registered_model.version}")
        except Exception as e:
            print(f"[WARN] Could not set commit hash tag: {e}")

    print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()