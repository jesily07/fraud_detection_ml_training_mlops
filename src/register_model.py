import os
import mlflow
import mlflow.sklearn
import joblib
import subprocess
import shutil
from urllib.parse import urlparse

def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown_commit"

def register_model():
    # === Load trained model ===
    model_path = "models/stack_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file not found at {model_path}")
    stack_model = joblib.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    # === MLflow Setup ===
    mlflow.set_experiment("fraud_detection_stack")
    commit_hash = get_git_commit_hash()

    with mlflow.start_run() as run:
        print(f"[INFO] Started run {run.info.run_id} in experiment fraud_detection_stack")
        print(f"[INFO] Using commit hash: {commit_hash}")

        # Log model without registering immediately → avoids race condition bug
        logged_model_info = mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="fraud-stack-model"
        )
        print(f"[INFO] Model logged at: {logged_model_info.model_uri}")

        # Register model from run URI
        model_uri = f"runs:/{run.info.run_id}/fraud-stack-model"
        registered_model = mlflow.register_model(model_uri=model_uri, name="fraud_stack_model")
        print(f"[INFO] Created registered model 'fraud_stack_model' version {registered_model.version}")

        # Tag commit hash in model registry
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name="fraud_stack_model",
            version=registered_model.version,
            key="mlflow.source.git.commit",
            value=commit_hash
        )
        print(f"[OK] Tagged commit hash ({commit_hash}) to model version {registered_model.version}")

    # === Local cleanup if tracking URI is file-based ===
    tracking_uri = mlflow.get_tracking_uri()
    scheme = urlparse(tracking_uri).scheme
    if scheme in ("", "file"):
        fallback_model_dir = os.path.join("mlruns", run.info.experiment_id, "models")
        if os.path.exists(fallback_model_dir):
            shutil.rmtree(fallback_model_dir, ignore_errors=True)
            print(f"[CLEANUP] Removed local fallback model dir: {fallback_model_dir}")
    else:
        print("[CLEANUP] Skipped — remote tracking URI detected (cloud-safe)")

    print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()