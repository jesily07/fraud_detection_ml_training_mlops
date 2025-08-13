import os
import joblib
import mlflow
import mlflow.sklearn
import subprocess

def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

def register_model():
    experiment_name = "fraud_detection_stack"
    mlflow.set_experiment(experiment_name)
    commit_hash = get_git_commit_hash()

    model_path = "models/stack_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found at {model_path}")

    stack_model = joblib.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[INFO] Started run {run_id} in experiment {experiment_name}")
        print(f"[INFO] Using commit hash: {commit_hash}")

        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # Step 1 — Log model only
        mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="fraud-stack-model"
        )
        print(f"[INFO] Model logged to run {run_id}")

        # Step 2 — Register from run URI (ensures files exist)
        model_uri = f"runs:/{run_id}/fraud-stack-model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="fraud_stack_model"
        )
        print(f"[INFO] Created model version {registered_model.version} of 'fraud_stack_model'")

        # Step 3 — Add commit hash tag
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name="fraud_stack_model",
            version=registered_model.version,
            key="mlflow.source.git.commit",
            value=commit_hash
        )
        print(f"[OK] Commit hash tag added: {commit_hash}")

    print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()