import os
import mlflow
import mlflow.sklearn
import joblib
import subprocess

def register_model():
    # === Setup experiment ===
    experiment_name = "fraud_detection_stack"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print(f"[INFO] Started run {run.info.run_id} in experiment {run.info.experiment_id}")

        # === Load locally saved artifacts from training step ===
        model_dir = "models"
        stack_model_path = os.path.join(model_dir, "stack_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(stack_model_path):
            raise FileNotFoundError(f"[ERROR] stack_model.pkl not found in {model_dir}")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"[ERROR] scaler.pkl not found in {model_dir}")

        stack_model = joblib.load(stack_model_path)

        # === Auto-capture Git commit hash ===
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
        except Exception:
            commit_hash = "unknown"

        # === Log scaler as an extra artifact ===
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # === Log & register the model in one step ===
        logged_model_info = mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="fraud-stack-model",
            registered_model_name="fraud_stack_model"
        )

        # === Tag the run with commit hash for reproducibility ===
        mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        print(f"[INFO] Model logged & registered at URI: {logged_model_info.model_uri}")
        print(f"[OK] Commit hash tagged: {commit_hash}")
        print("[DONE] register_model finished without warnings.")

if __name__ == "__main__":
    register_model()