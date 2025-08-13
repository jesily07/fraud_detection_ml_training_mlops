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

        # === Load trained model and scaler ===
        model_dir = "models"
        stack_model_path = os.path.join(model_dir, "stack_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(stack_model_path):
            raise FileNotFoundError(f"Stack model not found at {stack_model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        stack_model = joblib.load(stack_model_path)
        scaler = joblib.load(scaler_path)

        # === Auto-capture git commit hash ===
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
        except Exception:
            commit_hash = "unknown"
        print(f"[INFO] Using commit hash: {commit_hash}")

        # === Log and register model in one step ===
        # The 'registered_model_name' parameter handles both logging to the run
        # and registering a new model version in the MLflow Model Registry.
        logged_model_info = mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="fraud-stack-model",
            registered_model_name="fraud_stack_model"
        )
        print(f"[OK] Model logged and registered at: {logged_model_info.model_uri}")

        # === Add commit hash as a tag for this run ===
        mlflow.set_tag("mlflow.source.git.commit", commit_hash)
        print(f"[OK] Tagged run with commit hash: {commit_hash}")

        # === Optional: Log additional artifacts like scaler ===
        # You can now log the scaler separately if needed.
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        print(f"[INFO] Logged scaler artifact from: {scaler_path}")

    print(f"[DONE] register_model finished.")


if __name__ == "__main__":
    register_model()