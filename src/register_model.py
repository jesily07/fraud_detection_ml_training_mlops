import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import subprocess
import tempfile

def register_model():
    # === Setup experiment ===
    experiment_name = "fraud_detection_stack"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        print(f"[INFO] Started run {run.info.run_id} in experiment {run.info.experiment_id}")

        # === Load trained model ===
        model_dir = "models"
        stack_model_path = os.path.join(model_dir, "stack_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not all(map(os.path.exists, [stack_model_path, scaler_path])):
            raise FileNotFoundError("Model or scaler file missing")

        stack_model = joblib.load(stack_model_path)
        scaler = joblib.load(scaler_path)

        # === Create temporary directory for model artifacts ===
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model and scaler in temp directory
            model_path = os.path.join(tmp_dir, "model.pkl")
            scaler_path_in_tmp = os.path.join(tmp_dir, "scaler.pkl")
            joblib.dump(stack_model, model_path)
            joblib.dump(scaler, scaler_path_in_tmp)
            
            # Log entire directory as MLflow artifacts
            mlflow.log_artifacts(tmp_dir, artifact_path="model")
            print(f"[INFO] Logged model artifacts from: {tmp_dir}")

        # === Log model separately for registry ===
        model_info = mlflow.sklearn.log_model(
            sk_model=stack_model,
            artifact_path="registered_model",
            registered_model_name="fraud_stack_model",
            extra_pip_requirements=["scikit-learn", "joblib"]
        )

        # === Auto-capture git commit hash ===
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            commit_hash = "unknown"
        
        print(f"[INFO] Using commit hash: {commit_hash}")

        # === Tag model version ===
        client = MlflowClient()
        model_version = model_info.model_version
        
        client.set_model_version_tag(
            name="fraud_stack_model",
            version=model_version,
            key="mlflow.source.git.commit",
            value=commit_hash
        )

        print(f"[OK] Tagged model version {model_version} with commit hash")
        print(f"[DONE] Model registered: {model_info.model_uri}")

if __name__ == "__main__":
    register_model()