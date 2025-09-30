import os
import joblib
import mlflow
import mlflow.sklearn
import tempfile
import subprocess
import warnings
from mlflow.tracking import MlflowClient

# ==============================
# CONFIGURATION
# ==============================
EXPERIMENT_NAME = "fraud_detection_stack"
MODEL_NAME_BASE = "fraud_stack_model"
MODEL_NAME_TUNED = "fraud_stack_model_optuna"
LOCAL_MODEL_PATH_BASE = os.path.join("models", "stack_model.pkl")
LOCAL_MODEL_PATH_TUNED = os.path.join("models", "stack_model_optuna.pkl")

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

def get_git_commit_hash():
    """Returns the short commit hash if available, else 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

def register_model(model_path, model_name, run_name="register_model"):
    """Generic model registration helper"""
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not os.path.exists(model_path):
        print(f"[WARN] Skipping {model_name} → Model not found at {model_path}")
        return None

    stack_model = joblib.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    with mlflow.start_run(run_name=run_name) as run:
        commit_hash = get_git_commit_hash()
        print(f"[INFO] Started run {run.info.run_id} in experiment {EXPERIMENT_NAME}")
        print(f"[INFO] Using commit hash: {commit_hash}")

        saved_path = os.path.join(tempfile.mkdtemp(), "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)

        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
        except Exception as e:
            print(f"[WARN] Pre-log failed: {e}")

        model_uri = f"file://{saved_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"[INFO] Registered model '{model_name}' (version {mv.version})")

        try:
            client = MlflowClient()
            client.set_model_version_tag(model_name, mv.version, "mlflow.source.git.commit", commit_hash)
            print(f"[OK] Commit tag set for {model_name} v{mv.version}")
        except Exception as e:
            print(f"[WARN] Failed to set commit tag: {e}")

        return mv.version

def main():
    print("\n[STEP] Registering Base Model...")
    base_version = register_model(LOCAL_MODEL_PATH_BASE, MODEL_NAME_BASE, run_name="register_base_model")

    print("\n[STEP] Registering Tuned (Optuna) Model...")
    tuned_version = register_model(LOCAL_MODEL_PATH_TUNED, MODEL_NAME_TUNED, run_name="register_tuned_model")

    if base_version:
        print(f"[INFO] Base model registered as version {base_version}")
    if tuned_version:
        print(f"[INFO] Tuned model registered as version {tuned_version}")

    if not base_version and not tuned_version:
        print("[ERROR] No models found to register.")

if __name__ == "__main__":
    main()