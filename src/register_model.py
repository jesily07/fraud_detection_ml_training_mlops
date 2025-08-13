import os
import joblib
import mlflow
import mlflow.sklearn
import tempfile
import subprocess
import warnings

# ==============================
# CONFIGURATION
# ==============================
EXPERIMENT_NAME = "fraud_detection_stack"
MODEL_NAME = "fraud_stack_model"
LOCAL_MODEL_PATH = os.path.join("models", "stack_model.pkl")

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

def register_model():
    # Ensure the MLflow experiment exists
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load the local trained model
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {LOCAL_MODEL_PATH}")
    stack_model = joblib.load(LOCAL_MODEL_PATH)
    print(f"[INFO] Loaded model from: {LOCAL_MODEL_PATH}")

    # Start an MLflow run
    with mlflow.start_run(run_name="register_fraud_stack") as run:
        commit_hash = get_git_commit_hash()
        print(f"[INFO] Started run {run.info.run_id} in experiment {EXPERIMENT_NAME}")
        print(f"[INFO] Using commit hash: {commit_hash}")

        # Save model to a temp directory for registration
        saved_path = os.path.join(tempfile.mkdtemp(), "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # ===== Manual pre-log step (optional) =====
        # This is where the harmless [WARN] can appear if python_env.yaml is not found.
        # Registration will still succeed regardless.
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
            print(f"[INFO] Pre-logged all artifacts from: {saved_path}")
        except Exception as e:
            print(f"[WARN - cosmetic only] Could not pre-log artifacts: {e}")

        # Register the model
        model_uri = f"file://{saved_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"[INFO] Created registered model '{MODEL_NAME}' (version {mv.version})")

        # Tag model version with commit hash
        try:
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(MODEL_NAME, mv.version, "mlflow.source.git.commit", commit_hash)
            print(f"[OK] Wrote commit tag to model version {mv.version}")
        except Exception as e:
            print(f"[WARN] Failed to set commit tag: {e}")

if __name__ == "__main__":
    register_model()