import os
import time
import shutil
import tempfile
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import subprocess

EXPERIMENT_NAME = "fraud_detection_stack"
REGISTERED_MODEL_NAME = "fraud_stack_model"
GIT_COMMIT_TAG = "mlflow.source.git.commit"
WAIT_FOR_FOLDER_SECONDS = 5

def get_git_commit():
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown"

def wait_for_model_folder(exp_id, version_number):
    """Wait for the m-<uuid> folder to appear, retry for up to WAIT_FOR_FOLDER_SECONDS."""
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    for _ in range(WAIT_FOR_FOLDER_SECONDS):
        if os.path.exists(models_dir):
            for name in os.listdir(models_dir):
                meta_path = os.path.join(models_dir, name, "meta.yaml")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        if f"version: {version_number}" in f.read():
                            return os.path.join(models_dir, name)
        time.sleep(1)
    return None

def write_commit_tag(model_folder, commit_hash):
    tags_dir = os.path.join(model_folder, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    tag_path = os.path.join(tags_dir, GIT_COMMIT_TAG)
    with open(tag_path, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    print(f"[OK] Wrote commit tag: {tag_path}")

def register_model():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Always start with a clean run
    mlflow.end_run()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        print(f"[INFO] Started run {run_id} in experiment {exp_id}")

        # Load your trained model
        model_path = "models/stack_model.pkl"  # Match train_model.py output

        # Save locally first
        temp_dir = tempfile.mkdtemp()
        saved_path = os.path.join(temp_dir, "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # Ensure registered model exists
        try:
            client.create_registered_model(REGISTERED_MODEL_NAME)
            print(f"[INFO] Created registered model '{REGISTERED_MODEL_NAME}'")
        except mlflow.exceptions.RestException:
            print(f"[INFO] Registered model '{REGISTERED_MODEL_NAME}' already exists")

        # Create model version from saved model
        version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=saved_path,
            run_id=run_id
        )
        print(f"[INFO] Created model version {version.version} (source={saved_path})")

        # Wait for real m-<uuid> folder
        model_folder = wait_for_model_folder(exp_id, version.version)
        if not model_folder:
            fallback_name = f"m-fallback-{run_id[:8]}"
            model_folder = os.path.join("mlruns", str(exp_id), "models", fallback_name)
            os.makedirs(model_folder, exist_ok=True)
            print(f"⚠ Could not find m-<uuid> folder after {WAIT_FOR_FOLDER_SECONDS}s. Using fallback: {model_folder}")

        # Write commit tag
        commit_hash = get_git_commit()
        write_commit_tag(model_folder, commit_hash)

        # Log artifacts *before* cleanup
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
            print(f"[OK] Logged manual_model artifacts from {saved_path}")
        except Exception as e:
            print(f"[WARN] Failed to log artifacts: {e}")

        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

        print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()