import os
import joblib
import subprocess
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml

MODEL_NAME = "fraud_stack_model"
EXPERIMENT_NAME = "fraud_detection_stack"

mlflow.set_experiment(EXPERIMENT_NAME)

def get_git_commit_hash():
    """Fetch the current git commit hash."""
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

def find_model_uuid_from_version(exp_id, model_version):
    """
    Locate the actual m-<uuid> folder by reading meta.yaml for the given model version.
    This avoids guessing and works across MLflow versions.
    """
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.exists(models_dir):
        return None

    for folder in os.listdir(models_dir):
        if folder.startswith("m-"):
            meta_path = os.path.join(models_dir, folder, "meta.yaml")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = yaml.safe_load(f)
                    if str(meta.get("version")) == str(model_version):
                        return folder
                except Exception:
                    pass
    return None

def write_commit_tag(exp_id, model_uuid, commit_hash):
    """Write mlflow.source.git.commit tag file inside the given model folder."""
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_uuid, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    print(f" Commit tag written successfully at: {tag_file}")

def register_model():
    client = MlflowClient()

    # Load trained model & scaler
    stack_model = joblib.load("models/stack_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    commit_hash = get_git_commit_hash()
    if commit_hash:
        mlflow.set_tag("mlflow.source.git.commit", commit_hash)

    with mlflow.start_run() as run:
        # Log params (example)
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")

        # --- Step 1: Log model first so m-<uuid> folder is created ---
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # --- Step 2: Register model version ---
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass  # Ignore if already exists

        mv = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run.info.run_id
        )

        # --- Step 3: Locate the real m-<uuid> folder ---
        model_uuid = find_model_uuid_from_version(run.info.experiment_id, mv.version)
        if not model_uuid:
            print(" Could not determine model folder for commit tag.")
            return

        # --- Step 4: Write commit tag after folder exists ---
        if commit_hash:
            write_commit_tag(run.info.experiment_id, model_uuid, commit_hash)

        # --- Step 5: Verification ---
        model_folder_path = os.path.join("mlruns", str(run.info.experiment_id), "models", model_uuid)
        if os.path.exists(model_folder_path):
            print(f" Verified model folder exists: {model_folder_path}")
        else:
            print(f" Model folder not found after registration: {model_folder_path}")

if __name__ == "__main__":
    register_model()