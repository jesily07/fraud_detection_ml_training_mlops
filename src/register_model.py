import os
import subprocess
import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "fraud_detection_stack"
REGISTERED_MODEL_NAME = "fraud_stack_model"
MODELS_DIR = "models"
STACK_MODEL_PATH = os.path.join(MODELS_DIR, "stack_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def get_git_commit_hash():
    """Get current git commit hash if available."""
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


def find_model_uuid_from_version(exp_id, version_number):
    """
    Scan mlruns/<exp_id>/models/ to find m-<uuid> folder
    whose meta.yaml matches given version_number.
    """
    models_root = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.exists(models_root):
        return None

    for folder in os.listdir(models_root):
        if folder.startswith("m-"):
            meta_path = os.path.join(models_root, folder, "meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                    if str(meta.get("version")) == str(version_number):
                        return folder
    return None


def write_commit_tag(exp_id, model_uuid, commit_hash):
    """
    Write commit hash to mlflow.source.git.commit inside model folder.
    """
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_uuid, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    print(f" Commit tag written successfully at: {tag_file}")


def register_model():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Load trained artifacts
    if not os.path.exists(STACK_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {STACK_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

    stack_model = joblib.load(STACK_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with mlflow.start_run() as run:
        # Git commit tag
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        # Pre-register model version to create folder structure
        try:
            model_details = client.create_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            pass  # Model may already exist

        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=os.path.join(mlflow.get_artifact_uri(), "model"),
            run_id=run.info.run_id
        )

        # Try finding actual m-<uuid> folder from meta.yaml
        model_uuid_folder = find_model_uuid_from_version(run.info.experiment_id, mv.version)
        if model_uuid_folder and commit_hash:
            write_commit_tag(run.info.experiment_id, model_uuid_folder, commit_hash)
        else:
            print(" Could not determine model folder for commit tag.")

        # Log model to MLflow
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Post-run verification
        if model_uuid_folder:
            folder_path = os.path.join("mlruns", str(run.info.experiment_id), "models", model_uuid_folder)
            if os.path.exists(folder_path):
                print(f" Verified model folder exists: {folder_path}")
            else:
                print(f" Model folder not found: {folder_path}")

    print(" Model registration complete.")


if __name__ == "__main__":
    register_model()