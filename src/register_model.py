import os
import shutil
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
from datetime import datetime

# ===========================================
# CONFIG — disable MLflow auto commit tag
# ===========================================
os.environ["MLFLOW_AUTOLOGGING_DISABLE"] = "1"
os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING"] = "1"
os.environ["MLFLOW_DISABLE_SOURCE_LOGGING"] = "1"

EXPERIMENT_NAME = "fraud_detection_stack"
MODEL_NAME = "FraudDetectionModel"
MODEL_DIR = "models/stack_model.pkl"
COMMIT_HASH = "abc123def456"  # replace with actual `git rev-parse HEAD`

# ===========================================
# UTILS
# ===========================================
def find_model_uuid(base_path, version_number):
    """
    Find the m-<uuid> folder for a given model version by checking meta.yaml
    """
    models_path = os.path.join(base_path, "models")
    if not os.path.exists(models_path):
        return None

    for folder in os.listdir(models_path):
        if folder.startswith("m-"):
            meta_path = os.path.join(models_path, folder, "meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    content = f.read()
                    if f"version: {version_number}" in content:
                        return folder
    return None

def write_commit_tag(model_folder, commit_hash):
    """
    Write commit hash to mlflow.source.git.commit file inside tags/
    """
    tags_dir = os.path.join(model_folder, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    commit_file = os.path.join(tags_dir, "mlflow.source.git.commit")
    with open(commit_file, "w") as f:
        f.write(commit_hash)
    print(f"[INFO] Commit tag written to: {commit_file}")

# ===========================================
# MAIN
# ===========================================
def register_model():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Ensure fresh run
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run() as run:
        # Load the trained model
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Trained model not found at {MODEL_DIR}")
        model = joblib.load(MODEL_DIR)

        # Step 1: Log model without MLflow’s internal commit tag
        print("[INFO] Logging model to MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Step 2: Register model
        print("[INFO] Registering model version...")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run.info.run_id
        )
        print(f"[INFO] Model version {mv.version} created.")

        # Step 3: Locate m-<uuid> folder
        base_path = os.path.join("mlruns", run.info.experiment_id)
        model_uuid_folder = find_model_uuid(base_path, mv.version)
        if not model_uuid_folder:
            print("[ERROR] Could not locate m-<uuid> folder.")
            return

        model_folder_path = os.path.join(base_path, "models", model_uuid_folder)

        # Step 4: Write commit tag manually
        write_commit_tag(model_folder_path, COMMIT_HASH)

        # Step 5: Verify file exists
        commit_path = os.path.join(model_folder_path, "tags", "mlflow.source.git.commit")
        if os.path.exists(commit_path):
            print(f"[SUCCESS] Commit tag file verified: {commit_path}")
        else:
            print("[WARNING] Commit tag file missing after write!")

if __name__ == "__main__":
    register_model()