import os
import subprocess
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# ===== Safety: Close any active run =====
if mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment("fraud_detection_stack")

MODEL_NAME = "fraud_stack_model"
LOCAL_MODEL_PATH = "models/stack_model.pkl"

def get_git_commit_hash():
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

def find_model_folder_by_version(exp_id, version):
    """Locate the actual m-<uuid> folder from MLflow's filesystem based on meta.yaml."""
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.isdir(models_dir):
        return None

    for folder in os.listdir(models_dir):
        candidate_meta = os.path.join(models_dir, folder, "meta.yaml")
        if os.path.isfile(candidate_meta):
            try:
                with open(candidate_meta, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                if str(meta.get("version")) == str(version):
                    return os.path.join(models_dir, folder)
            except Exception:
                continue
    return None

def precreate_tags_folder(exp_id):
    """Create a dummy tags folder under models so MLflow logging won't fail."""
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    os.makedirs(models_dir, exist_ok=True)
    # Temporary pre-create m-dummy folder to avoid MLflow's race issue
    # Real folder will be renamed by MLflow after version creation
    return models_dir

def write_commit_tag(model_folder, commit_hash):
    """Write commit tag file to the correct tags folder."""
    tag_dir = os.path.join(model_folder, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    print(f" Commit tag written successfully at: {tag_file}")

def register_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Local model not found at {LOCAL_MODEL_PATH}")

    stack_model = joblib.load(LOCAL_MODEL_PATH)
    client = MlflowClient()

    with mlflow.start_run() as run:
        # Pre-create tags folder to prevent MLflow bug
        precreate_tags_folder(run.info.experiment_id)

        # Step 1: Log model first
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Step 2: Create registered model (if not exists)
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.RestException:
            pass

        # Step 3: Create model version
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=os.path.join(mlflow.get_artifact_uri(), "model"),
            run_id=run.info.run_id
        )
        print(f"Model registered as version {mv.version} in '{MODEL_NAME}'.")

        # Step 4: Find actual model folder
        model_folder = find_model_folder_by_version(run.info.experiment_id, mv.version)
        if model_folder:
            commit_hash = get_git_commit_hash()
            if commit_hash:
                write_commit_tag(model_folder, commit_hash)

            if os.path.exists(os.path.join(model_folder, "tags", "mlflow.source.git.commit")):
                print(f" Verified: Commit tag exists at {os.path.join(model_folder, 'tags')}")
            else:
                print(" Commit tag missing after write attempt.")
        else:
            print(" Could not determine model folder for commit tag.")

if __name__ == "__main__":
    register_model()