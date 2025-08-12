import os
import subprocess
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import warnings

# Suppress MLflow warnings
warnings.filterwarnings("ignore", category=FutureWarning)

mlflow.set_experiment("fraud_detection_stack")

def get_git_commit_hash():
    """Get the current Git commit hash."""
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

def find_model_folder(exp_id, version_number):
    """
    Locate the m-<uuid> folder for the given model version.
    Reads meta.yaml to confirm the version match.
    """
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.isdir(models_dir):
        return None

    for entry in os.listdir(models_dir):
        if entry.startswith("m-"):
            meta_path = os.path.join(models_dir, entry, "meta.yaml")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)
                    if str(meta.get("version")) == str(version_number):
                        return os.path.join(models_dir, entry)
    return None

def write_commit_tag(tag_dir, commit_hash):
    """Write commit hash to mlflow.source.git.commit file."""
    os.makedirs(tag_dir, exist_ok=True)
    tag_file = os.path.join(tag_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    print(f" Commit tag written: {tag_file}")

def register_model():
    # Ensure no active run
    if mlflow.active_run():
        mlflow.end_run()

    client = MlflowClient()

    # Load locally saved model from training step
    model_path = "models/stack_model.pkl"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained model or scaler not found in 'models/' directory.")

    stack_model = joblib.load(model_path)

    # Get Git commit hash
    commit_hash = get_git_commit_hash()

    # Start a fresh run
    with mlflow.start_run() as run:
        exp_id = run.info.experiment_id

        # Log model artifact ONLY (no registration yet)
        mlflow.sklearn.log_model(stack_model, artifact_path="model")

        # Log params/metrics (optional)
        mlflow.log_param("model_type", "Stacking (RF + XGB + LR)")
        mlflow.log_param("registered_separately", True)

        # Create registered model if not exists
        model_name = "fraud_stack_model"
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        # Create model version (now the m-<uuid> folder exists)
        mv = client.create_model_version(
            name=model_name,
            source=os.path.join(mlflow.get_artifact_uri(), "model"),
            run_id=run.info.run_id
        )

        # Find the actual m-<uuid> folder
        model_folder = find_model_folder(exp_id, mv.version)
        if not model_folder:
            print(" Could not locate m-<uuid> folder for commit tag.")
        else:
            tags_dir = os.path.join(model_folder, "tags")
            if commit_hash:
                write_commit_tag(tags_dir, commit_hash)

            # Post-run verification
            if os.path.isdir(model_folder):
                print(f" Verified model folder: {model_folder}")
            else:
                print(" Model folder not found after creation.")

if __name__ == "__main__":
    register_model()