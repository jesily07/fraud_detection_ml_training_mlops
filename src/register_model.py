# register_model.py
import os
import subprocess
import joblib
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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

def find_model_uuid_by_version(exp_id, version):
    models_dir = os.path.join("mlruns", str(exp_id), "models")
    for folder in os.listdir(models_dir):
        meta_path = os.path.join(models_dir, folder, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            if str(meta.get("version")) == str(version):
                return folder
    return None

def write_commit_tag(exp_id, model_uuid, commit_hash):
    tag_dir = os.path.join("mlruns", str(exp_id), "models", model_uuid, "tags")
    os.makedirs(tag_dir, exist_ok=True)
    with open(os.path.join(tag_dir, "mlflow.source.git.commit"), "w", encoding="utf-8") as f:
        f.write(commit_hash)

def register():
    mlflow.set_experiment("fraud_detection_stack")
    client = MlflowClient()

    with mlflow.start_run() as run:
        # Load trained model
        model_path = "models/stack_model.pkl"
        scaler_path = "models/scaler.pkl"
        stack_model = joblib.load(model_path)

        # Get commit hash
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        # Pre-create model version
        try:
            client.create_registered_model("fraud_stack_model")
        except:
            pass  # already exists

        mv = client.create_model_version(
            name="fraud_stack_model",
            source=os.path.join(mlflow.get_artifact_uri(), "model"),
            run_id=run.info.run_id
        )

        # Find real m-<uuid>
        model_uuid_folder = find_model_uuid_by_version(run.info.experiment_id, mv.version)
        if model_uuid_folder:
            if commit_hash:
                write_commit_tag(run.info.experiment_id, model_uuid_folder, commit_hash)
            print(f" Commit tag written to: {model_uuid_folder}")
        else:
            print(" Could not determine model folder for commit tag.")

        # Log model AFTER commit tag is there
        mlflow.sklearn.log_model(stack_model, artifact_path="model")
        print(" Model registered in MLflow.")

if __name__ == "__main__":
    register()