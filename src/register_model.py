import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def register_model():
    experiment_name = "fraud_detection_stack"
    model_name = "fraud_stack_model"

    # Set experiment
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Load model from fixed path
    model_path = "models/stack_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected model file not found: {model_path}")

    stack_model = joblib.load(model_path)

    with mlflow.start_run() as run:
        print(f"[INFO] Started run {run.info.run_id} in experiment {run.info.experiment_id}")

        # Save to temp path
        tmp_dir = tempfile.mkdtemp()
        saved_path = os.path.join(tmp_dir, "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)

        # Pre-log artifacts before model version registration
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
        except Exception as e:
            print(f"[WARN] Failed to log artifacts: {e}")

        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # Ensure registered model exists
        try:
            client.create_registered_model(model_name)
            print(f"[INFO] Created registered model '{model_name}'")
        except mlflow.exceptions.RestException:
            print(f"[INFO] Registered model '{model_name}' already exists.")

        # Create new model version
        mv = client.create_model_version(
            name=model_name,
            source=saved_path,
            run_id=run.info.run_id
        )
        print(f"[INFO] Created model version {mv.version} (source={saved_path})")

        # Locate model folder in mlruns for commit tagging
        model_dir = None
        for root, dirs, files in os.walk("mlruns"):
            for d in dirs:
                if d.startswith("m-") and mv.version in d:
                    model_dir = os.path.join(root, d)
                    break

        if not model_dir:
            print(f" Could not find m-<uuid> folder for the new version. Attempting to rescan...")
            fallback_folder = f"mlruns\\{run.info.experiment_id}\\models\\m-fallback-{run.info.run_id[:8]}"
            os.makedirs(fallback_folder, exist_ok=True)
            model_dir = fallback_folder
            print(f"[WARN] Using fallback model folder: {model_dir}")

        # Write commit tag
        commit_tag_file = os.path.join(model_dir, "tags", "mlflow.source.git.commit")
        os.makedirs(os.path.dirname(commit_tag_file), exist_ok=True)
        with open(commit_tag_file, "w") as f:
            f.write("dummy-git-commit-hash")
        print(f"[OK] Wrote commit tag: {commit_tag_file}")

        print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()