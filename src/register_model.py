import os
import mlflow
import joblib
import tempfile
from mlflow.tracking import MlflowClient

def register_model():
    # === Config ===
    experiment_name = "fraud_detection_stack"
    model_name = "fraud_stack_model"
    model_file = "models/stack_model.pkl"
    git_commit_hash = "YOUR_COMMIT_HASH"  # Replace or fetch dynamically

    # === Ensure experiment exists ===
    mlflow.set_experiment(experiment_name)

    # === End any active run before starting ===
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        print(f"[INFO] Started run {run_id} in experiment {exp_id}")

        # === Load model ===
        stack_model = joblib.load(model_file)

        # === Save model to a temp dir ===
        saved_path = os.path.join(tempfile.mkdtemp(), "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # === Pre-log artifacts from saved_path (no file name assumptions) ===
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
            print(f"[INFO] Pre-logged all artifacts from: {saved_path}")
        except Exception as e:
            print(f"[WARN] Could not pre-log artifacts from saved_path: {e}")

        # === Create registered model if needed ===
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
            print(f"[INFO] Created registered model '{model_name}'")
        except Exception:
            print(f"[INFO] Registered model '{model_name}' already exists.")

        # === Create model version ===
        mv = client.create_model_version(
            name=model_name,
            source=saved_path,
            run_id=run_id
        )
        print(f"[INFO] Created model version {mv.version} (source={saved_path})")

        # === Locate model folder for tagging ===
        model_folder = None
        models_dir = os.path.join("mlruns", exp_id, "models")
        if os.path.exists(models_dir):
            for folder in os.listdir(models_dir):
                if folder.startswith("m-"):
                    model_folder = os.path.join(models_dir, folder)
                    break

        if not model_folder:
            fallback_name = f"m-fallback-{run_id[:8]}"
            model_folder = os.path.join(models_dir, fallback_name)
            print(f"[INFO] Using fallback model folder: {model_folder}")
            os.makedirs(model_folder, exist_ok=True)

        # === Write commit tag ===
        tags_dir = os.path.join(model_folder, "tags")
        os.makedirs(tags_dir, exist_ok=True)
        with open(os.path.join(tags_dir, "mlflow.source.git.commit"), "w") as f:
            f.write(git_commit_hash)
        print(f"[OK] Wrote commit tag to: {tags_dir}")

        print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()