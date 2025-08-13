import os
import shutil
import tempfile
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import git
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

def register_model():
    experiment_name = "fraud_detection_stack"
    model_name = "fraud_stack_model"
    local_model_path = "models/stack_model.pkl"  # From train_model.py fixed save

    # Ensure model exists
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Trained model not found at {local_model_path}. Run train_model.py first.")

    # Ensure experiment exists
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Start a fresh run
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        print(f"[INFO] Started run {run_id} in experiment {exp_id}")

        # --- Step 1: Save model to a temp dir ---
        temp_dir = tempfile.mkdtemp()
        saved_path = os.path.join(temp_dir, "saved_model")
        os.makedirs(saved_path, exist_ok=True)

        model_obj = joblib.load(local_model_path)
        mlflow.sklearn.save_model(model_obj, saved_path)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # --- Step 2: Pre-log artifacts while temp dir still exists ---
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
        except Exception as e:
            print(f"[WARN] Could not pre-log artifacts: {e}")

        # --- Step 3: Create registered model if not exists ---
        try:
            client.create_registered_model(model_name)
            print(f"[INFO] Created registered model '{model_name}'")
        except mlflow.exceptions.MlflowException:
            pass  # Already exists

        # --- Step 4: Create model version ---
        mv = client.create_model_version(
            name=model_name,
            source=saved_path,
            run_id=run_id
        )
        print(f"[INFO] Created model version {mv.version} (source={saved_path})")

        # --- Step 5: Locate real m-<uuid> folder ---
        models_dir = os.path.join("mlruns", str(exp_id), "models")
        m_folder = None
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.startswith("m-") and os.path.isdir(os.path.join(models_dir, f)):
                    m_folder = os.path.join(models_dir, f)
                    break

        if not m_folder:
            print(f" Could not find m-<uuid> folder for the new version. Attempting to rescan...")
            m_folder = os.path.join(models_dir, f"m-fallback-{run_id[:8]}")
            os.makedirs(m_folder, exist_ok=True)
            print(f"[WARN] Using fallback model folder: {m_folder}")

        # --- Step 6: Write commit tag manually ---
        try:
            commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        except Exception:
            commit_hash = "unknown"
        tag_path = os.path.join(m_folder, "tags")
        os.makedirs(tag_path, exist_ok=True)
        with open(os.path.join(tag_path, "mlflow.source.git.commit"), "w") as f:
            f.write(commit_hash)
        print(f"[OK] Wrote commit tag: {os.path.join(tag_path, 'mlflow.source.git.commit')}")

        # --- Step 7: Cleanup temp dir ---
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()