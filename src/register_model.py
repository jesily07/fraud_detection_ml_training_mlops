# src/register_model.py
import os
import shutil
import tempfile
import subprocess
import joblib
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import warnings

# silence noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
EXPERIMENT = "fraud_detection_stack"
REGISTERED_MODEL_NAME = "fraud_stack_model"
LOCAL_MODEL_PICKLE = "models/stack_model.pkl"    # produced by train_model.py
LOCAL_SCALER_PICKLE = "models/scaler.pkl"
TEMP_SAVE_DIR = "temp_saved_model"               # temporary folder to save model before register

# Ensure MLflow doesn't keep an active run hanging
if mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment(EXPERIMENT)


def get_git_commit_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def save_model_locally(stack_model, save_dir):
    """Save model to an explicit local folder MLflow can use as 'source'."""
    # Remove any previous temp folder
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # Use mlflow.sklearn.save_model which writes MLflow-compatible model folder
    mlflow.sklearn.save_model(stack_model, path=save_dir)
    return save_dir


def find_model_folder_by_version(exp_id, version):
    """Scan mlruns/<exp_id>/models for m-<uuid> with meta.yaml version == version."""
    models_root = os.path.join("mlruns", str(exp_id), "models")
    if not os.path.isdir(models_root):
        return None
    for entry in os.listdir(models_root):
        if not entry.startswith("m-"):
            continue
        meta_path = os.path.join(models_root, entry, "meta.yaml")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            if str(meta.get("version")) == str(version):
                return os.path.join(models_root, entry)
        except Exception:
            continue
    return None


def write_commit_tag_to_folder(model_folder, commit_hash):
    tags_dir = os.path.join(model_folder, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    tag_file = os.path.join(tags_dir, "mlflow.source.git.commit")
    with open(tag_file, "w", encoding="utf-8") as f:
        f.write(commit_hash)
    return tag_file


def register_model():
    # Basic checks
    if not os.path.exists(LOCAL_MODEL_PICKLE):
        raise FileNotFoundError(f"Trained model not found: {LOCAL_MODEL_PICKLE}")

    # Load trained model (from train_model.py)
    stack_model = joblib.load(LOCAL_MODEL_PICKLE)

    client = MlflowClient()

    # Start fresh run
    with mlflow.start_run() as run:
        exp_id = run.info.experiment_id
        run_id = run.info.run_id
        print(f"[INFO] Started run {run_id} in experiment {exp_id}")

        # Save model to a local folder MLflow can register from
        saved_path = save_model_locally(stack_model, TEMP_SAVE_DIR)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # Create registered model (if missing)
        try:
            client.get_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            client.create_registered_model(REGISTERED_MODEL_NAME)
            print(f"[INFO] Created registered model '{REGISTERED_MODEL_NAME}'")

        # Create model version using local saved folder as source
        # IMPORTANT: using local path as source avoids calling mlflow.sklearn.log_model() internals
        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=os.path.abspath(saved_path),   # use absolute path
            run_id=run_id
        )
        print(f"[INFO] Created model version {mv.version} (source={saved_path})")

        # Find exact m-<uuid> folder by inspecting meta.yaml
        model_folder = find_model_folder_by_version(exp_id, mv.version)
        if model_folder is None:
            print("⚠ Could not find m-<uuid> folder for the new version. Attempting to rescan...")
            # try small rescans (timing)
            import time
            for _ in range(5):
                time.sleep(0.5)
                model_folder = find_model_folder_by_version(exp_id, mv.version)
                if model_folder:
                    break

        if model_folder is None:
            # Fallback: create a fallback folder under mlruns/<exp_id>/models/m-<runid>
            fallback_name = f"m-fallback-{run_id[:8]}"
            models_root = os.path.join("mlruns", str(exp_id), "models")
            os.makedirs(os.path.join(models_root, fallback_name), exist_ok=True)
            model_folder = os.path.join(models_root, fallback_name)
            print(f"[WARN] Using fallback model folder: {model_folder}")

        # Write commit tag
        commit_hash = get_git_commit_hash()
        if commit_hash:
            tag_file = write_commit_tag_to_folder(model_folder, commit_hash)
            print(f"[OK] Wrote commit tag: {tag_file}")
        else:
            print("⚠ No git commit hash found; skipping commit tag write.")

        # OPTIONAL: attach saved model folder as artifacts to run (no log_model)
        # This will copy the saved model dir into the run's artifact location.
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
            print(f"[INFO] Saved model artifacts logged to run under 'manual_model/'")
        except Exception as e:
            print(f"[WARN] Failed to log artifacts to run: {e}")

        # Verification
        commit_file_path = os.path.join(model_folder, "tags", "mlflow.source.git.commit")
        if os.path.exists(commit_file_path):
            print(f"[SUCCESS] Commit tag verified at: {commit_file_path}")
        else:
            print(f"[ERROR] Commit tag not found after write attempt: {commit_file_path}")

        # Clean up temp saved folder if desired
        try:
            shutil.rmtree(saved_path)
        except Exception:
            pass

    # End of run
    print("[DONE] register_model finished.")


if __name__ == "__main__":
    register_model()