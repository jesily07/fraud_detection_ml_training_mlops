import os
import tempfile
import mlflow
import mlflow.sklearn
import joblib
import subprocess

def register_model():
    # === Setup experiment ===
    experiment_name = "fraud_detection_stack"
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    print(f"[INFO] Started run {run.info.run_id} in experiment {run.info.experiment_id}")

    # === Load trained model ===
    model_dir = "models"
    stack_model_path = os.path.join(model_dir, "stack_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(stack_model_path):
        raise FileNotFoundError(f"Stack model not found at {stack_model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    stack_model = joblib.load(stack_model_path)
    scaler = joblib.load(scaler_path)

    # === Save model to temp directory ===
    saved_path = os.path.join(tempfile.mkdtemp(), "saved_model")
    mlflow.sklearn.save_model(stack_model, saved_path)
    print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

    # === Pre-log artifacts directly from saved_path BEFORE registry ===
    try:
        mlflow.log_artifacts(saved_path, artifact_path="manual_model")
        print(f"[INFO] Pre-logged all artifacts from: {saved_path}")
    except Exception as e:
        print(f"[WARN] Could not pre-log artifacts from saved_path: {e}")

    # === Auto-capture git commit hash ===
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"
    print(f"[INFO] Using commit hash: {commit_hash}")

    # === Register model ===
    model_name = "fraud_stack_model"
    mv = mlflow.register_model(
        model_uri=f"file://{saved_path}",
        name=model_name
    )

    # === Tag model version with commit hash ===
    mv_folder = os.path.join("mlruns", str(run.info.experiment_id), "models", f"m-{mv.version}")
    fallback_folder = os.path.join("mlruns", str(run.info.experiment_id), "models", f"m-fallback-{mv.version}")

    if os.path.exists(mv_folder):
        tag_path = os.path.join(mv_folder, "tags", "mlflow.source.git.commit")
    else:
        tag_path = os.path.join(fallback_folder, "tags", "mlflow.source.git.commit")

    os.makedirs(os.path.dirname(tag_path), exist_ok=True)
    with open(tag_path, "w") as f:
        f.write(commit_hash)
    print(f"[OK] Wrote commit tag ({commit_hash}) to: {tag_path}")

    mlflow.end_run()
    print("[DONE] register_model finished.")

if __name__ == "__main__":
    register_model()