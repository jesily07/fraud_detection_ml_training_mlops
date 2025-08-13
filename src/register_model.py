import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import subprocess

def register_model():
    # === Start MLflow run ===
    mlflow.set_experiment("fraud_detection_stack")
    with mlflow.start_run() as run:
        print(f"[INFO] Started run {run.info.run_id} in experiment {run.info.experiment_id}")

        # === Load trained model ===
        stack_model = joblib.load("models/stack_model.pkl")

        # === Create temp path for model saving ===
        saved_path = os.path.join(tempfile.mkdtemp(), "saved_model")
        mlflow.sklearn.save_model(stack_model, saved_path)
        print(f"[INFO] Model saved locally to {saved_path} (ready for create_model_version)")

        # === Pre-log artifacts from saved_path BEFORE registration ===
        try:
            mlflow.log_artifacts(saved_path, artifact_path="manual_model")
            print(f"[INFO] Pre-logged all artifacts from: {saved_path}")
        except Exception as e:
            print(f"[WARN] Could not pre-log artifacts from saved_path: {e}")

        # === Create registered model if not exists ===
        client = mlflow.tracking.MlflowClient()
        model_name = "fraud_stack_model"
        try:
            client.create_registered_model(model_name)
            print(f"[INFO] Created registered model '{model_name}'")
        except mlflow.exceptions.RestException:
            print(f"[INFO] Registered model '{model_name}' already exists")

        # === Create new model version ===
        mv = client.create_model_version(
            name=model_name,
            source=saved_path,
            run_id=run.info.run_id
        )
        print(f"[INFO] Created model version {mv.version} (source={saved_path})")

        # === Auto-capture current Git commit ===
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            commit_tag_path = os.path.join(
                os.path.dirname(mv.source), "tags", "mlflow.source.git.commit"
            )
            os.makedirs(os.path.dirname(commit_tag_path), exist_ok=True)
            with open(commit_tag_path, "w") as f:
                f.write(commit_hash)
            print(f"[OK] Wrote commit tag ({commit_hash}) to: {commit_tag_path}")
        except Exception as e:
            print(f"[WARN] Could not capture Git commit hash: {e}")

if __name__ == "__main__":
    register_model()