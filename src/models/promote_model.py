# promote_model.py

import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

# ----------------------------
# Config
# ----------------------------
TUNED_MODEL_NAME = "fraud_stack_model_optuna"
BASE_MODEL_NAME = "fraud_stack_model"  # stays unpromoted
STAGE = "Staging"  # target stage for tuned model
THRESHOLD_STAGING = 0.98
THRESHOLD_PRODUCTION = 0.99  # auto-promote if above this


def promote_latest_tuned_model():
    client = MlflowClient()

    # ------------------------
    # Get all versions of tuned model
    # ------------------------
    tuned_versions = client.search_model_versions(f"name='{TUNED_MODEL_NAME}'")
    if not tuned_versions:
        print(f"[ERROR] No versions found for {TUNED_MODEL_NAME}. Run register_model.py first.")
        return

    # Pick latest version by timestamp
    latest_model = max(tuned_versions, key=lambda v: v.last_updated_timestamp)
    latest_version = int(latest_model.version)
    print(f"[INFO] Latest tuned model version: {latest_version}")

    # ------------------------
    # Try to fetch ROC-AUC from MLflow run
    # ------------------------
    run_id = getattr(latest_model, "run_id", None)
    roc_auc = None
    if run_id:
        try:
            run = client.get_run(run_id)
            metrics = run.data.metrics
            roc_auc = metrics.get("roc_auc")
            if roc_auc:
                print(f"[INFO] Retrieved ROC-AUC from MLflow run {run_id}: {roc_auc:.4f}")
        except Exception as e:
            print(f"[WARN] Could not fetch metrics from MLflow run: {e}")

    # ------------------------
    # Fallback: Recompute ROC-AUC using model + test_data.pkl
    # ------------------------
    if roc_auc is None:
        print("[FALLBACK] Recomputing ROC-AUC from local artifacts...")
        model_path = "models/stack_model_optuna.pkl"
        test_path = "models/test_data_optuna.pkl"

        if not os.path.exists(model_path) or not os.path.exists(test_path):
            print("[ERROR] Cannot recompute metrics: missing model or test data.")
            return

        model = joblib.load(model_path)
        X_test_scaled, y_test = joblib.load(test_path)

        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"[INFO] Recomputed ROC-AUC: {roc_auc:.4f}")

    # ------------------------
    # Check thresholds
    # ------------------------
    if roc_auc < THRESHOLD_STAGING:
        print(f"[SKIP] Model v{latest_version} did not meet staging threshold ({roc_auc:.4f} < {THRESHOLD_STAGING}).")
        return

    # Archive older tuned versions in Staging/Production
    tuned_versions = client.search_model_versions(f"name='{TUNED_MODEL_NAME}'")
    for v in tuned_versions:
        if int(v.version) != latest_version and v.current_stage in ["Staging", "Production"]:
            client.transition_model_version_stage(
                name=TUNED_MODEL_NAME,
                version=int(v.version),
                stage="Archived"
            )
            print(f"[ARCHIVE] Archived {TUNED_MODEL_NAME} v{v.version} (was in {v.current_stage})")

    # Decide promotion stage
    target_stage = "Production" if roc_auc >= THRESHOLD_PRODUCTION else "Staging"

    # Promote latest tuned model
    client.transition_model_version_stage(
        name=TUNED_MODEL_NAME,
        version=latest_version,
        stage=target_stage,
        archive_existing_versions=True
    )
    print(f"[PROMOTE] {TUNED_MODEL_NAME} v{latest_version} → {target_stage}")

    # Log stage decision into MLflow tags (if run_id available)
    if run_id:
        client.set_tag(run_id, "promotion_stage", target_stage)
        client.set_tag(run_id, "promotion_reason", f"ROC-AUC={roc_auc:.4f}")


if __name__ == "__main__":
    promote_latest_tuned_model()