# explain_model.py

import os
import joblib
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
import json


# ----------------------------
# Config
# ----------------------------
MODEL_BASE = "models/stack_model.pkl"
MODEL_OPTUNA = "models/stack_model_optuna.pkl"
MLFLOW_TUNED_URI = "models:/fraud_stack_model_optuna/latest"
MLFLOW_BASE_URI = "models:/fraud_stack_model/latest"


# ----------------------------
# Explain meta-model (LogReg on meta-features)
# ----------------------------
def explain_meta_model(model, X_test, model_type):
    paths = []

    # Meta features (outputs from base learners)
    X_meta = model.transform(X_test)

    # Build readable feature names (e.g. rf_prob, xgb_prob)
    base_names = [name for name, _ in model.estimators]
    meta_feature_names = [f"{name}_prob" for name in base_names]

    # Wrap into DataFrame for shap plots with labels
    X_meta_df = pd.DataFrame(X_meta, columns=meta_feature_names)

    est = model.final_estimator_

    # Try SHAP first
    try:
        explainer = shap.LinearExplainer(est, X_meta_df)
        shap_values = explainer(X_meta_df)

        # SHAP summary plots
        summary_path = os.path.join("artifacts_tmp", f"shap_summary_meta_{model_type}.png")
        plt.figure()
        shap.summary_plot(shap_values, X_meta_df, show=False)
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()

        bar_path = os.path.join("artifacts_tmp", f"shap_bar_meta_{model_type}.png")
        plt.figure()
        shap.summary_plot(shap_values, X_meta_df, plot_type="bar", show=False)
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()

        paths.extend([summary_path, bar_path])

        # Compute mean abs SHAP values for feature importance
        shap_vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=meta_feature_names)

    except Exception as e:
        print(f"[WARN] SHAP failed for meta-model: {e}")
        print("[INFO] Falling back to coef × mean importance...")

        if hasattr(est, "coef_"):
            coef = est.coef_.flatten()
            mean_vals = X_meta_df.mean().values
            importances = np.abs(coef * mean_vals)
            feature_importance = pd.Series(importances, index=meta_feature_names)
        else:
            raise RuntimeError("Final estimator has no coef_ — cannot compute fallback importance.")

    # Sort and take top 5
    top5 = feature_importance.sort_values(ascending=False).head(5)

    # Save JSON
    summary_json = os.path.join("artifacts_tmp", f"shap_top5_meta_{model_type}.json")
    with open(summary_json, "w") as f:
        json.dump(top5.to_dict(), f, indent=2)
    paths.append(summary_json)

    # Save Markdown
    summary_md = os.path.join("artifacts_tmp", f"shap_top5_meta_{model_type}.md")
    with open(summary_md, "w") as f:
        f.write("# Top 5 Most Important Meta-Features (SHAP)\n\n")
        for feat, val in top5.items():
            f.write(f"- **{feat}**: {val:.4f}\n")
    paths.append(summary_md)

    return paths, top5


# ----------------------------
# Explain base learners individually (optional)
# ----------------------------
def explain_base_learners(model, X_test, model_type):
    paths = []
    for i, est in enumerate(model.estimators_):
        try:
            if hasattr(est, "predict_proba"):  # RF, XGB
                explainer = shap.TreeExplainer(est)
                shap_values = explainer(X_test)

                out_path = os.path.join("artifacts_tmp", f"shap_base_{i}_{type(est).__name__}_{model_type}.png")
                plt.figure()
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()

                paths.append(out_path)
        except Exception as e:
            print(f"[WARN] Skipping base learner {i} ({type(est).__name__}): {e}")
    return paths


# ----------------------------
# Robust model loader
# ----------------------------
def load_model_robust(path_local, mlflow_uri):
    try:
        print(f"[INFO] Trying local joblib load: {path_local}")
        return joblib.load(path_local)
    except Exception as e1:
        print(f"[WARN] Local load failed: {e1}")
        try:
            print(f"[INFO] Falling back to MLflow model load: {mlflow_uri}")
            return mlflow.sklearn.load_model(mlflow_uri)
        except Exception as e2:
            raise RuntimeError(f"Both local and MLflow model loading failed.\nLocal error: {e1}\nMLflow error: {e2}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain-all", action="store_true", help="Also explain base learners (RF, XGB)")
    args = parser.parse_args()

    os.makedirs("artifacts_tmp", exist_ok=True)

    # Detect which model to use
    if os.path.exists(MODEL_OPTUNA):
        print("[INFO] Using optuna model for explainability")
        model = load_model_robust(MODEL_OPTUNA, MLFLOW_TUNED_URI)
        model_type = "optuna"
    elif os.path.exists(MODEL_BASE):
        print("[INFO] Using base model for explainability")
        model = load_model_robust(MODEL_BASE, MLFLOW_BASE_URI)
        model_type = "base"
    else:
        raise FileNotFoundError("No model found in models/. Run train_model.py or optimize_model.py first.")

    # Load test data
    if model_type == "optuna":
        X_test, y_test = joblib.load("models/test_data_optuna.pkl")
    else:
        X_test, y_test = joblib.load("models/test_data.pkl")

    X_subset = X_test[:200]  # subset for speed

    # Default → explain meta-model only
    print(f"[INFO] Explaining meta-model (LogReg) of {model_type} stack...")
    paths, top5 = explain_meta_model(model, X_subset, model_type)

    # Optional → explain base learners too
    if args.explain_all:
        print("[INFO] Explaining base learners too...")
        paths.extend(explain_base_learners(model, X_subset, model_type))

    # Log artifacts + top5 to MLflow
    with mlflow.start_run(run_name=f"explain_{model_type}_stack_model"):
        for p in paths:
            mlflow.log_artifact(p, artifact_path="shap")

        # Log Top 5 features as params and metrics
        for rank, (feat, val) in enumerate(top5.items(), start=1):
            mlflow.log_param(f"top{rank}_feature", feat)
            mlflow.log_metric(f"top{rank}_importance", float(val))

    print(f"[INFO] SHAP explanations complete for {model_type} model. Artifacts + Top 5 logged to MLflow.")


if __name__ == "__main__":
    main()