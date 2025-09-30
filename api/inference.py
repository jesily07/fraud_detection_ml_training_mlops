# inference.py

import os
import joblib
import mlflow
import pandas as pd
import argparse

# ----------------------------
# Config
# ----------------------------
MODEL_BASE = "models/stack_model.pkl"
MODEL_OPTUNA = "models/stack_model_optuna.pkl"
MLFLOW_BASE_URI = "models:/fraud_stack_model/Production"
MLFLOW_OPTUNA_URI = "models:/fraud_stack_model_optuna/Production"

# ----------------------------
# Global cache
# ----------------------------
_cached_model = None
_cached_model_type = None


def load_model():
    """
    Load the fraud detection model once (cached).
    Priority: MLflow Production registry → local optuna → local base.
    """
    global _cached_model, _cached_model_type

    if _cached_model is not None:
        return _cached_model

    # Try MLflow (optuna model first)
    try:
        print("[INFO] Trying MLflow Production model load (optuna)...")
        _cached_model = mlflow.sklearn.load_model(MLFLOW_OPTUNA_URI)
        _cached_model_type = "mlflow_optuna"
        print("[INFO] Loaded Production model from MLflow (optuna).")
        return _cached_model
    except Exception as e1:
        print(f"[WARN] MLflow optuna load failed: {e1}")

    # Try MLflow (base model)
    try:
        print("[INFO] Trying MLflow Production model load (base)...")
        _cached_model = mlflow.sklearn.load_model(MLFLOW_BASE_URI)
        _cached_model_type = "mlflow_base"
        print("[INFO] Loaded Production model from MLflow (base).")
        return _cached_model
    except Exception as e2:
        print(f"[WARN] MLflow base load failed: {e2}")

    # Fallback: local optuna
    if os.path.exists(MODEL_OPTUNA):
        print(f"[INFO] Falling back to local optuna model: {MODEL_OPTUNA}")
        _cached_model = joblib.load(MODEL_OPTUNA)
        _cached_model_type = "local_optuna"
        return _cached_model

    # Fallback: local base
    if os.path.exists(MODEL_BASE):
        print(f"[INFO] Falling back to local base model: {MODEL_BASE}")
        _cached_model = joblib.load(MODEL_BASE)
        _cached_model_type = "local_base"
        return _cached_model

    raise FileNotFoundError("No available model found (MLflow or local).")


def run_inference(model, X: pd.DataFrame):
    """
    Run inference on a pandas DataFrame and return fraud probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
        return preds


# ----------------------------
# CLI entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, help="Path to input CSV for inference")
    args = parser.parse_args()

    model = load_model()

    if args.input_csv:
        print(f"[INFO] Using input CSV for inference: {args.input_csv}")
        X = pd.read_csv(args.input_csv)
    else:
        # fallback: use small test dataset
        print("[INFO] Using sample data for inference (10 rows)")
        if os.path.exists("models/test_data_optuna.pkl"):
            X, _ = joblib.load("models/test_data_optuna.pkl")
        elif os.path.exists("models/test_data.pkl"):
            X, _ = joblib.load("models/test_data.pkl")
        else:
            raise FileNotFoundError("No test dataset found in models/.")
        X = X[:10]

    preds = run_inference(model, X)

    os.makedirs("artifacts_tmp", exist_ok=True)
    out_path = os.path.join("artifacts_tmp", "inference_output.csv")
    pd.DataFrame({"fraud_probability": preds}).to_csv(out_path, index=False)

    print(f"[INFO] Inference complete. Predictions saved to {out_path}")
    print(pd.DataFrame({"fraud_probability": preds}).head())


if __name__ == "__main__":
    main()