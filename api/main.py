from fastapi import FastAPI
from pydantic import BaseModel, validator
import pandas as pd

# Import the module itself (so globals stay in sync)
import src.inference as inference

app = FastAPI(title="Fraud Detection API", version="0.2.1")


# ----------------------------
# Startup: preload model
# ----------------------------
@app.on_event("startup")
def startup_event():
    """
    Load the model into cache at API startup.
    Ensures /health shows correct metadata immediately.
    """
    try:
        inference.load_model()
        print(f"[INFO] Model preloaded at startup: {inference._cached_model_type}")
    except Exception as e:
        print(f"[WARN] Could not preload model at startup: {e}")


# ----------------------------
# Request schema
# ----------------------------
class Transaction(BaseModel):
    features: list[float]  # expects a numeric array of features (length = 30)

    @validator("features")
    def check_feature_length(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected 30 features, but got {len(v)}")
        return v


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    """
    Health check with basic model metadata.
    Returns status + model source, class, feature count, version.
    """
    model_type = inference._cached_model_type if inference._cached_model_type else "unloaded"

    if inference._cached_model is None:
        return {"status": "error", "model_source": model_type}

    metadata = {
        "status": "ok",
        "model_source": model_type,
        "model_class": type(inference._cached_model).__name__,
    }

    # Try to detect feature count
    try:
        if hasattr(inference._cached_model, "n_features_in_"):
            metadata["n_features"] = int(inference._cached_model.n_features_in_)
        else:
            metadata["n_features"] = "unknown"
    except Exception:
        metadata["n_features"] = "unknown"

    # Model version info
    if "mlflow" in model_type:
        metadata["model_version"] = "Production"
    else:
        metadata["model_version"] = "local"

    return metadata


@app.post("/predict")
def predict(transaction: Transaction):
    """
    Single prediction for a transaction.
    Input: JSON with numeric features array (length = 30)
    Output: Fraud probability
    """
    model = inference.load_model()  # cached after first load

    # Convert input features into a 1-row DataFrame
    X = pd.DataFrame([transaction.features])

    prob = inference.run_inference(model, X)[0]

    return {"fraud_probability": float(prob)}