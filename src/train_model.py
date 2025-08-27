# train_model.py +  MLflow logging + confusion matrix artifact + ROC Curve

# train_model.py
import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

mlflow.set_experiment("fraud_detection_stack")

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def log_confusion_matrix(y_true, y_pred, run_id):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    os.makedirs("artifacts_tmp", exist_ok=True)
    cm_path = f"artifacts_tmp/confusion_matrix_{run_id}.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

def log_roc_curve(y_true, y_proba, run_id):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    os.makedirs("artifacts_tmp", exist_ok=True)
    roc_path = f"artifacts_tmp/roc_curve_{run_id}.png"
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="plots")

def train():
    df = load_data("data/creditcard.csv")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1,
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        stack_model.fit(X_train_res, y_train_res)

        y_pred = stack_model.predict(X_test_scaled)
        y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(
            f"[INFO] Metrics logged: Accuracy={acc:.4f}, "
            f"Precision={prec:.4f}, Recall={rec:.4f}, "
            f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save & log models
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "stack_model.pkl")
        scaler_path = os.path.join("models", "scaler.pkl")
        joblib.dump(stack_model, model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.sklearn.log_model(stack_model, "model")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.log_artifact(scaler_path, artifact_path="models")

        # Log plots
        log_confusion_matrix(y_test, y_pred, run_id)
        log_roc_curve(y_test, y_proba, run_id)

        print("\n[INFO] Training complete. Artifacts logged in MLflow.")

if __name__ == "__main__":
    train()