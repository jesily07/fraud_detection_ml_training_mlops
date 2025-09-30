# The best tuned model is retrained and saved as models/stack_model_optuna.pkl + models/test_data_optuna.pkl, ready for downstream evaluation/registration.
# We don’t need to rerun train_model.py separately.

# optimize_model.py

import os
import joblib
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

# ----------------------------
# Load & preprocess data
# ----------------------------
def load_data(filepath="data/creditcard.csv"):
    df = pd.read_csv(filepath)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

# ----------------------------
# Objective function for Optuna
# ----------------------------
def objective(trial):
    X, y = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Hyperparameters to tune
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
    rf_max_depth = trial.suggest_int("rf_max_depth", 3, 15)

    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 15)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)

    # Define models
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=3,
        n_jobs=-1
    )

    stack_model.fit(X_train_res, y_train_res)
    y_pred_proba = stack_model.predict_proba(X_valid_scaled)[:, 1]

    roc_auc = roc_auc_score(y_valid, y_pred_proba)

    return roc_auc

# ----------------------------
# Main optimization function
# ----------------------------
def optimize():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="optuna_stacking_optimization"):
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        best_score = study.best_value

        print("\n[INFO] Best ROC-AUC: ", best_score)
        print("[INFO] Best Parameters: ", best_params)

        # Log results to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_roc_auc", best_score)

        # ------------------------
        # Retrain tuned model
        # ------------------------
        rf = RandomForestClassifier(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_max_depth"],
            random_state=42
        )
        xgb = XGBClassifier(
            n_estimators=best_params["xgb_n_estimators"],
            max_depth=best_params["xgb_max_depth"],
            learning_rate=best_params["xgb_learning_rate"],
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        meta_clf = LogisticRegression(max_iter=1000)

        stack_model = StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            final_estimator=meta_clf,
            cv=3,
            n_jobs=-1
        )

        stack_model.fit(X_train_res, y_train_res)

        # ------------------------
        # Save tuned model + test data
        # ------------------------
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "stack_model_optuna.pkl")
        test_data_path = os.path.join("models", "test_data_optuna.pkl")

        joblib.dump(stack_model, model_path)
        joblib.dump((X_test_scaled, y_test), test_data_path)

        print(f"[INFO] Tuned model saved to: {model_path}")
        print(f"[INFO] Test data saved to: {test_data_path}")

        # Save study for reproducibility
        study_path = os.path.join("models", "optuna_study.pkl")
        joblib.dump(study, study_path)
        mlflow.log_artifact(study_path, artifact_path="optuna")

        print(f"[INFO] Optuna study saved to: {study_path}")


if __name__ == "__main__":
    optimize()