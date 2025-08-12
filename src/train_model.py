# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y 

def evaluate_model(y_test, y_pred, y_proba):
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n ROC-AUC Score: {roc_auc:.4f}")
    return roc_auc

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
        n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    meta_clf = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    stack_model.fit(X_train_res, y_train_res)
    y_pred = stack_model.predict(X_test_scaled)
    y_proba = stack_model.predict_proba(X_test_scaled)[:, 1]
    evaluate_model(y_test, y_pred, y_proba)

    # Save artifacts for registration step
    joblib.dump(stack_model, "models/stack_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n Training complete. Artifacts saved to 'models/'.")

if __name__ == "__main__":
    train()