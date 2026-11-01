from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

POSITIVE_LABELS = {"1", "true", "yes", "y", "fraud", "fraudulent", "suspicious", "positive"}
NEGATIVE_LABELS = {"0", "false", "no", "n", "legitimate", "genuine", "normal", "non-fraud", "nonfraud", "negative"}

def resolve_target(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target '{requested}' not found. Columns: {list(df.columns)}")
        return requested
    for col in ("isFraud", "isFraud1", "is_fraud", "fraud", "fraud_status", "is_fraudulent"):
        if col in df.columns:
            return col
    raise ValueError(
        "No obvious fraud-label column was found. Your dataset must include a column "
        "that tells whether each transaction is fraudulent or legitimate."
    )

def prepare_features(df, target, positive_label=None):
    raw = df[target]
    if raw.isna().any():
        raise ValueError("The selected target column contains missing values. Remove or label those rows first.")
    unique_values = list(pd.unique(raw))
    if len(unique_values) != 2:
        preview = unique_values[:10]
        raise ValueError(
            f"Selected target '{target}' has {len(unique_values)} distinct values ({preview}). "
            "Choose a binary fraud-status column with exactly two classes."
        )

    if positive_label is not None:
        positive_text = str(positive_label).strip().casefold()
        y = raw.astype(str).str.strip().str.casefold().eq(positive_text).astype(int)
    else:
        normalized = {str(v).strip().casefold(): v for v in unique_values}
        positive_matches = [k for k in normalized if k in POSITIVE_LABELS]
        negative_matches = [k for k in normalized if k in NEGATIVE_LABELS]
        if len(positive_matches) == 1 and len(negative_matches) == 1:
            y = raw.astype(str).str.strip().str.casefold().eq(positive_matches[0]).astype(int)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().all() and set(numeric.unique()) == {0, 1}:
                y = numeric.astype(int)
            else:
                raise ValueError(
                    f"Target values are {unique_values}. Select which value means FRAUD in the training screen. "
                    "Do not choose transaction type, city, device, merchant category, or transaction ID as the target."
                )

    if y.nunique() != 2:
        raise ValueError("Training requires examples of both classes: fraud and non-fraud.")
    X = df.drop(columns=[target]).copy()
    X = X.drop(columns=[c for c in ("nameOrig", "nameDest", "transaction_id", "id") if c in X.columns])
    X = X.loc[:, ~X.columns.duplicated()]
    for col in X.select_dtypes(include="number").columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    if X.empty:
        raise ValueError("No usable feature columns remain.")
    return X, y.astype(int)

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers=transformers)

def train_and_evaluate(df, target, random_state=42, positive_label=None):
    X, y = prepare_features(df, target, positive_label=positive_label)
    if y.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for a train/test split. Add more labelled examples.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=random_state, n_jobs=-1
        )
    }
    pipelines, metrics = {}, {}
    for name, estimator in estimators.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator)
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
        }
        pipelines[name] = pipe
    best_name = max(metrics, key=lambda name: metrics[name]["f1_score"])
    best = pipelines[best_name]
    best_pred = (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    report = classification_report(
        y_test, best_pred, labels=[0, 1],
        target_names=["Non-fraud", "Fraud"], zero_division=0
    )
    return {
        "model": best, "model_name": best_name, "metrics": metrics,
        "best_metrics": metrics[best_name], "classification_report": report,
        "feature_columns": X.columns.tolist(), "target": target,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "fraud_rate": float(y.mean())
    }

def save_artifacts(result, model_path="models/fraud_detection_pipeline.joblib", report_dir="reports"):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": result["model"], "model_name": result["model_name"],
        "feature_columns": result["feature_columns"], "target": result["target"],
        "threshold": 0.5
    }, model_path)
    summary = {k: v for k, v in result.items() if k not in ("model", "classification_report")}
    Path(report_dir, "model_metrics.json").write_text(json.dumps(summary, indent=2))
    Path(report_dir, "classification_report.txt").write_text(result["classification_report"])
