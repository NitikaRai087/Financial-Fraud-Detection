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

def resolve_target(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target '{requested}' not found. Columns: {list(df.columns)}")
        return requested
    for col in ("isFraud", "isFraud1"):
        if col in df.columns:
            return col
    raise ValueError("Target not found. Expected isFraud or isFraud1; alternatively pass --target.")

def prepare_features(df, target):
    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only non-missing binary values: 0 and 1.")
    if y.nunique() != 2:
        raise ValueError("Training requires both classes: 0 (non-fraud) and 1 (fraud).")
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

def train_and_evaluate(df, target, random_state=42):
    X, y = prepare_features(df, target)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
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
