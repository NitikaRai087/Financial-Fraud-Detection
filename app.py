from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.fraud_pipeline import resolve_target, train_and_evaluate, save_artifacts

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Financial Fraud Detection Dashboard")
st.caption("Machine-learning transaction risk analysis | Educational portfolio project")

MODEL_PATH = Path("models/fraud_detection_pipeline.joblib")

# Load a saved model once per Streamlit session.
if "fraud_bundle" not in st.session_state:
    st.session_state.fraud_bundle = (
        joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    )

bundle = st.session_state.fraud_bundle

if bundle is None:
    st.info("The dashboard is ready. Upload a labelled transaction dataset below to train your first model.")
    st.subheader("Step 1 — Upload training data")
    training_file = st.file_uploader(
        "Choose your labelled transaction CSV",
        type=["csv"],
        key="training_csv",
        help="Your CSV must contain a binary fraud target column such as isFraud or isFraud1."
    )
    if training_file is None:
        st.markdown(
            "**Required target format:** 0 = legitimate transaction, 1 = fraudulent transaction. "
            "The dataset must contain examples of both classes."
        )
        st.stop()

    try:
        training_df = pd.read_csv(training_file)
        st.write(f"Dataset loaded: **{len(training_df):,} rows** and **{len(training_df.columns):,} columns**.")
        suggested = next((c for c in ("isFraud", "isFraud1") if c in training_df.columns), None)
        target_options = list(training_df.columns)
        default_index = target_options.index(suggested) if suggested else 0
        target_col = st.selectbox("Select the fraud target column", target_options, index=default_index)
        st.dataframe(training_df.head(8), use_container_width=True)

        if st.button("Train fraud detection model", type="primary"):
            with st.spinner("Training and evaluating Logistic Regression and Random Forest..."):
                result = train_and_evaluate(training_df, target_col)
                save_artifacts(result)
                st.session_state.fraud_bundle = joblib.load(MODEL_PATH)
                st.session_state.training_metrics = result["metrics"]
            st.success(f"Training complete. Best model: {result['model_name']}")
            st.rerun()
    except Exception as exc:
        st.error(f"Could not read or train from this dataset: {exc}")
    st.stop()

model = bundle["model"]
features = bundle["feature_columns"]
st.sidebar.header("Prediction settings")
threshold = st.sidebar.slider("Fraud alert threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.write("Selected model:", bundle["model_name"])
if st.sidebar.button("Train a different model"):
    st.session_state.fraud_bundle = None
    st.rerun()

overview, batch, single = st.tabs(["Overview", "Batch prediction", "Single transaction"])

with overview:
    metrics = st.session_state.get("training_metrics")
    if metrics:
        st.subheader("Model evaluation")
        metric_cols = st.columns(4)
        best = bundle["model_name"]
        for col, metric, label in zip(
            metric_cols, ("precision", "recall", "f1_score", "roc_auc"),
            ("Precision", "Recall", "F1-score", "ROC-AUC")
        ):
            col.metric(label, f"{metrics[best][metric]:.3f}")
    file = st.file_uploader("Upload a transaction CSV for overview", type=["csv"], key="overview")
    if file:
        df = pd.read_csv(file)
        target = bundle["target"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{len(df):,}")
        if target in df.columns:
            frauds = int(pd.to_numeric(df[target], errors="coerce").fillna(0).sum())
            c2.metric("Fraud cases", f"{frauds:,}")
            c3.metric("Fraud rate", f"{frauds / len(df):.2%}" if len(df) else "—")
            counts = df[target].value_counts().rename_axis("Class").reset_index(name="Transactions")
            st.plotly_chart(px.bar(counts, x="Class", y="Transactions", title="Fraud vs non-fraud"), use_container_width=True)
        if "amount" in df.columns:
            st.plotly_chart(px.histogram(df, x="amount", nbins=40, title="Transaction amount distribution"), use_container_width=True)
        if "type" in df.columns:
            counts = df.groupby("type", dropna=False).size().reset_index(name="Transactions")
            st.plotly_chart(px.bar(counts, x="type", y="Transactions", title="Transactions by type"), use_container_width=True)
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Upload your transaction dataset to explore its fraud distribution and transaction patterns.")

with batch:
    st.write("Upload a CSV with the same feature columns used during model training.")
    file = st.file_uploader("Choose CSV for batch scoring", type=["csv"], key="batch")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error("Missing model input columns: " + ", ".join(missing))
        else:
            result = df.copy()
            probabilities = model.predict_proba(df[features])[:, 1]
            result["fraud_probability"] = probabilities
            result["predicted_fraud"] = (probabilities >= threshold).astype(int)
            result["risk_status"] = np.where(result["predicted_fraud"] == 1, "Review", "Below threshold")
            st.metric("Transactions flagged for review", f"{int(result['predicted_fraud'].sum()):,}")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                               "fraud_predictions.csv", "text/csv")
            st.plotly_chart(px.histogram(result, x="fraud_probability", nbins=30,
                                         title="Predicted fraud probability"), use_container_width=True)

with single:
    st.write("Enter feature values for one transaction. For categorical fields, enter the same type of value used in your training data.")
    with st.form("single_transaction"):
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                if feature in ("type", "Acct type", "Time of day", "branch"):
                    values[feature] = st.text_input(feature, "")
                else:
                    values[feature] = st.number_input(feature, value=0.0)
        submitted = st.form_submit_button("Analyse transaction")
    if submitted:
        row = pd.DataFrame([values], columns=features)
        for col in row.columns:
            if row[col].dtype == object:
                row[col] = row[col].replace("", np.nan)
        try:
            probability = float(model.predict_proba(row)[0, 1])
            flagged = probability >= threshold
            a, b = st.columns(2)
            a.metric("Fraud probability", f"{probability:.2%}")
            b.metric("Prediction", "Flag for review" if flagged else "Below threshold")
            if flagged:
                st.warning("A model flag is a risk indicator, not proof of fraud.")
            else:
                st.info("A low score does not guarantee that the transaction is legitimate.")
        except Exception as exc:
            st.error(f"Could not score this transaction. Check that all inputs match the training data. Details: {exc}")

st.divider()
st.caption("Educational use only. Do not use model outputs as the sole basis for financial decisions.")
