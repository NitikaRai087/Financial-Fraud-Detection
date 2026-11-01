from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Financial Fraud Detection Dashboard")
st.caption("Machine-learning transaction risk analysis | Educational portfolio project")

MODEL_PATH = Path("models/fraud_detection_pipeline.joblib")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

bundle = load_bundle()
if bundle is None:
    st.warning("No trained model found. Train it first using the command in the README.")
    st.stop()

model = bundle["model"]
features = bundle["feature_columns"]
st.sidebar.header("Prediction settings")
threshold = st.sidebar.slider("Fraud alert threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.write("Selected model:", bundle["model_name"])

overview, batch, single = st.tabs(["Overview", "Batch prediction", "Single transaction"])

with overview:
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
    st.write("Enter feature values for one transaction. This form uses generic inputs; for best results, provide values in the same format as your training data.")
    with st.form("single_transaction"):
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                values[feature] = st.text_input(feature, "") if feature in ("type", "Acct type", "Time of day", "branch") else st.number_input(feature, value=0.0)
        submitted = st.form_submit_button("Analyse transaction")
    if submitted:
        row = pd.DataFrame([values], columns=features)
        for col in row.columns:
            if row[col].dtype == object:
                row[col] = row[col].replace("", np.nan)
        probability = float(model.predict_proba(row)[0, 1])
        flagged = probability >= threshold
        a, b = st.columns(2)
        a.metric("Fraud probability", f"{probability:.2%}")
        b.metric("Prediction", "Flag for review" if flagged else "Below threshold")
        st.warning("A model flag is a risk indicator, not proof of fraud.") if flagged else st.info("A low score does not guarantee that the transaction is legitimate.")

st.divider()
st.caption("Educational use only. Do not use model outputs as the sole basis for financial decisions.")
