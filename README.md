# Financial Fraud Detection Using Machine Learning

An end-to-end portfolio project for analysing financial transactions, training fraud-classification models, evaluating results, and exploring predictions in a Streamlit dashboard.

## Workflow
1. Load a transaction CSV.
2. Inspect class balance and clean missing/infinite values.
3. Preprocess numeric and categorical features.
4. Train Logistic Regression and Random Forest pipelines.
5. Compare precision, recall, F1-score and ROC-AUC.
6. Save the selected pipeline and evaluation reports.
7. Explore transactions and score CSV uploads in Streamlit.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib.

## Project structure
```
Financial-Fraud-Detection/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
├── reports/
└── src/
    ├── __init__.py
    └── fraud_pipeline.py
```

## Dataset
The training script accepts a CSV file. The binary target should be named `isFraud` or `isFraud1`, or specified with `--target`. Expected transaction features may include `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `step`, `branch`, and `Acct type`. Identifier fields such as `nameOrig` and `nameDest` are excluded.

Place your authorized dataset in `data/`. Do not commit private banking information or personally identifiable data.

## Setup on Windows
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train
```powershell
python train.py --data "data/Fraud Detection.csv" --target isFraud
```
If the target is `isFraud1`, use:
```powershell
python train.py --data "data/Fraud Detection.csv" --target isFraud1
```

Training generates:
- `models/fraud_detection_pipeline.joblib`
- `reports/model_metrics.json`
- `reports/classification_report.txt`

## Run dashboard
```powershell
streamlit run app.py
```
Open the local URL shown in the terminal, usually http://localhost:8501.

## Evaluation
Fraud data is commonly imbalanced, so accuracy alone is insufficient. Review precision, recall, F1-score, ROC-AUC and the confusion matrix. A lower alert threshold may detect more fraud but can increase false positives.

## Limitations
This is an educational project, not a production banking control. Model metrics depend on the dataset and split used. For real deployment, use time-aware validation, check for data leakage, tune thresholds on validation data, monitor drift, and require human review. Model flags are risk indicators, not proof of fraud.

## Resume description
**Financial Fraud Detection Using Machine Learning** — Built a fraud-classification workflow using Python, Pandas and Scikit-learn, covering data preprocessing, exploratory analysis, model comparison and evaluation using precision, recall, F1-score and ROC-AUC. Developed a Streamlit dashboard for transaction-risk exploration and batch predictions.

Suggested repository name: `Financial-Fraud-Detection`.
