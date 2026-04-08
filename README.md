# 💳 Fintech Transaction Anomaly Explorer

> **Live demo:** [your-app.streamlit.app](https://your-app.streamlit.app)  
> **Dataset:** [PaySim — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

An interactive Streamlit app for exploring fraud patterns across **6.3 million real PaySim transactions** and getting real-time fraud probability scores from a trained Random Forest model.

---

## Business context

Fraud detection teams need tools that let analysts quickly filter large transaction datasets, spot anomalies visually, and get model-backed probability scores on suspicious transactions — without writing code. This app does exactly that on the real PaySim dataset, which faithfully simulates a month of mobile money transactions including fraud behaviour observed in real financial systems.

**Key outcome:** An analyst can go from raw 6.3M-row data to a fraud probability on any transaction in under 30 seconds.

---

## Key findings from real data

From `01_eda.py` on the full 6.3M-row PaySim dataset:

| Finding | Detail |
|---|---|
| Fraud rate | ~0.13% — extreme class imbalance |
| Fraud transaction types | **TRANSFER and CASH_OUT only** — zero fraud in PAYMENT, DEBIT, CASH_IN |
| Balance drain signal | ~100% of fraud transactions drain origin account to exactly $0 |
| Amount skew | Fraud median amount ~6–10x higher than legitimate transactions |

---

## Model performance (real dataset)

| Model | AUC-ROC | Avg Precision | Fraud Recall |
|---|---|---|---|
| Logistic Regression | ~0.970 | ~0.55 | ~0.88 |
| **Random Forest** | **~0.997** | **~0.88** | **~0.93** |

> These are expected metrics on the real PaySim test set.  
> Actual results may vary slightly — re-run `03_train_model.py` to see yours.

**Top features by importance:**
1. `balance_drain_orig` — how much the origin account was drained
2. `orig_balance_mismatch` — inconsistency in balance updates (key fraud signal)
3. `balance_zero_after` — whether origin balance was wiped to $0
4. `log_amount` — log-transformed transaction amount

---

## Setup — step by step

### 1. Get the dataset
1. Go to **https://www.kaggle.com/datasets/ealaxi/paysim1**
2. Create a free Kaggle account and click **Download**
3. You'll get: `PS_20174392719_1491204439457_log.csv` (~470 MB)
4. Place it in the `data/` folder of this project

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
# EDA (optional — generates 5 plots in notebooks/eda_plots/)
python notebooks/01_eda.py

# Feature engineering (~2 min on 6.3M rows)
python notebooks/02_feature_engineering.py

# Train models (~5–10 min on real data)
python notebooks/03_train_model.py

# Launch the app
streamlit run app/streamlit_app.py
```

### 4. Deploy to Streamlit Community Cloud (free)
1. Push this repo to GitHub (**do not commit the CSV** — it's in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Connect your GitHub repo, set main file to `app/streamlit_app.py`
4. In **Secrets** or the file manager, upload your CSV to `data/`
5. Click Deploy

---

## Project structure

```
fraud-anomaly-detector/
├── data/
│   └── PS_20174392719_1491204439457_log.csv   ← download from Kaggle
├── notebooks/
│   ├── 01_eda.py                       # full EDA on 6.3M rows
│   ├── 02_feature_engineering.py       # feature pipeline + train/test split
│   ├── 03_train_model.py               # LR + RF training, evaluation plots
│   └── feature_engineering_utils.py   # shared utils (train/serve consistency)
├── models/
│   ├── fraud_model.pkl                 # best trained model
│   ├── model_meta.json                 # metrics + metadata
│   └── eval_plots/                     # ROC, PR, confusion matrix, importance
├── app/
│   └── streamlit_app.py               # interactive Streamlit application
├── .streamlit/
│   └── config.toml                    # theme settings
├── requirements.txt
├── .gitignore                         # excludes large CSV and model files
└── README.md
```

---

## App features

- **KPI cards** — total transactions, fraud count, fraud rate, fraud/legit volume
- **Anomaly scatter** — amount vs balance drain, log-scaled, stratified sample of 20k points
- **Type breakdown** — volume and fraud rate per transaction type
- **Amount distribution** — fraud vs legitimate histogram with 99th percentile cap
- **Hourly pattern** — transaction volume by hour of day
- **Live prediction form** — enter any transaction and get a fraud probability gauge + 5 risk signals
- **Filterable data table** — drill into raw transactions behind every chart

---

## Tech stack

`pandas` · `scikit-learn` · `streamlit` · `plotly` · `joblib` · `matplotlib` · `seaborn`

---

*Part of the [Madhushree Varatharaj](https://linkedin.com/in/madhushree-varatharaj) data science portfolio.*
